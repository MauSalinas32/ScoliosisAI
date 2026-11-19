"""
Robustness tests: inputs an examiner (or a hostile user) would use to make the service fail.
The contract is always the same: a clean 4xx/2xx answer, never a 500 and never a wrong silent result.
"""

import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image


def _png(arr: np.ndarray, mode: str) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def _headers(tokens, role="technician"):
    return {"Authorization": f"Bearer {tokens[role]}"}


class TestUnusualImages:
    @pytest.mark.parametrize("name,payload", [
        ("black.png", lambda: _png(np.zeros((300, 300), dtype=np.uint8), "L")),
        ("white.png", lambda: _png(np.full((300, 300), 255, dtype=np.uint8), "L")),
        ("noise.png", lambda: _png(np.random.default_rng(1).integers(0, 255, (300, 300), dtype=np.uint8), "L")),
        ("rgba.png", lambda: _png(np.full((300, 300, 4), 120, dtype=np.uint8), "RGBA")),
        ("sixteen_bit.png", lambda: _png(np.random.default_rng(2).integers(0, 65535, (300, 300), dtype=np.uint16), "I;16")),
        ("tiny.png", lambda: _png(np.zeros((8, 8), dtype=np.uint8), "L")),
        ("wide.png", lambda: _png(np.zeros((40, 4000), dtype=np.uint8), "L")),
    ])
    def test_upload_and_classify_never_return_500(self, client, auth_tokens, name, payload):
        up = client.post("/api/v1/images/upload", files={"file": (name, payload(), "image/png")},
                         headers=_headers(auth_tokens))
        assert up.status_code in (200, 413, 422), up.text
        if up.status_code == 200:
            cl = client.post(f"/api/v1/classify/{up.json()['image_id']}", headers=_headers(auth_tokens))
            assert cl.status_code in (200, 422), cl.text
            if cl.status_code == 200:
                probs = cl.json()["all_probabilities"].values()
                assert abs(sum(probs) - 1.0) < 0.01

    def test_truncated_jpeg_is_rejected_cleanly(self, client, auth_tokens, sample_xray_image):
        res = client.post("/api/v1/images/upload", files={"file": ("cut.jpg", sample_xray_image[: len(sample_xray_image) // 3], "image/jpeg")},
                          headers=_headers(auth_tokens))
        assert res.status_code in (200, 422)  # Pillow may decode a partial image; it must never crash

    def test_extension_and_content_mismatch(self, client, auth_tokens, sample_xray_image):
        res = client.post("/api/v1/images/upload", files={"file": ("looks_like.png", sample_xray_image, "image/png")},
                          headers=_headers(auth_tokens))
        assert res.status_code in (200, 422)


class TestIdentifiersAndState:
    @pytest.mark.parametrize("path", ["/api/v1/classify/999999", "/api/v1/images/999999", "/api/v1/results/999999"])
    def test_unknown_ids_return_404(self, client, auth_tokens, path):
        method = client.post if "classify" in path else client.get
        assert method(path, headers=_headers(auth_tokens, "radiologist")).status_code == 404

    @pytest.mark.parametrize("path", ["/api/v1/classify/abc", "/api/v1/images/-1", "/api/v1/results/1.5", "/api/v1/images/%00"])
    def test_malformed_ids_are_rejected_not_crashed(self, client, auth_tokens, path):
        method = client.post if "classify" in path else client.get
        assert method(path, headers=_headers(auth_tokens, "radiologist")).status_code in (404, 422)

    def test_no_authentication_header_is_401_everywhere(self, client):
        """Regression: eleven endpoints used require_token=False, so anonymous callers could list patients,
        upload, classify, read audit data and download clinical reports."""
        calls = [("get", "/api/v1/images", {}), ("get", "/api/v1/results", {}), ("get", "/api/v1/audit", {}),
                 ("get", "/api/v1/model/info", {}), ("post", "/api/v1/classify/1", {}),
                 ("get", "/api/v1/results/1", {}), ("get", "/api/v1/reports/any.pdf", {}),
                 ("post", "/api/v1/results/1/report", {"json": {"language": "en"}}),
                 ("post", "/api/v1/results/1/review", {"json": {"status": "confirmed", "notes": "x"}})]
        for method, path, kwargs in calls:
            assert getattr(client, method)(path, **kwargs).status_code == 401, path

    def test_concurrent_classification_is_consistent(self, client, auth_tokens, sample_xray_image):
        up = client.post("/api/v1/images/upload", files={"file": ("c.jpg", sample_xray_image, "image/jpeg")},
                         headers=_headers(auth_tokens))
        image_id = up.json()["image_id"]

        def call(_):
            r = client.post(f"/api/v1/classify/{image_id}", headers=_headers(auth_tokens))
            return r.status_code, r.json().get("classification")

        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(call, range(12)))
        assert all(code == 200 for code, _ in results)
        assert len({label for _, label in results}) == 1   # same image -> same diagnosis, always


class TestTransportHardening:
    def test_security_headers_and_no_store_on_api(self, client, auth_tokens):
        res = client.get("/api/v1/images", headers=_headers(auth_tokens))
        assert res.headers["X-Content-Type-Options"] == "nosniff"
        assert res.headers["X-Frame-Options"] == "DENY"
        assert res.headers["Cache-Control"] == "no-store"


class TestCentralErrorMapping:
    @pytest.mark.parametrize("exc_name,status", [
        ("AuthenticationError", 401), ("OneTimeCodeError", 401), ("AuthorizationError", 403),
        ("FileSizeExceededError", 413), ("UnsupportedFormatError", 422), ("CorruptedImageError", 422),
        ("ImageQualityError", 422), ("InvalidModelInputError", 422), ("ModelNotLoadedError", 503),
        ("ModelNotFoundError", 404), ("InferenceError", 500), ("RepositoryError", 500),
    ])
    def test_status_for_every_domain_exception(self, exc_name, status):
        import inspect
        from api.error_handlers import status_for
        from utils import exceptions
        cls = getattr(exceptions, exc_name)
        params = [p for p in inspect.signature(cls.__init__).parameters.values()
                  if p.name != "self" and p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_OR_KEYWORD,)]
        exc = cls(16.2, 15) if exc_name == "FileSizeExceededError" else cls(*["x"] * len(params))
        assert status_for(exc) == status

    def test_unexpected_error_hides_internals(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.error_handlers import register_exception_handlers

        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/boom")
        def boom():
            raise RuntimeError("secret internal detail: /etc/passwd")

        res = TestClient(app, raise_server_exceptions=False).get("/boom")
        assert res.status_code == 500
        assert "secret" not in res.text and "passwd" not in res.text


def test_report_response_does_not_disclose_server_paths(client, auth_tokens, sample_xray_image):
    up = client.post("/api/v1/images/upload", files={"file": ("r.jpg", sample_xray_image, "image/jpeg")}, headers=_headers(auth_tokens))
    cl = client.post(f"/api/v1/classify/{up.json()['image_id']}", headers=_headers(auth_tokens))
    res = client.post(f"/api/v1/results/{cl.json()['prediction_id']}/report", json={"language": "en"},
                      headers=_headers(auth_tokens, "radiologist"))
    assert res.status_code == 200
    assert "path" not in res.json() and "/" not in res.json()["filename"]
