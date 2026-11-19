"""
Security regression tests (Milestone 3 feedback Q5 + attack scenarios)
======================================================================
Each test reproduces an attack that previously worked or that an examiner is likely to try.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.routes import _2fa_codes


def _start_login(client: TestClient, username: str):
    res = client.post("/api/v1/auth/login", json={"username": username, "password": "password123"})
    assert res.status_code == 200
    return res


class TestTwoFactorCannotBeBypassed:
    def test_login_never_returns_a_token(self, client: TestClient):
        body = _start_login(client, "technician").json()
        assert body["status"] == "2fa_required"
        assert "access_token" not in body
        client.post("/api/v1/auth/verify-2fa", json={"username": "technician", "code": _2fa_codes["technician"]["code"]})

    def test_exhausting_code_attempts_does_not_unlock_anything(self, client: TestClient):
        """Regression: after 5 wrong codes the pending-2FA row was deleted and the pre-issued token started working."""
        _start_login(client, "technician")
        for _ in range(5):
            res = client.post("/api/v1/auth/verify-2fa", json={"username": "technician", "code": "000000"})
            assert res.status_code == 401
        # the code is invalidated: even the right one is refused now
        res = client.post("/api/v1/auth/verify-2fa", json={"username": "technician", "code": _2fa_codes["technician"]["code"]})
        assert res.status_code in (400, 401)
        assert "access_token" not in res.json()

    def test_forged_and_malformed_tokens_are_rejected(self, client: TestClient):
        for bad in ("Bearer ", "Bearer null", "Bearer abc.def.ghi", "Basic abc", "Bearer " + "A" * 5000):
            res = client.get("/api/v1/auth/me", headers={"Authorization": bad})
            assert res.status_code == 401

    def test_token_with_none_algorithm_is_rejected(self, client: TestClient):
        import base64
        import json

        def b64(d):
            return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

        forged = f"{b64({'alg': 'none', 'typ': 'JWT'})}.{b64({'sub': 'admin', 'role': 'admin', 'user_id': 1})}."
        res = client.get("/api/v1/audit", headers={"Authorization": f"Bearer {forged}"})
        assert res.status_code == 401


class TestRoleBasedAccessIsEnforcedOnTheServer:
    def _upload_and_classify(self, client, token, image_bytes):
        up = client.post("/api/v1/images/upload", files={"file": ("xray.jpg", image_bytes, "image/jpeg")},
                         headers={"Authorization": f"Bearer {token}"})
        assert up.status_code == 200, up.text
        image_id = up.json()["image_id"]
        cl = client.post(f"/api/v1/classify/{image_id}", headers={"Authorization": f"Bearer {token}"})
        assert cl.status_code == 200, cl.text
        return cl.json()["prediction_id"]

    def test_only_the_radiologist_can_confirm_a_diagnosis(self, client, auth_tokens, sample_xray_image):
        pid = self._upload_and_classify(client, auth_tokens["technician"], sample_xray_image)
        payload = {"status": "confirmed", "notes": "ok"}
        for role in ("technician", "admin", "ml_engineer"):
            res = client.post(f"/api/v1/results/{pid}/review", json=payload,
                              headers={"Authorization": f"Bearer {auth_tokens[role]}"})
            assert res.status_code == 403, f"{role} must not confirm diagnoses"
        ok = client.post(f"/api/v1/results/{pid}/review", json=payload,
                         headers={"Authorization": f"Bearer {auth_tokens['radiologist']}"})
        assert ok.status_code == 200

    def test_a_role_claim_edited_in_the_token_is_not_trusted(self, client, auth_tokens):
        """Changing the payload without the signing key must invalidate the token."""
        import base64
        import json
        head, payload, sig = auth_tokens["technician"].split(".")
        data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        data["role"] = "admin"
        tampered = base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()
        res = client.get("/api/v1/audit", headers={"Authorization": f"Bearer {head}.{tampered}.{sig}"})
        assert res.status_code == 401


class TestPasswordRecoveryDoesNotLeakAccounts:
    def test_same_answer_for_existing_and_unknown_users(self, client: TestClient):
        known = client.post("/api/v1/auth/forgot-password", json={"username": "technician"})
        unknown = client.post("/api/v1/auth/forgot-password", json={"username": "nobody_here"})
        assert known.status_code == unknown.status_code == 200
        assert known.json()["message"] == unknown.json()["message"]


class TestHostileUploads:
    @pytest.mark.parametrize("name", ["../../etc/passwd.jpg", "..\\..\\evil.png", "a" * 400 + ".jpg", "x.jpg\x00.exe"])
    def test_hostile_file_names_never_escape_storage(self, client, auth_tokens, sample_xray_image, name):
        res = client.post("/api/v1/images/upload", files={"file": (name, sample_xray_image, "image/jpeg")},
                          headers={"Authorization": f"Bearer {auth_tokens['technician']}"})
        assert res.status_code in (200, 400, 422)
        assert res.status_code != 500

    def test_decompression_bomb_is_rejected(self, client, auth_tokens):
        big = Image.new("L", (9000, 9000), 0)  # above MAX_IMAGE_DIMENSION
        buf = io.BytesIO()
        big.save(buf, format="PNG")
        res = client.post("/api/v1/images/upload", files={"file": ("big.png", buf.getvalue(), "image/png")},
                          headers={"Authorization": f"Bearer {auth_tokens['technician']}"})
        assert res.status_code in (413, 422)

    def test_text_file_renamed_to_image_is_rejected(self, client, auth_tokens):
        res = client.post("/api/v1/images/upload", files={"file": ("a.jpg", b"not an image at all", "image/jpeg")},
                          headers={"Authorization": f"Bearer {auth_tokens['technician']}"})
        assert res.status_code == 422


class TestSelfRegistrationCannotEscalateRole:
    """Regression: anyone could register as radiologist or ML engineer and then confirm diagnoses / read the audit trail."""

    @pytest.mark.parametrize("role", ["admin", "radiologist", "ml_engineer"])
    def test_anonymous_cannot_create_a_privileged_account(self, client, role):
        res = client.post("/api/v1/auth/register", json={"username": f"rogue_{role}", "password": "Password123!",
                                                          "full_name": "Rogue", "role": role})
        assert res.status_code == 403

    def test_a_technician_token_cannot_create_a_radiologist(self, client, auth_tokens):
        res = client.post("/api/v1/auth/register", headers={"Authorization": f"Bearer {auth_tokens['technician']}"},
                          json={"username": "rogue_rad2", "password": "Password123!", "full_name": "Rogue", "role": "radiologist"})
        assert res.status_code == 403

    def test_anonymous_can_register_a_technician_and_the_admin_can_create_a_radiologist(self, client, auth_tokens):
        ok = client.post("/api/v1/auth/register", json={"username": "new_tech", "password": "Password123!", "full_name": "New Tech"})
        assert ok.status_code == 200 and ok.json()["role"] == "technician"
        adm = client.post("/api/v1/auth/register", headers={"Authorization": f"Bearer {auth_tokens['admin']}"},
                          json={"username": "new_rad", "password": "Password123!", "full_name": "New Rad", "role": "radiologist"})
        assert adm.status_code == 200 and adm.json()["role"] == "radiologist"
