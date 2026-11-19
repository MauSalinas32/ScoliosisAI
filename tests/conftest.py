"""
Shared Test Fixtures and Configurations
=======================================
  - Every run is isolated: data, uploads, reports, logs and the SQLite database live in a
    throw-away temporary directory (the project's own database/storage is never touched).
    Set TEST_DATABASE_URL to run the same suite against PostgreSQL.
  - Model artefacts are read from <repo>/models (MODEL_DIR default).
  - FastAPI TestClient, synthetic medical images, and JWTs for every role obtained through
    the real two-step 2FA login.
"""

import io
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from PIL import Image
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Isolate state BEFORE any application module is imported (settings read the environment at import time)
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="scoliosis-tests-"))
os.environ["DATA_DIR"] = str(_TEST_ROOT / "data")
os.environ["LOGS_DIR"] = str(_TEST_ROOT / "logs")
os.environ["ENVIRONMENT"] = "development"
os.environ["OTP_LOG_CODES"] = "true"        # tests read the simulated SMS codes
os.environ.pop("ALLOW_UNAPPROVED_MODEL", None)

from infrastructure.database import init_database  # noqa: E402
from api.routes import _2fa_codes  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create the schema in the isolated database and remove the temporary directory afterwards."""
    init_database()
    yield
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)


def login_with_2fa(client, username: str, password: str = "password123") -> str:
    """Full two-step login; returns the JWT issued after the 2FA code is verified."""
    res = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert res.status_code == 200, res.text
    assert "access_token" not in res.json(), "no token may be issued before 2FA"
    code = _2fa_codes[username]["code"]
    res = client.post("/api/v1/auth/verify-2fa", json={"username": username, "code": code})
    assert res.status_code == 200, res.text
    return res.json()["access_token"]


@pytest.fixture(scope="session")
def app():
    """Application fixture reusing the single initialized FastAPI instance."""
    import main as main_module
    return main_module.app


@pytest.fixture(scope="session")
def client(app) -> Generator:
    """FastAPI TestClient fixture."""
    from fastapi.testclient import TestClient
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def auth_tokens(client) -> dict:
    """
    Log in default users and obtain their active JWT tokens.
    Completes 2FA verification code generation and validation.
    """
    return {u: login_with_2fa(client, u) for u in ("admin", "radiologist", "technician", "ml_engineer")}


@pytest.fixture
def sample_xray_image() -> bytes:
    """Generate a valid synthetic spinal X-ray image in JPEG format (400x600 px)."""
    x_coords = np.arange(400, dtype=np.int16)
    dist = np.abs(x_coords - 200)
    row = np.clip(np.maximum(0, 180 - dist) + 40, 0, 255).astype(np.uint8)
    arr = np.broadcast_to(row, (600, 400)).copy()

    img = Image.fromarray(arr, mode="L").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
def sample_invalid_small_image() -> bytes:
    """Generate an image below minimum size threshold (16x16 < 32x32) to test Fail-Fast."""
    img = Image.new("L", (16, 16), color=128).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_corrupt_file() -> bytes:
    """Generate random non-image bytes to test corruption handling."""
    return b"%CORRUPTED_IMAGE_HEADER_FAKE_DATA_1234567890%"
