"""
Real HTTP demonstration of server-side authorisation (Milestone 3 feedback Q5).

    # terminal 1:  uvicorn main:app --port 8000 > server.log 2>&1
    # terminal 2:  python scripts/rbac_demo.py --url http://localhost:8000 --log server.log | tee evidence/rbac_api_demo.txt
"""

import argparse
import io
import re
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image


def radiograph() -> bytes:
    x = np.arange(400)
    row = np.clip(np.maximum(0, 180 - np.abs(x - 200)) + 40, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(np.broadcast_to(row, (600, 400)).copy(), mode="L").convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--log", required=True, help="server log (development mode prints the simulated SMS codes)")
    args = parser.parse_args()
    c = httpx.Client(base_url=args.url + "/api/v1", timeout=60)

    def login(user: str) -> str:
        r = c.post("/auth/login", json={"username": user, "password": "password123"})
        assert "access_token" not in r.json(), "login must not return a token"
        time.sleep(0.5)
        code = re.findall(rf"CODE for '{user}': (\d{{6}})", Path(args.log).read_text())[-1]
        return c.post("/auth/verify-2fa", json={"username": user, "code": code}).json()["access_token"]

    def show(label: str, response: httpx.Response) -> None:
        print(f"{label:<62} -> HTTP {response.status_code}")

    print("== Step 1 of the login returns no token")
    r = c.post("/auth/login", json={"username": "technician", "password": "password123"})
    print(f"POST /auth/login -> HTTP {r.status_code}, body keys: {sorted(r.json())}")
    tokens = {u: login(u) for u in ("technician", "radiologist", "admin")}
    auth = {u: {"Authorization": f"Bearer {t}"} for u, t in tokens.items()}

    print("\n== A technician uploads and classifies; the result is pending_review")
    up = c.post("/images/upload", files={"file": ("xray.jpg", radiograph(), "image/jpeg")}, headers=auth["technician"])
    image_id = up.json()["image_id"]
    cl = c.post(f"/classify/{image_id}", headers=auth["technician"])
    pid = cl.json()["prediction_id"]
    stored = c.get(f"/results/{pid}", headers=auth["radiologist"]).json()
    print(f"prediction {pid}: {cl.json()['classification']} (stored status: {stored['status']})")

    print("\n== Confirming a diagnosis directly through the API (the UI is not involved)")
    body = {"status": "confirmed", "notes": "reviewed"}
    show("technician  POST /results/{id}/review", c.post(f"/results/{pid}/review", json=body, headers=auth["technician"]))
    show("admin       POST /results/{id}/review", c.post(f"/results/{pid}/review", json=body, headers=auth["admin"]))
    show("no token    POST /results/{id}/review", c.post(f"/results/{pid}/review", json=body))
    state = c.get(f"/results/{pid}", headers=auth["radiologist"]).json()
    print(f"record after the refused attempts: status = {state['status']}")
    show("radiologist POST /results/{id}/review", c.post(f"/results/{pid}/review", json=body, headers=auth["radiologist"]))
    state = c.get(f"/results/{pid}", headers=auth["radiologist"]).json()
    print(f"record after the radiologist's review: status = {state['status']}")

    print("\n== Other endpoints without credentials or with the wrong role")
    show("anonymous   GET  /results", c.get("/results"))
    show("anonymous   GET  /images", c.get("/images"))
    show("anonymous   GET  /audit", c.get("/audit"))
    show("technician  GET  /results", c.get("/results", headers=auth["technician"]))
    show("technician  GET  /audit", c.get("/audit", headers=auth["technician"]))
    show("radiologist GET  /audit", c.get("/audit", headers=auth["radiologist"]))
    show("admin       GET  /audit", c.get("/audit", headers=auth["admin"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
