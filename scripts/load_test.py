"""
HTTP Load Test (real capacity measurement)
===========================================
Measures what a deployed instance actually sustains — a single container, a Docker Compose
stack or a Kubernetes Service — instead of extrapolating from a sequential benchmark.

It uploads one radiograph, then sends N ``POST /api/v1/classify/{id}`` requests with C
concurrent clients and reports throughput, latency percentiles, the error rate and the
status-code distribution.

Authentication uses a JWT obtained once through the normal two-step login. The 2FA code is
asked interactively (it is printed in the server log), or pass an existing token:

    python scripts/load_test.py --url http://localhost:8000 --token "$TOKEN" -n 200 -c 8
    python scripts/load_test.py --url http://localhost:8000 --user technician --password ... -n 200 -c 8

Compare the same test against 1 replica and against N replicas behind the load balancer: the
ratio is the real scaling factor (usually below N: shared database, shared storage, CPU and
memory-bandwidth contention, load-balancer overhead, uneven distribution).
"""

import argparse
import getpass
import io
import json
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
from PIL import Image


def synthetic_radiograph(size: int = 1024) -> bytes:
    x = np.arange(size)
    column = np.clip(180 - np.abs(x - size // 2) / (size / 400), 0, None) + 40
    arr = np.tile(column, (size, 1)).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def obtain_token(client: httpx.Client, args) -> str:
    if args.token:
        return args.token
    password = args.password or getpass.getpass("Password: ")
    r = client.post("/api/v1/auth/login", json={"username": args.user, "password": password})
    r.raise_for_status()
    code = args.code or input("2FA code (see the server log): ").strip()
    r = client.post("/api/v1/auth/verify-2fa", json={"username": args.user, "code": code})
    r.raise_for_status()
    return r.json()["access_token"]


def percentile(values, q):
    return float(np.percentile(values, q)) if values else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--token")
    parser.add_argument("--user", default="technician")
    parser.add_argument("--password")
    parser.add_argument("--code")
    parser.add_argument("-n", "--requests", type=int, default=100)
    parser.add_argument("-c", "--concurrency", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--output", help="write the result as JSON")
    args = parser.parse_args()

    with httpx.Client(base_url=args.url, timeout=120) as client:
        headers = {"Authorization": f"Bearer {obtain_token(client, args)}"}
        up = client.post("/api/v1/images/upload", headers=headers,
                         files={"file": ("loadtest.jpg", synthetic_radiograph(args.image_size), "image/jpeg")})
        up.raise_for_status()
        image_id = up.json()["image_id"]

        def one_call(_):
            t0 = time.perf_counter()
            try:
                r = client.post(f"/api/v1/classify/{image_id}", headers=headers)
                return r.status_code, (time.perf_counter() - t0) * 1000
            except httpx.HTTPError as e:
                return type(e).__name__, (time.perf_counter() - t0) * 1000

        for _ in range(args.warmup):
            one_call(None)

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(one_call, range(args.requests)))
        elapsed = time.perf_counter() - started

    statuses = Counter(str(code) for code, _ in results)
    ok_latencies = [ms for code, ms in results if code == 200]
    report = {
        "url": args.url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "duration_s": round(elapsed, 2),
        "throughput_rps": round(len(ok_latencies) / elapsed, 2),
        "error_rate": round(1 - len(ok_latencies) / len(results), 4),
        "status_codes": dict(statuses),
        "latency_ms": {
            "mean": round(statistics.fmean(ok_latencies), 1) if ok_latencies else None,
            "p50": round(percentile(ok_latencies, 50), 1),
            "p95": round(percentile(ok_latencies, 95), 1),
            "p99": round(percentile(ok_latencies, 99), 1),
            "max": round(max(ok_latencies), 1) if ok_latencies else None,
        },
    }
    print(json.dumps(report, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return 0 if report["error_rate"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
