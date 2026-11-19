"""
Aggregate throughput with N independent API processes on one machine (no containers needed).

Each process is a real ``uvicorn main:app`` with its own database / storage directory and its own copy of
the model, driven through HTTP by scripts/load_test.py (POST /api/v1/classify). It shows what a
horizontal scale-out costs on shared hardware and why "N replicas = N x the throughput of one" is wrong.

    python scripts/process_scaling.py --processes 1 2 4 --requests 60 --clients 2 --output process_scaling.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def start(port: int, workdir: Path) -> subprocess.Popen:
    env = dict(os.environ, DATA_DIR=str(workdir / "data"), LOGS_DIR=str(workdir / "logs"), ENVIRONMENT="development",
               OTP_LOG_CODES="true", TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="1", PYTHONUNBUFFERED="1")
    log = open(workdir / "server.log", "w")
    return subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--port", str(port), "--log-level", "warning"],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)


def token_for(port: int, workdir: Path) -> str:
    base = f"http://127.0.0.1:{port}"
    with httpx.Client(base_url=base, timeout=60) as client:
        client.post("/api/v1/auth/login", json={"username": "technician", "password": "password123"}).raise_for_status()
        time.sleep(0.5)
        code = re.findall(r"CODE for 'technician': (\d{6})", (workdir / "server.log").read_text())[-1]
        res = client.post("/api/v1/auth/verify-2fa", json={"username": "technician", "code": code})
        res.raise_for_status()
        return res.json()["access_token"]


def run_level(n: int, base_port: int, requests: int, clients: int) -> dict:
    tmp = [Path(tempfile.mkdtemp(prefix=f"scal{n}-{i}-")) for i in range(n)]
    procs = [start(base_port + i, tmp[i]) for i in range(n)]
    try:
        for i in range(n):
            for _ in range(180):
                try:
                    if httpx.get(f"http://127.0.0.1:{base_port + i}/ready", timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError(f"process {i} did not become ready")
        tokens = [token_for(base_port + i, tmp[i]) for i in range(n)]

        def drive(i: int) -> dict:
            out = tmp[i] / "result.json"
            subprocess.run([sys.executable, str(ROOT / "scripts" / "load_test.py"), "--url", f"http://127.0.0.1:{base_port + i}",
                            "--token", tokens[i], "-n", str(requests), "-c", str(clients), "--warmup", "4", "--output", str(out)],
                           check=True, stdout=subprocess.DEVNULL)
            return json.loads(out.read_text())

        with ThreadPoolExecutor(max_workers=n) as pool:
            results = list(pool.map(drive, range(n)))
        return {"processes": n, "throughput_rps": round(sum(r["throughput_rps"] for r in results), 2),
                "max_error_rate": max(r["error_rate"] for r in results)}
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--base-port", type=int, default=8101)
    parser.add_argument("--output")
    args = parser.parse_args()
    rows = [run_level(n, args.base_port, args.requests, args.clients) for n in args.processes]
    base = rows[0]["throughput_rps"]
    for r in rows:
        r["speedup"] = round(r["throughput_rps"] / base, 2)
        r["efficiency_pct"] = round(100 * r["speedup"] / r["processes"])
        print(f"{r['processes']} process(es): {r['throughput_rps']} req/s  speed-up {r['speedup']}x  efficiency {r['efficiency_pct']}%  error rate {r['max_error_rate']}")
    if args.output:
        Path(args.output).write_text(json.dumps({"cpu_count": os.cpu_count(), "levels": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
