#!/usr/bin/env bash
# =============================================================================
# Replica test: PostgreSQL + 2 API replicas + nginx round-robin load balancer
# =============================================================================
# Demonstrates with real containers what Kubernetes relies on when it scales the Deployment:
#   A. both replicas start together against an empty database (schema creation race)
#   B. a 2FA code issued by replica 1 is accepted by replica 2 (codes live in the database)
#   C. an image uploaded through replica 1 can be classified through replica 2 (shared state)
#   D. measured throughput: 1 replica vs. 2 replicas behind the balancer
#   E. availability: stopping one replica does not break requests through the balancer
#
#   IMAGE=scoliosis-api:final PY=python bash scripts/replica_test.sh | tee replica_test.txt
#   DIAGNOSTICS=1 prints container status, OOM kills and log tails before cleanup (used by CI)
# Requirements: docker, python with httpx/numpy/pillow (pip install -r requirements-dev.txt)
set -uo pipefail
IMAGE=${IMAGE:-scoliosis-api:latest}
PY=${PY:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
NET=scol-replica-net
CPUS=${CPUS:-2}

diagnostics() {
  # Printed before the containers are removed, so a failure on a CI runner can be explained from its log
  echo; echo "== DIAGNOSTICS (containers, memory kills, last log lines)"
  docker ps -a --format 'table {{.Names}}\t{{.Status}}' 2>&1 | grep scol || true
  for c in scol-api1 scol-api2; do
    echo "-- $c oom_killed=$(docker inspect --format '{{.State.OOMKilled}} exit={{.State.ExitCode}}' $c 2>&1)"
    docker logs --tail 25 $c 2>&1 | cut -c1-200
  done
  echo "-- balancer"; docker logs --tail 10 scol-lb 2>&1 | cut -c1-200
}

cleanup() {
  [ "${DIAGNOSTICS:-0}" = "1" ] && diagnostics
  docker rm -f scol-pg scol-api1 scol-api2 scol-lb >/dev/null 2>&1
  docker network rm $NET >/dev/null 2>&1
  docker volume rm scol-data >/dev/null 2>&1
  rm -f /tmp/scol-nginx.conf
}
DIAGNOSTICS=0 cleanup
trap cleanup EXIT

say() { printf '\n== %s\n' "$*"; }
json() { "$PY" -c "import sys,json; d=json.load(sys.stdin); print($1)"; }

docker network create $NET >/dev/null
docker run -d --name scol-pg --network $NET -e POSTGRES_DB=medical_ai -e POSTGRES_USER=medical_ai \
  -e POSTGRES_PASSWORD=replica-test postgres:16.4-alpine >/dev/null
until docker exec scol-pg pg_isready -U medical_ai -d medical_ai >/dev/null 2>&1; do sleep 1; done

say "A. Start 2 replicas at the same time on an empty database"
start_replica() {
  docker run -d --name scol-api$1 --network $NET --cpus "$CPUS" --memory 2g --read-only --tmpfs /tmp \
    -v scol-data:/app/data -p "1800$1:8000" \
    -e SECRET_KEY=replica-test-secret-0123456789abcdef \
    -e DATABASE_URL=postgresql+psycopg2://medical_ai:replica-test@scol-pg:5432/medical_ai \
    -e DEFAULT_USERS_PASSWORD=password123 -e OTP_LOG_CODES=true "$IMAGE" >/dev/null
}
start_replica 1 & start_replica 2 & wait
for n in 1 2; do
  for _ in $(seq 1 90); do curl -fs "localhost:1800$n/ready" >/dev/null && break; sleep 2; done
  echo "replica $n: /health=$(curl -s -o /dev/null -w '%{http_code}' localhost:1800$n/health) /ready=$(curl -s -o /dev/null -w '%{http_code}' localhost:1800$n/ready)"
done
echo "tracebacks at start-up: replica1=$(docker logs scol-api1 2>&1 | grep -c Traceback) replica2=$(docker logs scol-api2 2>&1 | grep -c Traceback)"

cat > /tmp/scol-nginx.conf <<'NGINX'
upstream api { server scol-api1:8000 max_fails=1 fail_timeout=2s; server scol-api2:8000 max_fails=1 fail_timeout=2s; }
server { listen 80; client_max_body_size 16m;
  location / { proxy_pass http://api; proxy_read_timeout 120s; proxy_next_upstream error timeout http_502 http_503; } }
NGINX
docker run -d --name scol-lb --network $NET -p 18080:80 -v /tmp/scol-nginx.conf:/etc/nginx/conf.d/default.conf:ro nginx:1.27-alpine >/dev/null
sleep 2

say "B. 2FA code issued by replica 1, verified on replica 2"
curl -s -X POST localhost:18001/api/v1/auth/login -H 'Content-Type: application/json' \
  -d '{"username":"technician","password":"password123"}' | json "d['status']"
sleep 1
CODE=$(docker logs scol-api1 2>&1 | grep -o "for 'technician': [0-9]\{6\}" | tail -1 | grep -o '[0-9]\{6\}')
RES=$(curl -s -X POST localhost:18002/api/v1/auth/verify-2fa -H 'Content-Type: application/json' \
  -d "{\"username\":\"technician\",\"code\":\"$CODE\"}")
echo "verify on replica 2 -> status=$(echo "$RES" | json "d.get('status')")"
TOKEN=$(echo "$RES" | json "d['access_token']")

say "C. Upload through replica 1, classify through replica 2"
"$PY" - "$TOKEN" <<'PYEOF'
import sys, httpx
sys.path.insert(0, ".")
from scripts.load_test import synthetic_radiograph
h = {"Authorization": f"Bearer {sys.argv[1]}"}
up = httpx.post("http://localhost:18001/api/v1/images/upload", headers=h, files={"file": ("rep.jpg", synthetic_radiograph(), "image/jpeg")}, timeout=60)
cl = httpx.post(f"http://localhost:18002/api/v1/classify/{up.json()['image_id']}", headers=h, timeout=60)
print(f"upload@replica1={up.status_code}  classify@replica2={cl.status_code}  classification={cl.json().get('classification')}")
PYEOF

say "D. Throughput (POST /classify, 1024x1024 image, same client)"
cd "$HERE/.." || exit 1
echo "--- 1 replica (direct), 4 concurrent clients"
"$PY" scripts/load_test.py --url http://localhost:18001 --token "$TOKEN" -n 120 -c 4 --warmup 6 --output /tmp/scol_lt_1.json | "$PY" -c "import sys,json; d=json.load(sys.stdin); print(d['throughput_rps'],'req/s', d['latency_ms'], 'errors', d['error_rate'])"
echo "--- 2 replicas behind nginx, 8 concurrent clients (4 per replica)"
"$PY" scripts/load_test.py --url http://localhost:18080 --token "$TOKEN" -n 240 -c 8 --warmup 10 --output /tmp/scol_lt_2.json | "$PY" -c "import sys,json; d=json.load(sys.stdin); print(d['throughput_rps'],'req/s', d['latency_ms'], 'errors', d['error_rate'])"
"$PY" - <<'PYEOF'
import json
a, b = (json.load(open(f"/tmp/scol_lt_{i}.json")) for i in (1, 2))
sp = b["throughput_rps"] / a["throughput_rps"]
print(f"speed-up with 2 replicas = {sp:.2f}x (ideal 2.00x) -> scaling efficiency {sp/2:.0%}")
PYEOF

say "E. Availability: stop replica 2, keep sending requests through the balancer"
docker stop scol-api2 >/dev/null
OK=0; for _ in $(seq 1 20); do [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:18080/health)" = "200" ] && OK=$((OK+1)); done
echo "20 requests via balancer with one replica down: $OK x 200"
