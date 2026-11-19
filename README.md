# Scoliosis AI Diagnostic System

[![CI/CD Pipeline](https://github.com/MauSalinas32/ScoliosisAI/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/MauSalinas32/ScoliosisAI/actions)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-189%20passed-brightgreen.svg)]()
[![Test Coverage](https://img.shields.io/badge/coverage-79%25-brightgreen.svg)]()
[![Model Accuracy](https://img.shields.io/badge/accuracy-88.12%25-success.svg)]()
[![Version](https://img.shields.io/badge/version-3.3.0-informational.svg)]()

Intelligent clinical decision support system (CDSS) for scoliosis detection, spinal curvature classification, and affected vertebral level localization from coronal spinal radiographs (X-rays). Developed for the **Software Engineering II** course (Team 3).

> **Clinical safety notice:** this system is a *second-opinion decision-support tool*. Every prediction starts as `pending_review` and must be confirmed or corrected by a licensed radiologist. It does not replace professional medical judgment.

---

## Key Features

- **Multi-Class Pathological Classification:**
  1. `NO SCOLIOSIS` (Normal): physiologically aligned spine.
  2. `SERIOUS SCOLIOSIS (C-CURVE)`: single structural curvature with identified apex.
  3. `SEVERE SCOLIOSIS (S-CURVE)`: double structural curvature with thoracic and lumbar inflection points.
- **Anatomical Vertebral Localization:** automated tracking of spinal landmarks (T1-T12, L1-L5), apex vertebra and affected vertebral range (e.g. `T7 - T11`).
- **Deep Transfer Learning:** **MobileNetV2** (ImageNet) fine-tuned for spinal radiography, with in-graph `Rescaling(2.0, offset=-1.0)`, a 7x7 spatial curvature head and inverse-frequency class weights.
- **Human-in-the-Loop Safety:** `requires_review` flag when confidence < 60% or when the CNN and the geometric estimator disagree; radiologists can **Confirm** or **Correct** every result.
- **Security & Traceability:**
  - `bcrypt` password hashing, JWT `HS256`, and two-factor authentication (6-digit codes stored as `HMAC-SHA256` digests with 5-minute expiry and 5-attempt lockout).
  - Server-side Role-Based Access Control (`technician`, `radiologist`, `ml_engineer`, `admin`) returning `HTTP 403` on unauthorized calls. **Only the radiologist can confirm or correct a diagnosis.**
  - The JWT is issued only after the 2FA code is verified (`/auth/login` never returns a token); the demo users and the console-printed codes exist only when `ENVIRONMENT != production`.
  - DICOM PHI anonymization and an immutable append-only audit log (`America/Mexico_City` timezone).
- **Fail-Fast Input Validation:** rejects non-radiographic images, color photos, empty/corrupt files, oversized uploads (`HTTP 413`) and unsupported formats (`HTTP 422`); supports JPG, PNG, TIFF and 16-bit DICOM.
- **Model Registry & Quality Gate:** only approved model versions (macro-F1 >= 0.70, per-class sensitivity >= 0.60) can serve traffic; otherwise `/ready` returns `HTTP 503`.
- **Input contract & sensitivity probe:** the service rejects tensors outside `[0, 1]` (`HTTP 422`) and, when loading a model, refuses to serve one whose output does not react to its input (`ai/model_probe.py`), whatever `metadata.json` says.
- **Bilingual (EN/ES)** web portal and PDF clinical reports, light/dark mode.
- **DevOps:** GitHub Actions pipeline (lint/SAST, tests with coverage gate on SQLite and PostgreSQL, Docker build + smoke test + 2-replica test + Trivy, Kubernetes manifest validation, kind staging deploy), non-root multi-stage `Dockerfile`, and Kubernetes manifests in `k8s/` (HPA 2-6 replicas, PDB, PostgreSQL StatefulSet, Ingress).

---

## Model Evaluation (`scoliosis_classifier v1.1.0`)

Evaluated on the independent 101-radiograph held-out test split (33 Normal, 45 C-curve, 23 S-curve):

| Metric | `v1.0.0` (rejected) | `v1.1.0` (approved, active) |
| :--- | :---: | :---: |
| **Overall accuracy** | 44.55% | **88.12%** (89/101) |
| **Balanced accuracy** | 33.33% | **87.20%** |
| **Macro F1-score** | 20.55% | **88.07%** |
| Sensitivity - Normal | 0.00% | 87.88% (29/33) |
| Sensitivity - C-curve | 100.00% | 91.11% (41/45) |
| Sensitivity - S-curve | 0.00% | 82.61% (19/23) |
| Quality gate | `approved: false` | `approved: true` |

Confusion matrix (`v1.1.0`): `[[29, 4, 0], [2, 41, 2], [0, 4, 19]]` - no Normal case was classified as S-curve and vice versa.

Artifacts: [`models/scoliosis_classifier_v1.1.0/confusion_matrix.png`](models/scoliosis_classifier_v1.1.0/confusion_matrix.png) and [`models/scoliosis_classifier_v1.1.0/evaluation_results.json`](models/scoliosis_classifier_v1.1.0/evaluation_results.json).

> Scope of these figures: they measure agreement with the reference labels of the 101-image test split (derived from the expert annotations of the public dataset). The split is small (about +/-4 points of uncertainty) and the model's confidence is high even on its errors (ECE = 0.119), so the probability shown to the user is an indication, not a certainty; every result goes to radiologist review.

---

## Project Structure

```text
.
|-- .github/workflows/ci-cd.yml   # CI/CD pipeline
|-- ai/                           # Transfer learning, training, evaluation, quality-gate metrics
|-- api/                          # FastAPI routes (RBAC, 2FA, upload, classify, review, reports, audit)
|-- config/                       # Centralized settings
|-- models/                       # Versioned model artefacts (v1.0.0 rejected, v1.1.0 approved)
|-- scripts/                      # replica_test.sh, load_test.py, process_scaling.py, rbac_demo.py
|-- evidence/                     # outputs of the test, benchmark, replica and RBAC runs
|-- domain/                       # Entities and abstract contracts (DIP)
|-- frontend/                     # Clinical web portal (HTML/CSS/JS, EN/ES)
|-- infrastructure/               # Repositories, storage adapter, model registry
|-- k8s/                          # Kubernetes manifests (kustomize)
|-- pipeline/                     # Fail-fast preprocessing stages
|-- services/                     # Auth, image mgmt, inference (Strategy/Decorator), clinical results, audit
|-- tests/                        # 189 automated tests (unit, integration, security, profiling, system)
|-- utils/                        # Logging and exception hierarchy
|-- Dockerfile / docker-compose.yml
|-- requirements.txt / requirements-dev.txt
`-- main.py                       # Composition root / entrypoint
```

---

## Getting Started

```bash
git clone https://github.com/MauSalinas32/ScoliosisAI.git
cd ScoliosisAI
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows PowerShell  (Linux/macOS: source venv/bin/activate)
pip install -r requirements-dev.txt
uvicorn main:app --port 8000
```

- Web portal: `http://localhost:8000/`
- Swagger UI: `http://localhost:8000/docs`
- Liveness / readiness: `http://localhost:8000/health` and `http://localhost:8000/ready`

Default demo users (password `password123`): `radiologist`, `technician`, `admin`, `ml_engineer`. The 2FA code is printed in the server console after login.

### Tests, evaluation and benchmark

```bash
pytest -v --cov=services --cov=pipeline --cov=domain --cov=api --cov=infrastructure --cov=ai --cov-fail-under=75 tests/   # 189 tests
python ai/evaluate_model.py                       # recompute metrics on the test split
python ai/plot_confusion_matrix.py                # confusion matrix chart
python tests/profiling/benchmark_performance.py --iterations 20   # latency / throughput benchmark
python ai/model_probe.py --version 1.1.0          # input-sensitivity probe of a model artefact
IMAGE=<image> bash scripts/replica_test.sh        # PostgreSQL + 2 replicas + load balancer (needs Docker)
```

Mean end-to-end CPU latency is ~153 ms per radiograph (RNF-01 target: < 3 s).

---

## Containerization & Deployment

```bash
cp .env.example .env              # set SECRET_KEY and POSTGRES_PASSWORD
docker compose up --build -d      # API + PostgreSQL on http://localhost:8000
docker compose logs -f medical-ai-api   # 2FA codes appear here (development mode)
docker compose down
```

```bash
kubectl apply -f k8s/namespace.yaml
kubectl -n medical-system create secret generic medical-ai-secret \
  --from-literal=SECRET_KEY=... --from-literal=POSTGRES_USER=medical_ai --from-literal=POSTGRES_PASSWORD=... \
  --from-literal=POSTGRES_DB=medical_ai_prod --from-literal=INITIAL_ADMIN_PASSWORD=... \
  --from-literal=DATABASE_URL=postgresql+psycopg2://medical_ai:...@postgres-service:5432/medical_ai_prod
kubectl apply -k k8s/             # namespace: medical-system (real secrets are never committed; see k8s/secret.example.yaml)
kubectl get pods,hpa -n medical-system
```

---

## Authors & Academic Credits

- **Course:** Software Engineering II
- **Academic Term:** Fifth Semester, 2026B
- **Team 3:** Aguilar, Hernandez, Gutierrez, Salinas, Zepeda

