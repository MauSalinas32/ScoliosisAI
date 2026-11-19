# Changelog

All notable changes to the Scoliosis AI Diagnostic System are documented in this file.

## [3.3.0] - Milestone 4: Verified corrections, hardening and evidence
Corrections found by auditing the 3.2.0 code against the Milestone 3 feedback (each one has a regression test or a measured run).
- **Security**
  - **Anonymous access closed:** eleven endpoints were declared with `require_token=False`, so a request with no `Authorization` header skipped authentication and RBAC entirely (list/upload/classify images, list results, read audit entries, generate and download clinical PDF reports). Every endpoint now requires a valid token (`tests/integration/test_robustness.py::test_no_authentication_header_is_401_everywhere`).
  - **Patient identifiers no longer reach the disk:** the stored DICOM used to be kept byte-for-byte while only an in-memory metadata dictionary was masked (yet `is_anonymized` was `true`). `anonymize_dicom_bytes` now erases the direct identifiers and private tags before the file is stored (fails closed if it cannot).
  - CORS is no longer `*` with credentials (production: same origin only, configurable with `CORS_ALLOWED_ORIGINS`); security headers and `Cache-Control: no-store` on `/api/*`; ingress rate limit for brute-force attempts.
  - **2FA bypass closed:** `/auth/login` used to return a JWT before the code was verified and blocked it only while the code row existed; after 5 wrong codes (or on expiry) the row was deleted and the token worked without 2FA. Login now returns no token; the JWT is issued only by `/auth/verify-2fa` (`tests/integration/test_security_hardening.py`).
  - One-time codes are written to the console / kept in the in-memory test mirror only when `OTP_LOG_CODES=true` (default `false` when `ENVIRONMENT=production`); the database stores only HMAC digests.
  - `SECRET_KEY` has no public default: with `ENVIRONMENT=production` the service refuses to start with a missing, default or short (< 32 chars) key. The hard-coded key was removed from `docker-compose.yml` and `k8s/secret.yaml`.
  - Demo users (`password123`) are created only outside production; in production only `admin` is created from `INITIAL_ADMIN_PASSWORD`.
  - `/auth/forgot-password` answers identically for existing and unknown users (no account enumeration).
  - **Only the radiologist can confirm or correct a diagnosis** (server returns 403 to technician, administrator and ML engineer; the UI hides the buttons accordingly).
- **Model robustness**
  - `ScoliosisStrategy.validate_input` rejects tensors outside `[0, 1]`, with the wrong shape, or with NaN/Inf (`HTTP 422`) instead of silently repairing them: feeding `[0, 255]` values used to return "S-curve, 100 %".
  - `ai/model_probe.py` (input-sensitivity probe) runs when a model is loaded: a model whose output does not react to its input is never served, even when its metadata says `approved: true`.
  - Training fails when the dataset is missing unless `ALLOW_SYNTHETIC_DATA=true`; a model saved without evaluation metrics is never approved.
- **Reproducibility / deployment**
  - `bcrypt` pinned `< 5`: with the current bcrypt 5.x, passlib 1.7 raises `ValueError` while hashing, so on a clean install (e.g. the Docker image) no user could be created and nobody could log in. Found by the 2-replica test, invisible on a machine with an older bcrypt.
  - Model bootstrap on a fresh database registered, approved and activated v1.1.0 but then judged the stale in-memory object "unapproved", so the first start stayed `not_ready`; the version is now re-read.
  - `MODEL_DIR` now defaults to `<repo>/models`. It pointed to `data/models`, which is empty in a clean clone and is hidden by the volume mounted on `/app/data` in Docker and Kubernetes, so the service stayed `not_ready` there.
  - PostgreSQL schema creation is serialised with an advisory lock (two replicas starting together on an empty database no longer race); `psycopg2-binary` added.
  - `kustomization.yaml` no longer applies `secret.example.yaml` (it would overwrite the real secret with placeholders).
  - Dockerfile installs the wheelhouse through a bind mount (no 2 GB layer); `medmnist` (unused, pulled PyTorch + CUDA) removed; test tools moved to `requirements-dev.txt`.
  - `.gitignore` / `.dockerignore` added; `medical_system.db`, `reports/*.pdf` and `k8s/secret.yaml` removed from version control.
  - Central exception-to-HTTP mapping (`api/error_handlers.py`); unexpected errors and report generation no longer return internal text or server paths to the client.
  - Docker image: numeric UID/GID 10001 (matches the Kubernetes `securityContext`), one worker per container (TensorFlow memory against the 2 GiB pod limit); namespace labelled for Pod Security Admission (warn/audit `restricted`); ingress rate limit; `matplotlib` added to `requirements-dev.txt` (needed by `plot_confusion_matrix.py`).
  - Benchmark: the bottleneck is derived from the measurement (the old file always named inference although preprocessing was larger); `--iterations` / `--output` options; new `forward_pass_comparison.py` and `scripts/process_scaling.py`.
  - `.github/workflows/ci-cd.yml` added (lint, bandit, tests with coverage gate on SQLite and PostgreSQL, Docker smoke test, 2-replica test, Trivy, kubeconform, kind staging); `scripts/replica_test.sh` and `scripts/load_test.py` provide the replica / load evidence.
- **Tests**: 189 tests (was 102), 79 % coverage, all passing from a clean clone; the suite now runs in an isolated temporary directory with the standard FastAPI `TestClient` (the home-made ASGI driver failed with other httpx versions and the tests wrote into the project's database).

## [3.2.0] - Milestone 3 Consolidated: Model v1.1.0 Validation, Memory Optimization & Hardened RBAC
- **AI Model Iteration `v1.1.0` & Quality Gate (Criterion 3 — Questions 1, 2, 9 & 10):**
  - Fixed the input scaling mismatch in `MobileNetV2` by replacing raw `preprocess_input` with an in-model `Rescaling(2.0, offset=-1.0)` layer that maps `[0.0, 1.0] -> [-1.0, 1.0]`.
  - Added a spatial $7\times 7$ spinal curvature bottleneck alongside `GlobalAveragePooling2D` and inverse-frequency class weights ($w_k = \frac{N}{K \cdot N_k}$).
  - Implemented `ai/metrics.py` with `compute_majority_baseline`, `compute_calibration_summary`, and `passes_quality_gate` ($\text{Macro-F1} \ge 0.70$, per-class sensitivity $\ge 0.60$, strictly outperforming the $44.55\%$ majority-class baseline).
  - Trained, evaluated, and promoted `scoliosis_classifier v1.1.0` on the 101 held-out test radiographs (`88.12%` accuracy, `87.20%` balanced accuracy, `88.07%` macro-F1, `approved: true`), while preserving `v1.0.0` (`44.55%` accuracy, `33.33%` balanced accuracy, `approved: false`) as historical baseline evidence.
- **Expanded Automated Test Suite (Criterion 2 — Question 4):**
  - Expanded the test suite to **102 automated tests** (unit, integration, memory profiling, system E2E); coverage 78 % (figures corrected in 3.3.0 from the real run).
  - Added `test_critical_medical_non_collapse_and_sensitivity_probe_question_4` to detect mode collapse and verify multi-class responsiveness.
- **Server-Side Security, 2FA & Strict RBAC (Question 5):**
  - Persisted 6-digit one-time codes (`2FA` and password recovery) as `HMAC-SHA256` digests in the `one_time_codes` table with 5-minute expiration, 5-attempt lockout, and 3-resend limit.
  - Enforced server-side RBAC on every REST endpoint (`HTTP 403 Forbidden` when a `technician` attempts to confirm a diagnosis via `POST /api/v1/results/{id}/review`, view clinical history, or access audit logs).
- **Memory Profiling & Data-Type Optimization (Question 6):**
  - Replaced `int64`/`float64` temporary arrays in `ValidationStage` and `QualityCheckStage` with signed `int16` and `float32`, reducing peak memory on $1024\times 1024$ radiographs by ~70% (`67 MB -> ~19.5 MB`) while preventing `uint8` wrap-around and preserving numerical precision ($< 10^{-4}$ vs `float64`).
  - Replaced unbounded latency list in `MetricsInferencerDecorator` with a bounded ring buffer `deque(maxlen=1000)`.
- **CI/CD, Containers & Kubernetes Orchestration (Criteria 4, 5 & 6 — Questions 7 & 8):**
  - Configured blocking Trivy `CRITICAL` CVE gate (`exit-code: "1"`) and `--cov-fail-under=75` in `.github/workflows/ci-cd.yml` (the workflow file itself was added in 3.3.0).
  - Declarative Kubernetes manifests in `k8s/` including `StatefulSet` PostgreSQL 16, `ReadWriteMany` PVC, `/ready` readiness gating (`HTTP 503` without an approved model), `HPA` (2–6 replicas), and `PodDisruptionBudget`.

## [3.0.0 / 3.1.0] - Milestone 3: Testing, Validation, and Deployment
- Initial 5-stage GitHub Actions CI/CD workflow, multi-stage non-root `Dockerfile`, performance benchmarking (`@tf.function` speedup $89.4\text{ ms} \rightarrow 21.8\text{ ms}$), and baseline `v1.0.0` evaluation.

## [2.0.0] - Milestone 2: Scoliosis & Vertebral Analysis
- Fine-tuned MobileNetV2 for 3-class spinal morphology classification, automated vertebral landmark tracking ($T1\text{–}L5$), localized bilingual PDF reports (`America/Mexico_City`), 2FA, and RBAC.

## [1.0.0] - Milestone 1: Core Modular Architecture
- Modular monolith architecture with domain contracts (`ABC`), Dependency Inversion (`DIP`), Fail-Fast preprocessing pipeline, and GoF Strategy/Factory/Decorator patterns.

## [3.2.1] - Milestone 4: Ethical, Legal & Social Hardening
- Fail-safe hardening: PDF Unicode/long-token sanitizer, compensating disk cleanup on DB failure, query-filter and pagination validation, review-status validation, empty-file rejection, keyboard-mash protection (auto-repeat shield, in-flight lock, maxlength), thread-safe inference, filename truncation.
- Added Correct action for radiologists, test-artifact cleanup, Docker/GitHub configuration fixes (102 tests passing).
