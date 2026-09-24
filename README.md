# Task-01 — AI Security Red Team Assessment

A red team simulation against a production-style AI system, issued by Ethiopian Artificial
Intelligence. The assessment covers adversarial ML attacks, pipeline vulnerability analysis,
metadata leakage, and mitigations aligned with MITRE ATLAS and NIST AI RMF.

---

## Project layout

```
task-01/
├── target/          # Trained CNN + FastAPI inference server (the system under attack)
│   ├── app/         # Server source (model, inference, preprocessing, routes)
│   ├── training/    # Training script and dataset loader
│   ├── weights/     # Saved model weights
│   ├── data/        # CIFAR-10 raw data (auto-downloaded)
│   └── Dockerfile   # Container image for the inference server
├── attacks/         # Adversarial attack scripts
├── evidence/        # Screenshots, logs, adversarial samples
├── reports/         # Attack report and pipeline review
├── docs/            # Full documentation
├── video/           # Video walkthrough
└── ai-usage/        # AI usage log
```

---

## Scope

| Area | Description |
|------|-------------|
| Adversarial inputs | Crafted images that fool the classifier |
| Pipeline vulnerabilities | Weaknesses in the Docker-based serving system |
| Metadata leakage | Information exposed by the API that aids an attacker |
| Mitigations | Remediations mapped to MITRE ATLAS and NIST AI RMF |

---

## Quick-start

### 1. Prerequisites

- Docker (for the target inference server)
- Python 3.12 + `venv` (for the attack scripts only)

### 2. Build and run the target

The Docker commands are the same on all platforms. Run from the task-01 root:

```bash
docker build -t cifar10-target ./target
```

**Hardened run (integrity checks enabled)**

The inference server verifies the SHA-256 hash of the weights and metadata before
loading the model. Pass the expected hashes as environment variables:

```bash
docker run -p 127.0.0.1:8000:8000 \
  -e MODEL_SHA256=c83f56d8354266c487c0a537d4c44e56149f27fcf8950f469b279ecf340d5929 \
  -e MODEL_INFO_SHA256=b8435f91a1e1f5a9e96ea0c12c2f1fa29c8857221038e0eddb6759a701cbe6c8 \
  cifar10-target:latest
```

If either file has been tampered with, the container will exit immediately with
`File integrity check failed.` and the model will not load.

> **Note:** if you retrain the model, run `python -m target.training.train` and copy
> the new `-e` values it prints at the end of training.

The API is then available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

### 3. Set up the attack environment

The virtual environment is only needed for running the attack scripts — the target itself runs entirely in Docker.

**Linux / macOS**
```bash
# From the task-01 root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
# From the task-01 root
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4. Test a prediction

**Linux / macOS**
```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@/path/to/image.png"
```

**Windows (PowerShell)**
```powershell
curl.exe -X POST http://localhost:8000/predict `
         -F "file=@C:\path\to\image.png"
```

Example response:

```json
{
  "predicted_class": 3,
  "class_name": "cat",
  "confidence": 0.923100,
  "probabilities": {"0": 0.002100, "1": 0.001200, "2": 0.003400, "3": 0.923100,
                    "4": 0.001000, "5": 0.062000, "6": 0.002100, "7": 0.004000,
                    "8": 0.001000, "9": 0.001000}
}
```

---

## Deliverables

| # | Deliverable | Location |
|---|-------------|----------|
| 1 | Attack Report | `reports/attack-report.md` |
| 2 | Adversarial Examples | `attacks/`, `evidence/adversarial/` |
| 3 | Pipeline Review | `reports/attack-report.md` |
| 4 | Video Walkthrough | `video/` |
| 5 | AI Usage Log | `ai-usage/ai-usage-log.md` |

---

## Documentation

Full documentation lives in [`docs/`](docs/index.md).

| Document | Contents |
|----------|----------|
| [docs/index.md](docs/index.md) | Project overview and goals |
| [docs/architecture.md](docs/architecture.md) | System-level diagram |
| [docs/target/overview.md](docs/target/overview.md) | Target purpose and design decisions |
| [docs/target/architecture.md](docs/target/architecture.md) | CifarCNN layer-by-layer breakdown |
| [docs/target/api.md](docs/target/api.md) | REST API reference |
| [docs/target/setup.md](docs/target/setup.md) | Setup and deployment guide |
| [docs/target/preprocessing.md](docs/target/preprocessing.md) | Image preprocessing pipeline |

---

