# Task-01 — AI Security Red Team Assessment

## Goal

This project is a red team simulation against a production-style AI system, issued by
Ethiopian Artificial Intelligence. The work covers four areas:

- **Adversarial inputs** — crafted images designed to fool the CIFAR-10 classifier
- **Pipeline vulnerabilities** — weaknesses in the Docker-based serving system
- **Metadata leakage** — information exposed by the API that benefits an attacker
- **Mitigations** — remediations mapped to MITRE ATLAS and NIST AI RMF

The system under attack is built from scratch: a CIFAR-10 CNN wrapped in a FastAPI
inference server and deployed via Docker. This mirrors how red teams operate — recreating
the environment from intelligence and reconnaissance rather than privileged access.

---

## Components

| Component | Location |
|-----------|----------|
| Target model + API | `target/` |
| Adversarial attacks | `attacks/` |
| Pipeline & metadata review | `reports/attack-report.md` |
| Mitigations (MITRE ATLAS / NIST AI RMF) | `reports/attack-report.md` |
| Evidence | `evidence/` |
| Video walkthrough | `video/` |
| AI usage log | `ai-usage/ai-usage-log.md` |

---

## Documentation map

### Target System Documentation

| Document | Description |
|----------|-------------|
| [docs/target/overview.md](target/overview.md) | What the target is and the design decisions behind it |
| [docs/target/architecture.md](target/architecture.md) | CifarCNN layer-by-layer breakdown and training details |
| [docs/target/api.md](target/api.md) | Full REST API reference |
| [docs/target/preprocessing.md](target/preprocessing.md) | Image preprocessing pipeline |
| [docs/target/training.md](target/training.md) | How to train the model and generate weights |

### Attack Documentation

| Document | Description |
|----------|-------------|
| [docs/attacks/overview.md](attacks/overview.md) | Summary of all four attacks and when to use each |
| [docs/attacks/attack-01-fgsm.md](attacks/attack-01-fgsm.md) | FGSM (Fast Gradient Sign Method) — single-step white-box |
| [docs/attacks/attack-02-pgd.md](attacks/attack-02-pgd.md) | PGD (Projected Gradient Descent) — iterative white-box |
| [docs/attacks/attack-03-square.md](attacks/attack-03-square.md) | Square Attack — query-efficient black-box |
| [docs/attacks/attack-04-extraction.md](attacks/attack-04-extraction.md) | Model Extraction — black-box model stealing |

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

## Key design decisions

- The target is **fully containerised** so the attack environment stays separate.
  Attacks communicate with it over HTTP exactly as a real attacker would.
- The virtual environment (`.venv`) is used **only** for attack scripts, not the target.
- Attacks are scoped to the locally running model and local containers only. External
  systems are out of scope.
