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

| Document | Description |
|----------|-------------|
| [docs/architecture.md](architecture.md) | System-level diagram — target and attacker relationship |
| [docs/target/overview.md](target/overview.md) | What the target is and the design decisions behind it |
| [docs/target/architecture.md](target/architecture.md) | CifarCNN layer-by-layer breakdown and training details |
| [docs/target/api.md](target/api.md) | Full REST API reference |
| [docs/target/preprocessing.md](target/preprocessing.md) | Image preprocessing pipeline |

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
