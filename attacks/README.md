# Attacks Directory

This directory contains all adversarial attack implementations and runner scripts.

---

## Quick Reference

### Run All Attacks

```bash
python -m attacks.run_all_attacks
```

### Run Individual Attacks

```bash
python -m attacks.attack_01    # FGSM (Fast Gradient Sign Method)
python -m attacks.attack_02    # PGD (Projected Gradient Descent)
python -m attacks.attack_03    # Square Attack (Black-box)
python -m attacks.attack_04    # Model Extraction
```

### Run Specific Types

```bash
# White-box only (no API needed) - ~3 minutes
python -m attacks.run_all_attacks --whitebox

# Black-box only (API required) - ~10 minutes
python -m attacks.run_all_attacks --blackbox
```

---

## File Structure

```
attacks/
├── run_all_attacks.py     # Runner script for all attacks
├── attack_01.py           # FGSM implementation
├── attack_02.py           # PGD implementation
├── attack_03.py           # Square Attack implementation
├── attack_04.py           # Model Extraction orchestrator
├── utils.py               # Shared utilities (loaders, savers, etc.)
├── model_extraction/      # Model extraction submodule
│   ├── query.py           # API querying logic
│   ├── training.py        # Surrogate model training
│   ├── evaluation.py      # Agreement and transfer testing
│   └── surrogate_arch.py  # Surrogate CNN architecture
└── samples/               # Sample adversarial images
```

---

## Attack Summary

| Attack | Type | Time | API Required | Use Case |
|--------|------|------|--------------|----------|
| 01 - FGSM | White-box | ~45s | No | Fast baseline testing |
| 02 - PGD | White-box | ~2m | No | Robust evaluation |
| 03 - Square | Black-box | ~6m | Yes | Query-based attacks |
| 04 - Model Extraction | Black-box | ~4m | Yes | Model stealing |

---

## Prerequisites

1. **Virtual environment active:**
   ```bash
   source .venv/bin/activate
   ```

2. **Dependencies installed:**
   ```bash
   pip install -r requirements.txt
   ```

3. **For black-box attacks - API running:**
   ```bash
   docker run -p 127.0.0.1:8000:8000 \
     -e MODEL_SHA256=<hash> \
     -e MODEL_INFO_SHA256=<hash> \
     cifar10-target:latest
   ```

---

## Output Locations

All evidence is saved automatically:

- **Images:** `evidence/adversarial/{attack_name}/`
- **Logs:** `evidence/logs/{attack_name}/`

---

## Documentation

- **[Running All Attacks Guide](../docs/attacks/running-all-attacks.md)** — Detailed runner usage
- **[Attack Overview](../docs/attacks/overview.md)** — Comparison and when to use each
- **[Attack 01 - FGSM](../docs/attacks/attack-01-fgsm.md)** — Fast Gradient Sign Method
- **[Attack 02 - PGD](../docs/attacks/attack-02-pgd.md)** — Projected Gradient Descent
- **[Attack 03 - Square](../docs/attacks/attack-03-square.md)** — Black-box query-based
- **[Attack 04 - Extraction](../docs/attacks/attack-04-extraction.md)** — Model stealing

---

## Troubleshooting

**"ModuleNotFoundError"**  
→ Run from project root: `cd task-01 && python -m attacks.run_all_attacks`

**"Connection refused"**  
→ Start the Docker container (see Prerequisites above)

---

For more details, see the [full documentation](../docs/attacks/).
