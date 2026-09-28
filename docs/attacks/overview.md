# Attack Documentation Overview

This directory documents the four adversarial attacks implemented against the CIFAR-10 classifier. Each attack demonstrates different techniques for compromising the model's predictions, from white-box gradient-based methods to black-box query-based approaches.

---

## Attack Summary

| Attack | Type | Complexity | Primary Use Case |
|--------|------|------------|------------------|
| [Attack 01 - FGSM](attack-01-fgsm.md) | White-box | Low | Fast baseline adversarial testing |
| [Attack 02 - PGD](attack-02-pgd.md) | White-box | Medium | Stronger adversarial robustness evaluation |
| [Attack 03 - Square Attack](attack-03-square.md) | Black-box | Medium | Query-efficient score-based attacks |
| [Attack 04 - Model Extraction](attack-04-extraction.md) | Black-box | High | Model stealing and transfer attacks |

---

## When to Use Which Attack

**For quick adversarial testing:**  
→ Start with [FGSM](attack-01-fgsm.md) — fast, simple, provides baseline robustness metrics

**For rigorous robustness evaluation:**  
→ Use [PGD](attack-02-pgd.md) — iterative optimization produces stronger adversarial examples

**For realistic black-box scenarios:**  
→ Use [Square Attack](attack-03-square.md) — only requires API access, no model knowledge needed

**For model theft and IP protection testing:**  
→ Use [Model Extraction](attack-04-extraction.md) — demonstrates how attackers can steal model behavior

---

## Attack Types

**White-box attacks** (FGSM, PGD)  
Require full access to model architecture, weights, and gradients. Used for evaluating adversarial robustness during development.

**Black-box attacks** (Square Attack, Model Extraction)  
Only require API access to query the model. Simulate realistic attacker scenarios where internal model details are unknown.

---

## Prerequisites

All attacks require:
- Python virtual environment with dependencies installed (`pip install -r requirements.txt`)
- Target API running at `http://localhost:8000` (for API-based attacks)

See individual attack documentation for specific requirements.

---

## Running All Attacks

**Quick start:** Run all four attacks sequentially with a single command.

```bash
python -m attacks.run_all_attacks
```

For detailed options and filtering, see the **[Running All Attacks Guide](running-all-attacks.md)**.

The runner will:
- Check API availability for black-box attacks
- Execute attacks sequentially with progress reporting
- Provide a summary with timing information
- Exit with appropriate status codes

---

## Individual Attack Documentation

- [Attack 01 - FGSM (Fast Gradient Sign Method)](attack-01-fgsm.md)
- [Attack 02 - PGD (Projected Gradient Descent)](attack-02-pgd.md)
- [Attack 03 - Square Attack (Black-box)](attack-03-square.md)
- [Attack 04 - Model Extraction](attack-04-extraction.md)
