# Executive Summary

## Engagement Overview

Ethiopian Artificial Intelligence commissioned a red team security assessment of a locally-hosted PyTorch image classifier deployed via Docker and FastAPI. The target system classifies CIFAR-10 images across 10 categories (airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, trucks).

**Scope:** Adversarial ML attacks, pipeline security, metadata leakage  
**Methodology:** MITRE ATLAS tactics mapped against NIST AI 100-2 adversarial ML taxonomy

---

## Findings Summary

**Overall Risk: CRITICAL**

The assessment identified complete compromise of model integrity through adversarial perturbations and significant infrastructure vulnerabilities enabling model extraction. Four successful attack vectors were demonstrated:

1. **White-Box Evasion (FGSM)** — 92% success at imperceptible perturbations (ε=8/255); requires model weight access
2. **White-Box Evasion (PGD)** — 100% success at industry-standard threshold (ε=8/255); demonstrates complete model failure
3. **Black-Box Evasion (Square)** — 87% success with up to 500 API queries (median 31 for successful attacks); demonstrates practical API-only threat at minimal cost
4. **Model Extraction + Transfer** — 10,000 API queries enabled 80% behavioral fidelity; extracted surrogate enables 80% transfer of adversarial examples back to victim and offline attack capability

**Severity:** The model demonstrated **near-zero adversarial robustness** at industry-standard perturbation thresholds (ε=8/255). An attacker with only API access can:
- **Evade classification** with ~31 median queries per image (Square Attack) — practical in minutes
- **Extract and clone the model** with 10,000 queries (~2 minutes) enabling offline attacks
- **Craft transferable adversarial examples** at 80% success rate without model weights or internal access



---

## Business Impact

**Operational Risk:**
- **Evasion attacks** enable adversaries to bypass classification-based access controls, content filters, or automated monitoring systems
- **Model theft** via API queries allows competitors to replicate model behavior without training costs or data access
- **No detection capability** means malicious inputs pass through indistinguishable from legitimate traffic

**Compliance Risk:**
- Deployment of ML systems without robustness evaluation may violate regulatory requirements for AI safety (EU AI Act, emerging NIST guidelines)
- Lack of audit logging and authentication creates compliance gaps for production AI services

**Reputational Risk:**
- Public disclosure of adversarial vulnerability could undermine trust in AI-powered decision systems
- Model extraction demonstrates insufficient IP protection for ML assets

---

## Key Recommendations

### 1. Restrict API Response Verbosity (Priority: CRITICAL)
Remove full probability distributions from responses. Return only top-1 label or rounded confidence scores. This is the cheapest, fastest mitigation and simultaneously degrades both Square Attack (requires full output for iterative optimization) and Model Extraction (needs high-fidelity probabilities for distillation). Single implementation blocks two major attack vectors.

### 2. Add Authentication, Rate Limiting, and Query Logging (Priority: CRITICAL)
Implement per-API-key authentication, rate limit to 100 queries/minute, and log all predictions with timestamps and source. This prevents commodity query attacks and enables detection of extraction attempts. Also remove the `/model/info` endpoint which leaks input shape and class count—attacker-usable information for surrogate training.

### 3. Implement Adversarial Training (Priority: HIGH)
Train the model on adversarial examples generated during training to build robustness against gradient-based attacks. This directly addresses the model's core vulnerability (zero robustness at ε=8/255). Expected trade-off: 5-8% accuracy loss on clean data for 40-50% robustness improvement. This is the only mitigation that fixes the underlying weakness rather than detecting or limiting attacks.

---

## Conclusion

The target system is vulnerable to practical, low-cost attacks requiring only API access. The threat model is realistic: attackers need no model weights, no insider access, and no physical access—only HTTP queries. 

**Black-box attacks are the primary concern:** Square Attack achieved 87% success with commodity query budgets, and model extraction succeeded in under 10 minutes. The offline/API inconsistency reveals additional operational risk: the production model may silently diverge from the offline version in ways that affect predictions on edge cases.

Adversarial robustness must be treated as a core requirement, not an optional enhancement. The recommended mitigations align with NIST AI RMF "Manage" function and MITRE ATLAS mitigations for production ML systems.
