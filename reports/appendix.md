# Appendix

## A. Evidence Index

| Category | Location | Count |
|----------|----------|-------|
| FGSM images | `evidence/adversarial/fgsm/eps_*/` | 40 |
| PGD images | `evidence/adversarial/pgd/eps_*/` | 40 |
| Square examples | `evidence/adversarial/square/queries_*/` | 30 |
| Model extraction | `evidence/adversarial/model_extraction/` | 10 transfer examples |
| Logs | `evidence/logs/*/` | JSON summaries |

## B. Quick Reproduce

```bash
source .venv/bin/activate

# Terminal 1: Run target
docker run -p 127.0.0.1:8000:8000 \
    -e MODEL_SHA256=c83f56d8354266c487c0a537d4c44e56149f27fcf8950f469b279ecf340d5929 \
    -e MODEL_INFO_SHA256=b8435f91a1e1f5a9e96ea0c12c2f1fa29c8857221038e0eddb6759a701cbe6c8 \
    cifar10-target:latest

# Terminal 2: Run attacks
python -m attacks.attack_01  # FGSM
python -m attacks.attack_02  # PGD  
python -m attacks.attack_03  # Square
python -m attacks.attack_04  # Extraction
```

## C. Model

For complete model architecture and training details, see [Methodology](methodology.md).

**Summary:** CifarCNN (3 conv blocks + 2 dense layers) trained on CIFAR-10 for 20 epochs with Adam optimizer (lr=0.001). Test accuracy: 83.5%

## D. MITRE ATLAS

See [Methodology](methodology.md) for comprehensive MITRE ATLAS framework mapping.

**Model Extraction Details:**
- **Techniques:** AML.T0040 (AI Model Inference API Access) → AML.T0024.002 (Extract AI Model)
- **Method:** Query-based model stealing via knowledge distillation
- **Results:** 80.3% behavioral fidelity via 10,000 API queries; enables 80.12% adversarial transfer

## E. References

- MITRE ATLAS: https://atlas.mitre.org/
- NIST AI RMF: https://www.nist.gov/itl/ai-risk-management-framework
- torchattacks: https://github.com/Harry24k/adversarial-attacks-pytorch

## F. Disclaimer

Local testing only. Defensive research use only. Do not use for unauthorized access.
