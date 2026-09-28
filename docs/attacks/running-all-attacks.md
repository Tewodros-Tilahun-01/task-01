# Running All Attacks

A Python script is provided to execute all four attacks sequentially with progress tracking, error handling, and comprehensive reporting.

---

## Usage

**Location:** `attacks/run_all_attacks.py`

### Basic Usage

```bash
# Run all four attacks
python -m attacks.run_all_attacks

# Run only white-box attacks (01: FGSM, 02: PGD)
python -m attacks.run_all_attacks --whitebox

# Run only black-box attacks (03: Square, 04: Model Extraction)
python -m attacks.run_all_attacks --blackbox
```

### Features

- **API Health Check** — Automatically detects if the target API is running before starting black-box attacks
- **Progress Reporting** — Shows real-time status for each attack with duration tracking
- **Error Handling** — Continues execution even if one attack fails, reports all results at the end
- **Execution Summary** — Displays a formatted table with status and timing for each attack
- **Exit Codes** — Returns 0 if all attacks succeed, 1 if any fail
- **Filtering Options** — Run only white-box (`--whitebox`) or black-box (`--blackbox`) attacks

### Example Output

```
======================================================================
  Adversarial Attack Suite — Sequential Execution
======================================================================

Start time: 2024-01-15 14:30:00
Attacks to run: 4

Checking API availability at http://localhost:8000 ...
✓ API is running and healthy

──────────────────────────────────────────────────────────────────────
  Attack 01 — FGSM (Fast Gradient Sign Method)
──────────────────────────────────────────────────────────────────────

[... attack output ...]

✓ Attack 01 completed successfully
  Duration: 45.2 seconds

──────────────────────────────────────────────────────────────────────
  Attack 02 — PGD (Projected Gradient Descent)
──────────────────────────────────────────────────────────────────────

[... attack output ...]

✓ Attack 02 completed successfully
  Duration: 1.8 minutes

[... continues for all attacks ...]

======================================================================
  Attack Execution Summary
======================================================================

Total attacks executed: 4
Successful: 4
Failed: 0
Total duration: 12.5 minutes

Attack                              Status       Duration
──────────────────────────────────────────────────────────────────────
01 - FGSM (Fast Gradient Sign Method)    ✓ SUCCESS    45.2 seconds
02 - PGD (Projected Gradient Descent)    ✓ SUCCESS    1.8 minutes
03 - Square Attack (Black-box)           ✓ SUCCESS    6.2 minutes
04 - Model Extraction                    ✓ SUCCESS    3.7 minutes
======================================================================

End time: 2024-01-15 14:42:30

All evidence saved to:
  Images: evidence/adversarial/
  Logs:   evidence/logs/
```

---

## Prerequisites

Both runners require:

1. **Python virtual environment active**
   ```bash
   source .venv/bin/activate
   ```

2. **Dependencies installed**
   ```bash
   pip install -r requirements.txt
   ```

3. **For black-box attacks only** — Target API running at `http://localhost:8000`
   ```bash
   docker run -p 127.0.0.1:8000:8000 \
     -e MODEL_SHA256=<hash> \
     -e MODEL_INFO_SHA256=<hash> \
     cifar10-target:latest
   ```

---

## Execution Order

Attacks run in this order:

1. **Attack 01 - FGSM** (~45 seconds)
   - White-box, fast baseline adversarial attack
   
2. **Attack 02 - PGD** (~2 minutes)
   - White-box, iterative optimization attack
   
3. **Attack 03 - Square Attack** (~6 minutes)
   - Black-box, query-based attack
   - Requires API
   
4. **Attack 04 - Model Extraction** (~4 minutes)
   - Black-box, model stealing attack
   - Requires API

**Total time:** ~12-15 minutes for all four attacks

---

## Filtering Options

### White-box Only (No API Required)

Run only attacks that work against the local model without API access:

```bash
python -m attacks.run_all_attacks --whitebox
```

**Runs:** Attack 01 (FGSM), Attack 02 (PGD)  
**Time:** ~3 minutes  
**Use case:** Quick testing, model robustness evaluation, no Docker needed

### Black-box Only (API Required)

Run only attacks that use the API:

```bash
python -m attacks.run_all_attacks --blackbox
```

**Runs:** Attack 03 (Square), Attack 04 (Model Extraction)  
**Time:** ~10 minutes  
**Use case:** API security testing, production readiness evaluation

---

## Error Handling

### What Happens if an Attack Fails?

The runner catches exceptions and continues to the next attack, then reports all errors in the summary table. It exits with code 1 if any attack failed, making it suitable for CI/CD integration.

### What if the API is Down?

The runner will:
1. Detect the API is unavailable
2. Prompt you to either:
   - Start the API and continue, or
   - Skip black-box attacks and run white-box only, or
   - Abort execution

---

## Output Locations

All evidence is saved to:

```
evidence/
├── adversarial/
│   ├── fgsm/         # Attack 01 images
│   ├── pgd/          # Attack 02 images
│   ├── square/       # Attack 03 images
│   └── model_extraction/  # Attack 04 images and surrogate model
└── logs/
    ├── fgsm/         # Attack 01 JSON logs
    ├── pgd/          # Attack 02 JSON logs
    ├── square/       # Attack 03 JSON logs
    └── model_extraction/  # Attack 04 JSON logs
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'attacks'"

Make sure you're running from the project root (`task-01/`):

```bash
cd /path/to/task-01
python -m attacks.run_all_attacks
```

### "Connection refused" or "API not available"

The Docker container is not running. Start it:

```bash
docker run -p 127.0.0.1:8000:8000 \               127 err  05:00:52 PM 
    -e MODEL_SHA256=c83f56d8354266c487c0a537d4c44e56149f27fcf8950f469b279ecf340d5929 \
    -e MODEL_INFO_SHA256=b8435f91a1e1f5a9e96ea0c12c2f1fa29c8857221038e0eddb6759a701cbe6c8 \
    cifar10-target:latest
```

### Attacks take too long

This is expected — the full suite takes ~12-15 minutes. Consider:
- Run `--whitebox` only (~3 minutes)
- Reduce N_IMAGES in individual attack scripts
- Run attacks in parallel (not recommended for first-time testing)

---

## See Also

- [Attack Overview](overview.md) — Comparison of all four attacks
- [Attack 01 - FGSM](attack-01-fgsm.md) — Fast Gradient Sign Method
- [Attack 02 - PGD](attack-02-pgd.md) — Projected Gradient Descent
- [Attack 03 - Square Attack](attack-03-square.md) — Black-box query-based
- [Attack 04 - Model Extraction](attack-04-extraction.md) — Model stealing
