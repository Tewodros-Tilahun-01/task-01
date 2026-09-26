"""
attack_01.py — FGSM attack against the CIFAR-10 classifier.

What this script does:
1. Loads the trained model
2. Loads 200 test images from CIFAR-10
3. Runs FGSM at four epsilon values
4. Measures how many predictions flipped at each epsilon
5. Sends adversarial images to the API and saves the responses
6. Saves side-by-side comparison images and plots to evidence/

Run from the task-01 root with the venv active:
    python -m attacks.attack_01

Note: start the target container first if you want API evidence.
"""

import pathlib
import sys

import torch
import torchattacks

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from attacks.utils import (         # noqa: E402
    CIFAR10_CLASSES,
    EVIDENCE_ADV,
    EVIDENCE_LOGS,
    MEAN,
    STD,
    get_test_loader,
    load_model,
    plot_results,
    save_comparison,
    save_log,
    send_to_api,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EPSILONS = [2/255, 4/255, 8/255, 16/255]

# How many images to attack
N_IMAGES = 200

# How many comparison images to save per epsilon
N_SAVE = 10

API_URL = "http://localhost:8000/predict"

ADV_DIR = EVIDENCE_ADV / "fgsm"
LOG_DIR = EVIDENCE_LOGS / "fgsm"


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def accuracy(model: torch.nn.Module, images: torch.Tensor,
             labels: torch.Tensor, device: torch.device) -> float:

    """Run the model and return the fraction of correct predictions."""
    with torch.no_grad():
        preds = model(images.to(device)).argmax(dim=1)
    return (preds.cpu() == labels).float().mean().item()


def run_fgsm(model: torch.nn.Module, images: torch.Tensor,
             labels: torch.Tensor, epsilon: float,
             device: torch.device) -> dict:

    """Apply FGSM at the given epsilon and return predictions and results."""
    attack = torchattacks.FGSM(model, eps=epsilon)
    
    # Tell torchattacks about the normalization so it handles it correctly
    attack.set_normalization_used(mean=MEAN, std=STD)

    images = images.to(device)
    labels = labels.to(device)

    # generate adversarial images
    adv_images = attack(images, labels)

    with torch.no_grad():
        clean_preds = model(images).argmax(dim=1).cpu().tolist()
        adv_preds   = model(adv_images).argmax(dim=1).cpu().tolist()

    labels_list = labels.cpu().tolist()

    clean_acc = sum(p == t for p, t in zip(clean_preds, labels_list)) / len(labels_list)

    # filter to only images the model got right on clean input
    correct = [(cp, ap, t) for cp, ap, t in zip(clean_preds, adv_preds, labels_list) if cp == t]

    adv_acc      = sum(ap == t for _, ap, t in correct) / len(correct)
    success_rate = sum(ap != t for _, ap, t in correct) / len(correct)

    return {
        "clean_acc":    round(clean_acc,    4),
        "adv_acc":      round(adv_acc,      4),
        "success_rate": round(success_rate, 4),
        "clean_preds":  clean_preds,
        "adv_preds":    adv_preds,
        "adv_images":   adv_images.cpu(),
        "images":       images.cpu(),
        "labels":       labels_list,
    }


def save_evidence(results: dict, epsilon: float, n_save: int) -> None:
    """Save comparison images and API responses for one epsilon value."""
    images      = results["images"]
    adv_images  = results["adv_images"]
    labels      = results["labels"]
    clean_preds = results["clean_preds"]
    adv_preds   = results["adv_preds"]

    eps_dir = ADV_DIR / f"eps_{epsilon}"
    log_dir = LOG_DIR / f"eps_{epsilon}"
    eps_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # only keep images the model got right on the clean input
    indices = [i for i in range(len(labels)) if clean_preds[i] == labels[i]]

    # sort: successful attacks first
    indices = sorted(indices, key=lambda i: 0 if clean_preds[i] != adv_preds[i] else 1)

    api_log = []
    saved   = 0

    for i in indices:
        if saved >= n_save:
            break

        out_path = save_comparison(
            clean=images[i],
            adv=adv_images[i],
            true_label=labels[i],
            clean_pred=clean_preds[i],
            adv_pred=adv_preds[i],
            epsilon=epsilon,
            index=i,
            out_dir=eps_dir,
            attack_name="FGSM",
        )
        print(f"    Saved comparison: {out_path.name}")

        # send to API and record what it returned
        api_response = send_to_api(
            adv_images[i], url=API_URL,
            filename=f"adv_{i:04d}_eps{epsilon}.png"
        )
        api_log.append({
            "index":          i,
            "true_label":     labels[i],
            "true_class":     CIFAR10_CLASSES[labels[i]],
            "clean_pred":     clean_preds[i],
            "clean_class":    CIFAR10_CLASSES[clean_preds[i]],
            "adv_pred":       adv_preds[i],
            "adv_class":      CIFAR10_CLASSES[adv_preds[i]],
            "attack_success": clean_preds[i] == labels[i] and clean_preds[i] != adv_preds[i],
            "api_response":   api_response,
        })

        saved += 1

    log_path = log_dir / "attack_results.json"
    save_log({"epsilon": epsilon, "results": api_log}, log_path)
    print(f"    API log saved: {log_path}")




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Attack 01 — FGSM (Fast Gradient Sign Method)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")

    model = load_model(device)

    print(f"\nLoading {N_IMAGES} test images …")
    loader = get_test_loader(batch_size=N_IMAGES)
    images, labels = next(iter(loader))
    print(f"  Batch shape : {images.shape}")

    summary = []
    print(f"\nRunning FGSM at epsilons: {EPSILONS}\n")

    for epsilon in EPSILONS:
        print(f"  ε = {epsilon}")
        results = run_fgsm(model, images, labels, epsilon, device)

        print(f"    Clean accuracy  : {results['clean_acc']:.2%}")
        print(f"    Adversarial acc : {results['adv_acc']:.2%}")
        print(f"    Attack success  : {results['success_rate']:.2%}")

        save_evidence(results, epsilon, n_save=N_SAVE)

        summary.append({
            "epsilon":      epsilon,
            "clean_acc":    results["clean_acc"],
            "adv_acc":      results["adv_acc"],
            "success_rate": results["success_rate"],
        })
        print()

    # save summary
    summary_path = LOG_DIR / "fgsm_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    save_log({"attack": "FGSM", "n_images": N_IMAGES, "results": summary}, summary_path)
    print(f"  Summary saved: {summary_path}")

    plot_results(summary, "FGSM", ADV_DIR)

    # print final table
    print("\n" + "=" * 60)
    print(f"  {'Epsilon':<10} {'Clean Acc':<14} {'Adv Acc':<14} {'Success Rate'}")
    print("  " + "-" * 55)
    for r in summary:
        print(
            f"  {r['epsilon']:<10}"
            f"  {r['clean_acc']:.2%}        "
            f"  {r['adv_acc']:.2%}        "
            f"  {r['success_rate']:.2%}"
        )
    print("=" * 60)
    print("\nDone. Evidence saved to:")
    print(f"  Images : {ADV_DIR}")
    print(f"  Logs   : {LOG_DIR}")


if __name__ == "__main__":
    main()
