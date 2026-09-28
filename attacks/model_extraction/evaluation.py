"""Evaluate surrogate performance and test adversarial transferability."""

import pathlib
import sys

import torch
import torch.nn as nn
import torchattacks

# Path setup
_ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from attacks.utils import MEAN, STD, save_comparison, send_to_api  # noqa: E402


def evaluate_agreement(
    surrogate: nn.Module,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    api_url: str,
    device: torch.device,
    n_samples: int = 1000,
) -> float:
    """Measure agreement between surrogate and victim predictions."""
    surrogate.eval()
    
    test_images = test_images[:n_samples]
    test_labels = test_labels[:n_samples]
    
    print(f"\nEvaluating surrogate-victim agreement...")
    print(f"  Testing {len(test_images)} samples\n")
    
    # Get surrogate predictions
    print("  Computing surrogate predictions...")
    with torch.no_grad():
        test_images_device = test_images.to(device)
        surrogate_logits = surrogate(test_images_device)
        surrogate_preds = surrogate_logits.argmax(dim=1).cpu().numpy()
    print("    Done ✓")
    
    # Get victim predictions
    print("\n  Querying victim API:")
    victim_preds = []
    errors = 0
    
    for i, image in enumerate(test_images):
        response = send_to_api(image, url=api_url)
        
        if "error" in response:
            errors += 1
            victim_preds.append(-1)
        else:
            victim_preds.append(response["predicted_class"])
        
        if (i + 1) % 100 == 0 or i + 1 == len(test_images):
            print(f"    [{i+1:4d}/{len(test_images)}] queried", end="\r")
    print(f"    [{len(test_images):4d}/{len(test_images)}] queried ✓")
    
    if errors > 0:
        print(f"\n    Warning: {errors} API errors encountered")
    
    # Calculate agreement
    victim_preds = torch.tensor(victim_preds)
    surrogate_preds = torch.tensor(surrogate_preds)
    valid = victim_preds != -1
    agreements = (surrogate_preds[valid] == victim_preds[valid]).sum().item()
    total_valid = valid.sum().item()
    
    agreement_rate = agreements / total_valid if total_valid > 0 else 0.0
    
    print(f"\n  Agreement breakdown:")
    print(f"    Valid samples    : {total_valid}")
    print(f"    Agreements       : {agreements}")
    print(f"    Disagreements    : {total_valid - agreements}")
    print(f"    Agreement rate   : {agreement_rate:.2%}")
    
    return agreement_rate


def craft_surrogate_adversarials(
    surrogate: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    device: torch.device,
    attack_type: str = "pgd",
) -> torch.Tensor:
    """Craft adversarial examples on surrogate model.
    
    These will be tested for transferability to victim.
    """
    surrogate.eval()
    
    images = images.to(device)
    labels = labels.to(device)
    
    if attack_type == "pgd":
        attack = torchattacks.PGD(
            surrogate,
            eps=epsilon,
            steps=20,
            random_start=True,
        )
    else:
        attack = torchattacks.FGSM(surrogate, eps=epsilon)
    
    attack.set_normalization_used(mean=MEAN, std=STD)
    adv_images = attack(images, labels)
    
    return adv_images.cpu()


def test_transferability(
    surrogate: nn.Module,
    adv_images: torch.Tensor,
    clean_images: torch.Tensor,
    labels: torch.Tensor,
    api_url: str,
    output_dir: pathlib.Path,
    n_save: int = 5,
) -> dict:
    """Test if surrogate-crafted adversaries transfer to victim."""
    surrogate.eval()
    
    print(f"\nTesting transfer to victim API...")
    print(f"  Testing {len(adv_images)} adversarial samples\n")
    
    # Query victim with clean images
    print("  Querying clean images:")
    victim_clean_preds = []
    for i, image in enumerate(clean_images):
        response = send_to_api(image, url=api_url)
        if "error" in response:
            victim_clean_preds.append(-1)
        else:
            victim_clean_preds.append(response["predicted_class"])
        
        if (i + 1) % 50 == 0 or i + 1 == len(clean_images):
            print(f"    [{i+1:3d}/{len(clean_images)}] queried", end="\r")
    print(f"    [{len(clean_images):3d}/{len(clean_images)}] queried ✓")
    
    # Query victim with adversarial images
    print("\n  Querying adversarial images:")
    victim_adv_preds = []
    for i, image in enumerate(adv_images):
        response = send_to_api(image, url=api_url)
        if "error" in response:
            victim_adv_preds.append(-1)
        else:
            victim_adv_preds.append(response["predicted_class"])
        
        if (i + 1) % 50 == 0 or i + 1 == len(adv_images):
            print(f"    [{i+1:3d}/{len(adv_images)}] queried", end="\r")
    print(f"    [{len(adv_images):3d}/{len(adv_images)}] queried ✓")
    
    # Calculate transfer success
    victim_clean_preds = torch.tensor(victim_clean_preds)
    victim_adv_preds = torch.tensor(victim_adv_preds)
    labels = labels.cpu()
    victim_correct = (victim_clean_preds == labels) & (victim_clean_preds != -1)
    transfer_success = victim_correct & (victim_adv_preds != labels)
    
    n_successes = transfer_success.sum().item()
    n_total = victim_correct.sum().item()
    success_rate = n_successes / n_total if n_total > 0 else 0.0
    
    print(f"\n  Transfer results:")
    print(f"    Victim-correct samples : {n_total}")
    print(f"    Successful transfers   : {n_successes}")
    print(f"    Transfer success rate  : {success_rate:.2%}")
    
    # Save examples
    print(f"\n  Saving top {n_save} successful transfer examples...")
    success_indices = torch.where(transfer_success)[0].tolist()
    saved = 0
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx in success_indices[:n_save]:
        out_path = save_comparison(
            clean=clean_images[idx],
            adv=adv_images[idx],
            true_label=labels[idx].item(),
            clean_pred=victim_clean_preds[idx].item(),
            adv_pred=victim_adv_preds[idx].item(),
            epsilon=8/255,
            index=idx,
            out_dir=output_dir,
            attack_name="Transfer",
        )
        print(f"    Saved: {out_path.name}")
        saved += 1
    
    return {
        "success_rate": success_rate,
        "successes": n_successes,
        "total": n_total,
        "n_saved": saved,
    }
