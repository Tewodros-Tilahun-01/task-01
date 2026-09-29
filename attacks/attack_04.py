"""Model Extraction attack against CIFAR-10 classifier via API queries.

Demonstrates realistic black-box model stealing:
1. Query victim API with test images
2. Train surrogate using API responses (different architecture)
3. Measure agreement with victim
4. Test adversarial transferability

Run from task-01 root: python -m attacks.attack_04
Requires victim API running at localhost:8000
"""

import pathlib
import random
import sys
import time

import numpy as np
import torch

# Path setup
_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from attacks.model_extraction import (                      # noqa: E402
    collect_query_data,
    craft_surrogate_adversarials,
    evaluate_agreement,
    save_training_results,
    test_transferability,
    train_surrogate,
)
from attacks.utils import (                                 # noqa: E402
    EVIDENCE_ADV,
    EVIDENCE_LOGS,
    get_test_loader,
    save_log,
)

# Configuration

# Query configuration
N_QUERY_IMAGES = 10000
API_URL = "http://localhost:8000/predict"

# Training config
SURROGATE_EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.001
RANDOM_SEED = 42

# Transferability test config
EPSILON = 8/255
ATTACK_TYPE = "pgd"
N_TRANSFER_TEST = 200
N_SAVE_EXAMPLES = 10

# Output directories
ADV_DIR = EVIDENCE_ADV / "model_extraction"
LOG_DIR = EVIDENCE_LOGS / "model_extraction"


def main() -> None:
    print("=" * 60)
    print("  Attack 04 — Model Extraction via API Queries")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Query dataset size : {N_QUERY_IMAGES}")
    print(f"  Training epochs    : {SURROGATE_EPOCHS}")
    print(f"  Batch size         : {BATCH_SIZE}")
    print(f"  Learning rate      : {LEARNING_RATE}")
    print(f"  Transfer test size : {N_TRANSFER_TEST}")
    print(f"  FGSM epsilon       : {EPSILON} (8/255)")
    print(f"  Random seed        : {RANDOM_SEED}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device             : {device}\n")
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
   
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    attack_start = time.time()
    
    print("=" * 60)
    print("Step 1: Querying victim API to collect training data")
    print("=" * 60)
    
    loader = get_test_loader(batch_size=N_QUERY_IMAGES)
    
    query_start = time.time()
    query_data = collect_query_data(
        test_loader=loader,
        api_url=API_URL,
        n_images=N_QUERY_IMAGES,
        device=device,
    )
    query_time = time.time() - query_start
    
    if len(query_data) == 0:
        print("\n✗ No data collected. Is API running?")
        sys.exit(1)
    
    print(f"\n✓ Collected {len(query_data)} labeled samples in {query_time:.1f}s")
    print(f"  Average: {len(query_data)/query_time:.1f} queries/second")

    print("\n" + "=" * 60)
    print("Step 2: Training surrogate model with knowledge distillation")
    print("=" * 60)
    
    # To skip training and load a pre-trained model instead, comment out the training
    # block below and uncomment these lines:
    #
    # from attacks.model_extraction.surrogate_arch import RealisticSurrogate
    # surrogate = RealisticSurrogate(num_classes=10)
    # model_path = ADV_DIR / "surrogate_model.pth"
    # surrogate.load_state_dict(torch.load(model_path, map_location=device))
    # surrogate.to(device)
    # surrogate.eval()
    # print(f"\n✓ Loaded pre-trained surrogate from {model_path}")
    # loss_history = []  # Empty since we didn't train
    # train_time = 0
    
    train_start = time.time()
    surrogate, loss_history = train_surrogate(
        query_data=query_data,
        epochs=SURROGATE_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        device=device,
        seed=RANDOM_SEED,
    )
    train_time = time.time() - train_start
    
    print(f"\n✓ Surrogate training completed in {train_time:.1f}s")
    print(f"  Final loss: {loss_history[-1]:.4f}")
    
    ADV_DIR.mkdir(parents=True, exist_ok=True)
    model_path, plot_path = save_training_results(
        model=surrogate,
        loss_history=loss_history,
        output_dir=ADV_DIR,
    )
    print(f"  Model saved : {model_path}")
    print(f"  Plot saved  : {plot_path}")

    print("\n" + "=" * 60)
    print("Step 3: Evaluating surrogate agreement with victim")
    print("=" * 60)
    
    eval_loader = get_test_loader(batch_size=1000)
    eval_images, eval_labels = next(iter(eval_loader))
    
    agreement_rate = evaluate_agreement(
        surrogate=surrogate,
        test_images=eval_images,
        test_labels=eval_labels,
        api_url=API_URL,
        device=device,
        n_samples=1000,
    )
    
    print(f"\n✓ Agreement rate: {agreement_rate:.2%}")
    print(f"  Surrogate matches victim predictions on {agreement_rate:.1%} of test images")

    print("\n" + "=" *60)
    print("Step 4: Testing adversarial transferability")
    print("=" * 60)
    print(f"Crafting {ATTACK_TYPE.upper()} adversarial examples on surrogate (ε={EPSILON:.4f})...")
    
    transfer_loader = get_test_loader(batch_size=N_TRANSFER_TEST)
    transfer_images, transfer_labels = next(iter(transfer_loader))
    
    adv_images = craft_surrogate_adversarials(
        surrogate=surrogate,
        images=transfer_images,
        labels=transfer_labels,
        epsilon=EPSILON,
        device=device,
        attack_type=ATTACK_TYPE,
    )
    print(f"✓ Generated {len(adv_images)} adversarial examples\n")
    
    print("Testing transfer to victim API...")
    transfer_metrics = test_transferability(
        surrogate=surrogate,
        adv_images=adv_images,
        clean_images=transfer_images,
        labels=transfer_labels,
        api_url=API_URL,
        output_dir=ADV_DIR,
        n_save=N_SAVE_EXAMPLES,
    )
    
    print(f"\n✓ Transfer attack results:")
    print(f"  Success rate : {transfer_metrics['success_rate']:.2%}")
    print(f"  Successes    : {transfer_metrics['successes']}/{transfer_metrics['total']}")
    print(f"  Examples saved: {transfer_metrics['n_saved']}")

    attack_time = time.time() - attack_start
    
    summary = {
        "attack": "Model Extraction",
        "api_url": API_URL,
        "random_seed": RANDOM_SEED,
        "surrogate_architecture": "RealisticSurrogate",
        "note": "Surrogate uses different architecture than victim (realistic black-box)",
        "queries_sent": len(query_data),
        "query_time_seconds": round(query_time, 1),
        "queries_per_second": round(len(query_data) / query_time, 1),
        "training": {
            "epochs": SURROGATE_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "final_loss": round(loss_history[-1], 4),
            "training_time_seconds": round(train_time, 1),
        },
        "agreement_rate": round(agreement_rate, 4),
        "transferability": {
            "attack_type": ATTACK_TYPE,
            "epsilon": EPSILON,
            "epsilon_str": "8/255",
            "test_size": N_TRANSFER_TEST,
            "success_rate": round(transfer_metrics['success_rate'], 4),
            "successes": transfer_metrics['successes'],
            "total": transfer_metrics['total'],
        },
        "total_attack_time_seconds": round(attack_time, 1),
    }
    
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = LOG_DIR / "extraction_summary.json"
    save_log(summary, summary_path)
    
    print("\n" + "=" * 60)
    print("ATTACK SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<30} {'Value':<20}")
    print("-" * 50)
    print(f"{'Total queries':<30} {len(query_data)}")
    print(f"{'Query time':<30} {query_time:.1f}s")
    print(f"{'Training epochs':<30} {SURROGATE_EPOCHS}")
    print(f"{'Training time':<30} {train_time:.1f}s")
    print(f"{'Final training loss':<30} {loss_history[-1]:.4f}")
    print(f"{'Agreement rate':<30} {agreement_rate:.2%}")
    print(f"{'Transfer success rate':<30} {transfer_metrics['success_rate']:.2%}")
    print(f"{'Transfer successes':<30} {transfer_metrics['successes']}/{transfer_metrics['total']}")
    print(f"{'Total attack time':<30} {attack_time:.1f}s")
    print("=" * 60)
    
    print("\n✓ Attack complete. Evidence saved to:")
    print(f"  Model & plots : {ADV_DIR}")
    print(f"  Logs          : {LOG_DIR}")
    print(f"  Summary       : {summary_path}")


if __name__ == "__main__":
    main()
