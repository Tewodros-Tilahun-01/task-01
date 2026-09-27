"""
attack_03.py — Square Attack (Black-Box) against the CIFAR-10 classifier.

What this script does:
1. Loads test images from CIFAR-10
2. Runs Square Attack using ONLY API queries (no model access)
3. Tests different query budgets (50, 200, 500)
4. Measures attack success rate and query efficiency
5. Saves side-by-side comparison images and plots to evidence/

Run from the task-01 root with the venv active:
    python -m attacks.attack_03

Note: Requires the target API running at localhost:8000
"""

import pathlib
import sys
import time

import numpy as np
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
    save_comparison,
    save_log,
    send_to_api,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EPSILON = 8/255

QUERY_BUDGETS = [50, 200, 500]

N_IMAGES = 200

N_SAVE = 10

API_URL = "http://localhost:8000/predict"

ADV_DIR = EVIDENCE_ADV / "square"
LOG_DIR = EVIDENCE_LOGS / "square"


# ---------------------------------------------------------------------------
# API Model Wrapper
# ---------------------------------------------------------------------------

class APIModelWrapper(torch.nn.Module):
    """
    Wraps API calls to look like a PyTorch model for torchattacks.Square.
    This enables true black-box attacks using only the prediction API.
    """
    
    def __init__(self, api_url):
        super().__init__()
        self.api_url = api_url
        self.query_count = 0  # Track total API queries made
    
    def forward(self, images):
        """Query API for each image and return pseudo-logits."""
        all_logits = []
        
        for i in range(images.shape[0]):
            self.query_count += 1
            response = send_to_api(images[i], url=self.api_url)
            
            if "error" in response:
                logits = torch.zeros(10)
            else:
                # Convert API probabilities to pseudo-logits using log
                probs = [response["probabilities"][str(j)] for j in range(10)]
                logits = torch.tensor([np.log(max(p, 1e-10)) for p in probs])
            
            all_logits.append(logits)
        
        return torch.stack(all_logits)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def run_square_attack(model_wrapper, images, labels, query_budget, device):
    """Run Square Attack on images and return results."""
    budget_start = model_wrapper.query_count
    
    # Initialize Square Attack with fixed seed for reproducible results
    attack = torchattacks.Square(
        model_wrapper, norm="Linf", eps=EPSILON, n_queries=query_budget,
        n_restarts=1, p_init=0.8, loss="margin", resc_schedule=True, seed=42
    )
    attack.set_normalization_used(mean=MEAN, std=STD)
    
    results = []
    
    for i in range(len(images)):
        image = images[i:i+1].to(device)
        label = labels[i:i+1].to(device)
        
        print(f"    Image {i:2d} (true={labels[i].item()}, {CIFAR10_CLASSES[labels[i].item()]:>10s}): ", 
              end="", flush=True)
        
        img_start = model_wrapper.query_count
        
        # get clean prediction first
        with torch.no_grad():
            clean_pred = model_wrapper(image).argmax(dim=1).item()
        
        # only attack images the model got right
        if clean_pred != labels[i].item():
            print(f"SKIP (already wrong: pred={clean_pred}/{CIFAR10_CLASSES[clean_pred]})")
            continue
        
        # run attack
        start_time = time.time()
        adv_image = attack(image, label)
        elapsed = time.time() - start_time
        
        # get adversarial prediction
        with torch.no_grad():
            adv_pred = model_wrapper(adv_image).argmax(dim=1).item()
        
        # total queries includes: clean check + attack queries + final check
        queries = model_wrapper.query_count - img_start
        success = (adv_pred != labels[i].item())
        
        results.append({
            "index": i,
            "true_label": labels[i].item(),
            "true_class": CIFAR10_CLASSES[labels[i].item()],
            "clean_pred": clean_pred,
            "clean_class": CIFAR10_CLASSES[clean_pred],
            "adv_pred": adv_pred,
            "adv_class": CIFAR10_CLASSES[adv_pred],
            "clean_image": image.squeeze(0).cpu(),
            "adv_image": adv_image.squeeze(0).cpu(),
            "success": success,
            "queries_used": queries,
            "elapsed_time": elapsed,
        })
        
        if success:
            print(f"SUCCESS in {queries:4d} queries (pred: {adv_pred}/{CIFAR10_CLASSES[adv_pred]:>10s}) [{elapsed:.1f}s]")
        else:
            print(f"FAILED  after {queries:4d} queries (pred: {adv_pred}/{CIFAR10_CLASSES[adv_pred]:>10s}) [{elapsed:.1f}s]")
    
    total_queries = model_wrapper.query_count - budget_start
    return results, total_queries


def save_evidence(results, query_budget):
    """Save comparison images and API logs for one query budget."""
    budget_str = f"queries_{query_budget}"
    budget_dir = ADV_DIR / budget_str
    log_dir = LOG_DIR / budget_str
    budget_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # sort successful attacks by query efficiency (fewer queries first)
    successful = sorted([r for r in results if r["success"]], 
                       key=lambda r: r["queries_used"])
    
    api_log = []
    saved = 0
    
    for result in successful:
        if saved >= N_SAVE:
            break
        
        out_path = save_comparison(
            clean=result["clean_image"], adv=result["adv_image"],
            true_label=result["true_label"], clean_pred=result["clean_pred"],
            adv_pred=result["adv_pred"], epsilon=EPSILON, index=result["index"],
            out_dir=budget_dir, attack_name=f"Square-{query_budget}q",
        )
        print(f"    Saved: {out_path.name}")
        
        api_log.append({
            "index": result["index"],
            "true_label": result["true_label"],
            "true_class": result["true_class"],
            "clean_pred": result["clean_pred"],
            "clean_class": result["clean_class"],
            "adv_pred": result["adv_pred"],
            "adv_class": result["adv_class"],
            "queries_used": result["queries_used"],
            "elapsed_time": round(result["elapsed_time"], 2),
            "attack_success": result["success"],
        })
        saved += 1
    
    log_path = log_dir / "attack_results.json"
    save_log({
        "epsilon": EPSILON,
        "epsilon_str": "8/255",
        "query_budget": query_budget,
        "results": api_log,
    }, log_path)
    print(f"    Log saved: {log_path}")


def calculate_statistics(results):
    """Calculate summary statistics."""
    total = len(results)
    successful = [r for r in results if r["success"]]
    n_success = len(successful)
    
    success_rate = n_success / total if total > 0 else 0.0
    avg_queries = np.mean([r["queries_used"] for r in successful]) if n_success > 0 else 0
    median_queries = np.median([r["queries_used"] for r in successful]) if n_success > 0 else 0
    
    return {
        "total_images": total,
        "successful_attacks": n_success,
        "success_rate": round(success_rate, 4),
        "clean_acc": round(1.0, 4),  # all tested images were correctly classified initially
        "adv_acc": round((total - n_success) / total, 4) if total > 0 else 0.0,
        "avg_queries": round(avg_queries, 1),
        "median_queries": int(median_queries),
        "total_time": round(sum(r["elapsed_time"] for r in results), 1),
    }


def plot_results(summary):
    """Save success rate and accuracy plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    ADV_DIR.mkdir(parents=True, exist_ok=True)
    
    query_budgets = [r["query_budget"] for r in summary]
    query_strs = [str(q) for q in query_budgets]
    success_rates = [r["success_rate"] for r in summary]
    clean_accs = [r["clean_acc"] for r in summary]
    adv_accs = [r["adv_acc"] for r in summary]
    x_pos = list(range(len(query_strs)))
    
    # plot 1 — success rate
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_pos, success_rates, marker="o", color="red", linewidth=2, markersize=8)
    ax.set_xlabel("Query Budget", fontsize=11)
    ax.set_ylabel("Attack Success Rate", fontsize=11)
    ax.set_title("Square Attack — Success Rate vs Query Budget", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(query_strs)
    ax.grid(True, alpha=0.3, linestyle="--")
    
    for i, (x, y) in enumerate(zip(x_pos, success_rates)):
        offset_y = 10 if y < 0.95 else -20
        ax.annotate(f"{y:.1%}", (x, y), textcoords="offset points",
                    xytext=(0, offset_y), ha="center", fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.7))
    fig.tight_layout()
    path1 = ADV_DIR / "square_success_rate.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"\n  Plot saved: {path1}")
    
    # plot 2 — accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_pos, clean_accs, marker="s", label="Clean accuracy",
            color="steelblue", linewidth=2, markersize=8)
    ax.plot(x_pos, adv_accs, marker="o", label="Adversarial accuracy",
            color="red", linewidth=2, markersize=8)
    ax.set_xlabel("Query Budget", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Square Attack — Clean vs Adversarial Accuracy", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(query_strs)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    path2 = ADV_DIR / "square_accuracy.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {path2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Attack 03 — Square Attack (Black-Box via API)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    
    model_wrapper = APIModelWrapper(API_URL).to(device)
    model_wrapper.eval()
    
    print(f"\nLoading {N_IMAGES} test images …")
    loader = get_test_loader(batch_size=N_IMAGES)
    images, labels = next(iter(loader))
    print(f"  Batch shape : {images.shape}")
    
    summary = []
    print(f"\nRunning Square Attack at query budgets: {QUERY_BUDGETS}\n")
    
    for query_budget in QUERY_BUDGETS:
        print(f"  Budget = {query_budget}")
        results, total_queries = run_square_attack(model_wrapper, images, labels, query_budget, device)
        
        stats = calculate_statistics(results)
        stats["query_budget"] = query_budget
        stats["total_api_calls"] = total_queries
        summary.append(stats)
        
        print(f"\n  Results:")
        print(f"    Success rate:    {stats['success_rate']:.1%} ({stats['successful_attacks']}/{stats['total_images']})")
        print(f"    Avg queries:     {stats['avg_queries']:.0f}")
        print(f"    Median queries:  {stats['median_queries']}")
        print(f"    Total time:      {stats['total_time']:.1f}s")
        print(f"    Total API calls: {total_queries}")
        
        print(f"\n  Saving evidence:")
        save_evidence(results, query_budget)
        print()
    
    # save summary
    summary_path = LOG_DIR / "square_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    save_log({
        "attack": "Square Attack",
        "epsilon": EPSILON,
        "epsilon_str": "8/255",
        "n_images": N_IMAGES,
        "query_budgets": QUERY_BUDGETS,
        "results": summary,
    }, summary_path)
    print(f"  Summary saved: {summary_path}")
    
    plot_results(summary)
    
    # print final table
    print("\n" + "=" * 70)
    print(f"  {'Budget':<12} {'Success Rate':<15} {'Avg Queries':<15} {'Time'}")
    print("  " + "-" * 65)
    for s in summary:
        print(
            f"  {s['query_budget']:<12}"
            f"  {s['success_rate']:.2%}        "
            f"  {s['avg_queries']:<10.0f}     "
            f"  {s['total_time']:.1f}s"
        )
    print("=" * 70)
    print("\nDone. Evidence saved to:")
    print(f"  Images : {ADV_DIR}")
    print(f"  Logs   : {LOG_DIR}")


if __name__ == "__main__":
    main()
