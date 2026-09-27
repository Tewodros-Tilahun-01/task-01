"""Train surrogate model using knowledge distillation from victim API."""

import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Path setup
_ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from attacks.model_extraction.surrogate_arch import RealisticSurrogate  # noqa: E402


def create_distillation_loader(
    query_data: list[dict],
    batch_size: int,
    seed: int = 42,
) -> DataLoader:
    """Convert query data into DataLoader for training."""
    images = torch.stack([item["image"] for item in query_data])
    probs = torch.tensor([item["victim_probs"] for item in query_data], dtype=torch.float32)
    dataset = TensorDataset(images, probs)
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        generator=generator,
    )
    
    return loader


def train_surrogate(
    query_data: list[dict],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int = 42,
) -> tuple[nn.Module, list[float]]:
    """Train surrogate model using knowledge distillation.
    
    Returns trained model and loss history.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    print(f"\nInitializing realistic surrogate model (seed={seed})...")
    print(f"  Architecture: RealisticSurrogate (3 conv blocks)")
    print(f"  Note: Different from victim's exact architecture")
    model = RealisticSurrogate(num_classes=10).to(device)
    model.train()
    train_loader = create_distillation_loader(query_data, batch_size, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    
    loss_history = []
    
    print(f"Training for {epochs} epochs...")
    print(f"  Dataset size : {len(query_data)}")
    print(f"  Batch size   : {batch_size}")
    print(f"  Batches/epoch: {len(train_loader)}")
    print(f"  Learning rate: {lr}")
    print()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        
        for batch_images, batch_victim_probs in train_loader:
            batch_images = batch_images.to(device)
            batch_victim_probs = batch_victim_probs.to(device)
            
            logits = model(batch_images)
            log_probs = F.log_softmax(logits, dim=1)
            loss = criterion(log_probs, batch_victim_probs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"  Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f}")
    
    print(f"\n✓ Training completed")
    model.eval()
    
    return model, loss_history


def save_training_results(
    model: nn.Module,
    loss_history: list[float],
    output_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Save trained model and training loss plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "surrogate_model.pth"
    torch.save(model.state_dict(), model_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = list(range(1, len(loss_history) + 1))
    ax.plot(epochs, loss_history, marker='o', linewidth=2, markersize=4, color='steelblue')
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("KL Divergence Loss", fontsize=11)
    ax.set_title("Surrogate Model Training — Knowledge Distillation Loss", 
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    
    final_loss = loss_history[-1]
    ax.annotate(
        f"Final: {final_loss:.4f}",
        xy=(len(loss_history), final_loss),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="steelblue", alpha=0.8),
        arrowprops=dict(arrowstyle="->", color="steelblue"),
    )
    
    fig.tight_layout()
    plot_path = output_dir / "training_loss.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    
    return model_path, plot_path
