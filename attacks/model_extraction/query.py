"""Query victim API to collect training data for surrogate model."""

import pathlib
import sys
import time

import torch
from torch.utils.data import DataLoader

# Path setup
_ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from attacks.utils import send_to_api  # noqa: E402


def collect_query_data(
    test_loader: DataLoader,
    api_url: str,
    n_images: int,
    device: torch.device,
) -> list[dict]:
    """Query victim API to collect labeled training data.
    
    Returns list of dicts with image, true_label, and victim_probs.
    """
    query_data = []
    errors = 0
    start_time = time.time()
    
    print(f"Querying {n_images} images from {api_url}...")
    print("Progress: ", end="", flush=True)
    
    images_collected = 0
    
    for batch_images, batch_labels in test_loader:
        batch_images = batch_images.to(device)
        
        for i in range(len(batch_images)):
            if images_collected >= n_images:
                break
            
            image = batch_images[i]
            true_label = batch_labels[i].item()
            
            response = send_to_api(image, url=api_url)
            if "error" in response:
                errors += 1
                print(f"\n  Warning: API error for image {images_collected}: {response['error']}")
                continue
            victim_probs = [
                response["probabilities"][str(class_id)]
                for class_id in range(10)
            ]
            query_data.append({
                "image": image.cpu(),
                "true_label": true_label,
                "victim_probs": victim_probs,
            })
            
            images_collected += 1
            if images_collected % 500 == 0:
                elapsed = time.time() - start_time
                rate = images_collected / elapsed
                eta = (n_images - images_collected) / rate if rate > 0 else 0
                print(f"\r  [{images_collected:>5}/{n_images}] "
                      f"{rate:.1f} queries/s, "
                      f"ETA: {eta:.0f}s", end="", flush=True)
        
        if images_collected >= n_images:
            break
    
    print()
    
    elapsed = time.time() - start_time
    
    if errors > 0:
        print(f"  Note: {errors} API errors encountered")
    
    return query_data
