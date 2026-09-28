"""
run_all_attacks.py — Execute all four adversarial attacks sequentially.

What this script does:
1. Runs Attack 01 (FGSM) - Fast white-box baseline
2. Runs Attack 02 (PGD) - Stronger white-box attack
3. Runs Attack 03 (Square Attack) - Black-box query-based
4. Runs Attack 04 (Model Extraction) - Black-box model stealing

Prerequisites:
- Python virtual environment active
- Dependencies installed (pip install -r requirements.txt)
- For attacks 03 & 04: Target API running at http://localhost:8000

Run from the task-01 root with the venv active:
    python -m attacks.run_all_attacks

Options:
    --whitebox     Run only white-box attacks (01 & 02)
    --blackbox     Run only black-box attacks (03 & 04)
"""

import argparse
import pathlib
import sys
import time
from datetime import datetime

import requests

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = "http://localhost:8000"
API_TIMEOUT = 10 # seconds


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def print_banner(message: str, char: str = "=") -> None:
    """Print a formatted banner."""
    width = 70
    print("\n" + char * width)
    print(f"  {message}")
    print(char * width + "\n")


def check_api_availability() -> bool:
    """Check if the target API is running."""
    try:
        response = requests.get(f"{API_URL}", timeout=API_TIMEOUT)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def run_attack(attack_number: int, attack_name: str, module_name: str) -> dict:
    """Run a single attack and return timing information."""
   
    
    start_time = time.time()
    
    try:
        # Dynamically import and run the attack module
        module = __import__(f"attacks.{module_name}", fromlist=["main"])
        module.main()
        
        elapsed = time.time() - start_time
        
        print(f"\n✓ Attack {attack_number:02d} completed successfully")
        print(f"  Duration: {format_duration(elapsed)}")
        
        return {
            "attack_number": attack_number,
            "attack_name": attack_name,
            "module": module_name,
            "status": "success",
            "duration": elapsed,
            "error": None
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        
        print(f"\n✗ Attack {attack_number:02d} failed")
        print(f"  Error: {str(e)}")
        print(f"  Duration: {format_duration(elapsed)}")
        
        return {
            "attack_number": attack_number,
            "attack_name": attack_name,
            "module": module_name,
            "status": "failed",
            "duration": elapsed,
            "error": str(e)
        }


def print_summary(results: list, total_duration: float) -> None:
    """Print a summary of all attack results."""
    print_banner(" ALL Attacks Execution Summary")
    
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    
    print(f"Total attacks executed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total duration: {format_duration(total_duration)}\n")
    
    print(f"{'Attack':<35} {'Status':<12} {'Duration'}")
    print("─" * 70)
    
    for r in results:
        status_symbol = "✓" if r["status"] == "success" else "✗"
        status_text = f"{status_symbol} {r['status'].upper()}"
        
        print(
            f"{r['attack_number']:02d} - {r['attack_name']:<28} "
            f"{status_text:<12} {format_duration(r['duration'])}"
        )
        
        if r["error"]:
            print(f"     Error: {r['error']}")
    
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all adversarial attacks sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all attacks
  python -m attacks.run_all_attacks

  # Run only white-box attacks (no API needed)
  python -m attacks.run_all_attacks --whitebox

  # Run only black-box attacks (requires API)
  python -m attacks.run_all_attacks --blackbox

  # Run all but skip API-dependent attacks if API is down
  python -m attacks.run_all_attacks --whitebox
        """
    )
    
    parser.add_argument(
        "--whitebox",
        action="store_true",
        help="Run only white-box attacks (01 & 02)"
    )
    
    parser.add_argument(
        "--blackbox",
        action="store_true",
        help="Run only black-box attacks (03 & 04)"
    )
    
    args = parser.parse_args()
    
    # Define all attacks
    all_attacks = [
        (1, "FGSM (Fast Gradient Sign Method)", "attack_01", "whitebox"),
        (2, "PGD (Projected Gradient Descent)", "attack_02", "whitebox"),
        (3, "Square Attack (Black-box)", "attack_03", "blackbox"),
        (4, "Model Extraction", "attack_04", "blackbox"),
    ]
    
    # Filter attacks based on arguments
    attacks_to_run = []
    
    if args.whitebox:
        attacks_to_run = [a for a in all_attacks if a[3] == "whitebox"]
    elif args.blackbox:
        attacks_to_run = [a for a in all_attacks if a[3] == "blackbox"]
    else:
        attacks_to_run = all_attacks
    
    # Print header
    print_banner("Adversarial Attack Suite — Sequential Execution")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check API availability if needed
    needs_api = any(a[3] == "blackbox" for a in attacks_to_run)
    
    if needs_api:
        print(f"\nChecking API availability at {API_URL} ...")
        api_available = check_api_availability()
        
        if api_available:
            print("✓ API is running and healthy")
        else:
            print("✗ API is not available")
            print("\nBlack-box attacks require the target API.")
            print("Start the API with:")
            print("  docker run -p 127.0.0.1:8000:8000 \\")
            print("    -e MODEL_SHA256=<hash> \\")
            print("    -e MODEL_INFO_SHA256=<hash> \\")
            print("    cifar10-target:latest")
            
            # If user explicitly requested black-box only, exit
            if args.blackbox:
                print("\nCannot run black-box attacks without API. Exiting.")
                sys.exit(1)
            
            # Otherwise, filter out black-box attacks and continue with white-box only
            print("\nSkipping black-box attacks. Running white-box attacks only.")
            attacks_to_run = [a for a in attacks_to_run if a[3] == "whitebox"]
            
            if not attacks_to_run:
                print("No attacks to run. Exiting.")
                sys.exit(1)
    
    print(f"\nAttacks to run: {len(attacks_to_run)}")
    
    # Run attacks
    results = []
    overall_start = time.time()
    
    for attack_num, attack_name, module_name, attack_type in attacks_to_run:
        result = run_attack(attack_num, attack_name, module_name)
        results.append(result)
        
        # Add a small delay between attacks
        if attack_num < len(attacks_to_run):
            time.sleep(2)
    
    overall_duration = time.time() - overall_start
    
    # Print summary
    print_summary(results, overall_duration)
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nAll evidence saved to:")
    print("  Images: evidence/adversarial/")
    print("  Logs:   evidence/logs/")
    
    # Exit with appropriate code
    if any(r["status"] == "failed" for r in results):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
