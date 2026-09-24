# Training the Model

The weights must exist before building the Docker image. If `target/weights/` is empty
or missing, follow the steps below.

---

## Prerequisites

Python 3.12 with the virtual environment set up:

**Linux / macOS**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Run training

From the task-01 root:

```bash
python -m target.training.train
```

This downloads CIFAR-10 (~170 MB on first run), trains for 20 epochs, and writes:

| File | Description |
|------|-------------|
| `target/weights/cifar_cnn.pth` | Best checkpoint (highest val accuracy) |
| `target/weights/model_info.json` | Architecture metadata used by the inference server |

At the end of training the script prints the SHA-256 hashes needed to run the container:

```
MODEL_SHA256=<hash>
MODEL_INFO_SHA256=<hash>
```

Copy these — they are required when running the Docker container.

---

## After training

Then rebuild the image and run it with the hashes printed above:

```bash
docker build -t cifar10-target ./target
```

```bash
docker run -p 8000:8000 -e MODEL_SHA256=<hash from training> -e MODEL_INFO_SHA256=<hash from training> cifar10-target
```
