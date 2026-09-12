# flower-classifier

Simple Image Recognition Tool for flow image classifier

## Setup environment

Requires Python < 3.14

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

```powershell
winget install Graphviz.Graphviz
```

```python
import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"
```

## Launching

```bash
jupyter-lab
```

## Deploy with TFX and Docker

```bash
MSYS_NO_PATHCONV=1 docker run -d --name tf_serving_flowers \
  -p 8501:8501 \
  -v "$(pwd)/models/flower_classifier:/models/flower_classifier" \
  -e MODEL_NAME=flower_classifier \
  tensorflow/serving
```

## Testing

```bash

```
