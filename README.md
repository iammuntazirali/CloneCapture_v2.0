# Clone Catcher

Clone Catcher is an experimental, full-stack image-classification project that estimates whether a
face image belongs to the **real** or **synthetic/fake** class. It combines a React web interface,
a FastAPI inference service, and PyTorch models and experiments.

> **Important:** This is a research/demo project, not a forensic tool. A prediction is a model
> estimate, not proof that an image is authentic or manipulated. The dataset primarily compares
> real photographs with GAN-generated synthetic faces; the project has not established reliable
> detection of face swaps, every deepfake technique, or images from the real world.

## At a glance

| Item | Details |
|---|---|
| Project name | Clone Catcher |
| Project type | Full-stack machine-learning prototype |
| Main task | Binary classification of face images as `REAL` or `FAKE` |
| Web client | React 18, JavaScript, Vite |
| API | Python, FastAPI, Uvicorn |
| Machine learning | PyTorch, Torchvision, `timm`, Pillow |
| Models explored | Four-block custom CNN and Xception |
| Model used by the web API | Trained Xception checkpoint |
| Input supported by the web app/API | One JPG/JPEG or PNG image, up to 10 MB |
| Output | Predicted class, real score, fake score, and predicted-class model score |

## What the application does

1. A user selects or drops a JPG/JPEG/PNG image into the web page.
2. The browser shows a preview and sends the selected file as multipart form data to the API.
3. FastAPI checks the upload size and verifies that the file is a supported, readable image.
4. Pillow converts it to RGB. Torchvision resizes it to `224 × 224`, converts it to a tensor, and
   applies ImageNet channel normalization.
5. The loaded Xception model runs inference. Softmax converts its two output logits into `REAL` and
   `FAKE` scores.
6. The API returns the prediction and scores. The web page displays the result and lets the user
   analyze another image.

The current inference path classifies the **whole uploaded image**. It does not detect, crop, or
track a face; analyze video; verify identity; explain which pixels influenced the result; or compare
an image with an original source.

## Repository layout

```text
CloneCapture_v2.0/
├── .gitignore                   # Excludes local datasets, checkpoints, and build artifacts
├── backend/
│   ├── main.py                   # FastAPI app, image validation, preprocessing, inference
│   ├── requirements.txt          # API runtime dependencies
│   └── test-requirements.txt     # API test dependencies
├── frontend/
│   ├── index.html                # Web app entry HTML
│   ├── package.json              # React/Vite scripts and dependencies
│   ├── package-lock.json         # Locked npm dependency graph
│   ├── vite.config.js            # Vite React plugin configuration
│   └── src/
│       ├── App.jsx               # Upload flow, API request, results and error states
│       ├── App.css               # Responsive page styling
│       └── main.jsx              # React application entry point
├── ml/
│   ├── models/
│   │   ├── custom_cnn.py         # CNN trained from random initialization
│   │   └── xception_model.py     # Xception wrapper around timm
│   ├── preprocessing/
│   │   ├── dataset.py            # Dataset loading, labels, transforms
│   │   └── test_dataset.py       # Dataset and tensor-shape checks
│   ├── training/
│   │   ├── train.py              # Custom CNN training
│   │   └── train_xception.py     # Xception fine-tuning
│   ├── evaluation/
│   │   ├── evaluate.py           # Custom CNN test-set evaluation
│   │   ├── evaluate_xception.py  # Xception test-set evaluation
│   │   └── compare_models.py     # Side-by-side model evaluation
│   └── inference/
│       └── predict_image.py      # Command-line Xception prediction
├── tests/
│   └── test_backend.py           # API health and image-upload tests
├── docs/
│   ├── custom_cnn_results.md     # CNN experiment results and limitations
│   ├── xception_results.md       # Xception experiment results and limitations
│   ├── dataset_integrity_findings.md
│   ├── model_comparison.txt
│   ├── near_duplicate_audit.csv
│   ├── pixel_similarity_audit.csv
│   ├── near_duplicate_review.png # Contact sheet of selected candidate pairs
│   └── train_test_contact_sheet.png # Train/test candidate-pair contact sheet
├── check_gpu.py                 # PyTorch CUDA availability check
├── requirements.txt             # Core PyTorch/Torchvision and image dependencies
└── README.md
```

### Files intentionally not included

The repository's `.gitignore` excludes `*.pth` model checkpoints, dataset directories, `node_modules`,
and virtual environments. In the checked-out project, there is no `checkpoints/xception_best.pth`
file and no dataset directory. You must provide the relevant assets locally before running inference,
the backend tests, training, or evaluation. The API expects the trained checkpoint at:

```text
checkpoints/xception_best.pth
```

Training and evaluation scripts expect the processed dataset at:

```text
ml/datasets/processed/real-vs-fake/Real vs Fake Faces/8020/
├── train/
│   ├── real/
│   └── fake/
├── valid/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

The `real` and `fake` class directories should contain JPG, JPEG, or PNG files. The dataset is not
downloaded automatically by this project. The training scripts create `checkpoints/` when saving
their outputs. The Xception training script saves `xception_best.pth`; the API needs that file to
start successfully.

## How the main components work

### Frontend

`frontend/src/App.jsx` implements the single-page upload and results flow:

- Accepts JPG/JPEG and PNG image MIME types, using a file picker or drag and drop.
- Displays an image preview, filename, approximate size, and a replace/remove action.
- Sends the selected file to `POST /predict` using `FormData`.
- Displays the predicted class, its model score, and both class scores.
- Shows request errors and an analysis-in-progress state.
- Reads the API base URL from `VITE_API_URL`; if it is not set, it uses
  `http://localhost:8000`.

The browser's MIME-type check is a convenience for the user; the backend performs its own content
and format checks. The frontend is a development/demo interface, not a production upload service.

### Backend and API

`backend/main.py` creates the FastAPI application and loads the checkpoint once during application
startup. It selects CUDA when PyTorch reports that CUDA is available; otherwise it runs on CPU.
The model is set to evaluation mode and each prediction runs without gradient calculation.

The API uses the `XceptionModel` wrapper from `ml/models/xception_model.py`, configured for two
classes with pretrained initialization disabled when restoring the trained checkpoint. The class
ordering is significant: index `0` is `REAL`, and index `1` is `FAKE`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Reports whether the model loaded and which device (`cpu` or `cuda`) is in use. |
| `POST /predict` | Accepts a multipart upload with the field name `file` and returns a prediction. |

Example successful prediction response:

```json
{
  "prediction": "FAKE",
  "real_score": 1.24,
  "fake_score": 98.76,
  "model_score": 98.76
}
```

The example numbers are illustrative only. A successful response contains:

- `prediction`: `REAL` or `FAKE`, chosen by the larger class probability.
- `real_score`: the `REAL` softmax output converted to a percentage.
- `fake_score`: the `FAKE` softmax output converted to a percentage.
- `model_score`: the score belonging to the predicted class.

These are softmax scores, not calibrated probabilities or certainty guarantees. The backend accepts
only JPEG and PNG image content and limits uploads to 10 MiB. It reports unsupported formats with
HTTP `415`, corrupted/unreadable or empty uploads with `400`, uploads over the size limit with
`413`, and an unavailable/loading model with `503`. The API's CORS allowlist is limited to the Vite
development origins `http://localhost:5173` and `http://127.0.0.1:5173`.

### Machine-learning models

**Custom CNN baseline:** `ml/models/custom_cnn.py` defines four convolutional blocks. Each block
contains a 3×3 convolution, batch normalization, ReLU, and 2×2 max-pooling. The channels progress
from 3 to 32, 64, 128, and 256. Adaptive average pooling, flattening, dropout, and a linear
two-class layer produce the output logits. It is trained from random initialization.

**Xception model:** `ml/models/xception_model.py` wraps the `xception` architecture provided by
`timm`. Training starts from ImageNet-pretrained weights and replaces the classifier with a
two-class output layer. The saved, fine-tuned Xception checkpoint is the model used by both the
web API and the command-line inference script.

Both training and inference use 224 × 224 RGB images normalized channel-wise with ImageNet means
`[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`. The dataset loader maps
`real` to label `0` and `fake` to label `1`.

## Recorded experiment results

The following are the results recorded in the experiment reports for the 1,000-image test split.
The split is described as balanced, with 500 real and 500 fake images. They are reported here as
**dataset-specific experimental results**, not expected production performance.

| Metric | Custom CNN | Xception |
|---|---:|---:|
| Test loss | 0.3640 | 0.0329 |
| Accuracy | 83.40% | 99.00% |
| Precision (fake class) | 87.95% | 98.80% |
| Recall (fake class) | 77.40% | 99.20% |
| F1 score (fake class) | 82.34% | 99.00% |
| ROC-AUC | 0.9201 | 0.9996 |

Confusion matrices use **actual labels as rows** and **predictions as columns**, ordered Real then
Fake:

| Model | Actual Real → Predicted Real | Actual Real → Predicted Fake | Actual Fake → Predicted Real | Actual Fake → Predicted Fake |
|---|---:|---:|---:|---:|
| Custom CNN | 447 | 53 | 113 | 387 |
| Xception | 494 | 6 | 4 | 496 |

Accuracy is the share of all test images classified correctly. Fake-class precision is the share
of images predicted fake that are actually fake; fake-class recall is the share of actual fake
images detected as fake. F1 is the harmonic mean of that precision and recall. ROC-AUC measures
ranking discrimination across thresholds; it does not establish reliability on a different
dataset.

### Experiment configuration

| Setting | Custom CNN | Xception |
|---|---|---|
| Input | RGB, 224 × 224 | RGB, 224 × 224 |
| Optimizer | Adam | Adam |
| Learning rate | 0.001 | 0.0001 |
| Batch size | 16 | 8 |
| Maximum epochs | 20 | 15 |
| Random seed | 42 | 42 |
| Initialization | Random | ImageNet-pretrained Xception |
| Best checkpoint selection | Highest validation F1 | Highest validation F1 |
| Reported best epoch | 19 | 11 |
| Mixed precision | No | FP16 on CUDA |

For the reported experiment, the dataset configuration contains 8,000 training images, 1,000
validation images, and 1,000 test images. The test set is evaluated by the evaluation scripts;
training selects the best checkpoint using validation F1. See
[`docs/custom_cnn_results.md`](docs/custom_cnn_results.md),
[`docs/xception_results.md`](docs/xception_results.md), and
[`docs/model_comparison.txt`](docs/model_comparison.txt) for the recorded reports.

## Dataset and research limitations

The documented source is the **Real vs Fake Faces Balanced Dataset with Multiple Dataset Splits**,
version 1.0, DOI [10.5281/zenodo.14532968](https://doi.org/10.5281/zenodo.14532968), a restructured
version of RVF10K. The audit report says the real images originate from NVIDIA's FFHQ and the fake
images from a 1 Million Fake Faces dataset. The project has not verified the exact generator/version
or source-level provenance for every image.

The integrity audit reports:

- No exact SHA-256 duplicate groups among the 10,000 audited images.
- 33 train/test candidate pairs from perceptual-hash screening at Hamming distance 6. These are
  candidates, not confirmed duplicates; visual inspection found both similar-looking pairs and
  false positives.
- Identity-level and source-photo/source-video-level independence across splits remains unverified.
- The dataset focuses on real photographs versus GAN-generated synthetic faces. It does not by
  itself demonstrate performance on face swaps or other manipulation families.

Possible cross-split visual similarity may inflate held-out scores. The evaluation is from one
dataset and one reported split; cross-dataset generalization, unseen identities, and unseen
generators/manipulation methods have not been established. Treat the results as preliminary until
the source-level split and independent evaluation are addressed. The detailed findings are in
[`docs/dataset_integrity_findings.md`](docs/dataset_integrity_findings.md).

## Setup and run locally

### Prerequisites

- Python 3.10 or later is recommended.
- Node.js and npm.
- A compatible PyTorch/Torchvision installation. The root `requirements.txt` pins a CUDA 12.6
  build; for a different CUDA version or a CPU-only environment, install a compatible PyTorch and
  Torchvision pair using the official PyTorch installation selector before installing the remaining
  project packages.
- A local `checkpoints/xception_best.pth` model checkpoint (not included in this repository).

Run commands from the project root unless a command explicitly changes directory. On Windows,
PowerShell examples are provided.

### 1. Install Python dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r backend\requirements.txt
python -m pip install timm scikit-learn
```

The root manifest supplies core PyTorch/Torchvision and image dependencies; the backend manifest
supplies FastAPI, multipart upload support, and Uvicorn. `timm` is needed for Xception, and
`scikit-learn` is needed by training/evaluation scripts. If the pinned PyTorch build does not match
your hardware or Python environment, install a compatible build first and adjust the root
requirements installation for that environment.

### 2. Add the trained checkpoint

Place the trained Xception checkpoint at:

```text
checkpoints/xception_best.pth
```

The application intentionally fails during startup if this file is absent or incompatible. It does
not silently substitute randomly initialized weights.

### 3. Start the API

```powershell
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Check `http://127.0.0.1:8000/health`. A ready response should contain
`"status": "healthy"` and `"model_loaded": true`.

### 4. Start the web interface

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal (normally `http://localhost:5173`). To point the frontend
to a different API URL, set `VITE_API_URL` in the frontend environment before starting Vite.

## Training, evaluation, and command-line inference

These commands require the processed dataset layout described above. Training and evaluation also
require the Python machine-learning dependencies and can be compute-intensive.

Train the custom CNN:

```powershell
python -m ml.training.train
```

Train/fine-tune Xception (may download its pretrained initialization if it is not already cached):

```powershell
python -m ml.training.train_xception
```

Evaluate the saved custom CNN or Xception checkpoint on the test split:

```powershell
python -m ml.evaluation.evaluate
python -m ml.evaluation.evaluate_xception
```

Compare both checkpoints and write the comparison to `docs/model_comparison.txt`:

```powershell
python -m ml.evaluation.compare_models
```

Run Xception inference on a single image from the project root:

```powershell
python -m ml.inference.predict_image --image "C:\path\to\image.jpg"
```

The standalone dataset diagnostic can be run with:

```powershell
python ml\preprocessing\test_dataset.py
```

## Tests and checks

Install the test requirements and run the backend tests from the project root:

```powershell
python -m pip install -r backend\test-requirements.txt
python -m pytest tests\test_backend.py
```

The backend tests exercise health reporting, valid JPEG/PNG predictions, unsupported formats, and
corrupted image handling. They start the application and therefore also require a compatible
`checkpoints/xception_best.pth` file.

Build the frontend production bundle:

```powershell
cd frontend
npm run build
```

Check PyTorch and CUDA availability:

```powershell
cd ..
python check_gpu.py
```

## Project status and scope

The repository contains the application source, model definitions, training/evaluation scripts,
experiment reports, and dataset-integrity audit materials. It does not contain the large dataset
or model checkpoint, so it is not a self-contained one-command demo. Video analysis appears in
early project objectives but is **not implemented** in the current API or frontend. The currently
implemented application analyzes still images only.
