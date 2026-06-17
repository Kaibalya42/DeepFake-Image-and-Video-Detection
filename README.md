# DeepFake-Image-and-Video-Detection

# DeepFake Image and Video Detection

A deep learning pipeline that detects deepfake faces in **images and videos**. Faces are extracted with MTCNN and classified using two trained models — a pretrained **EfficientNet-B0 CNN** and a **Hybrid CNN + Transformer** network — trained on a combined dataset built from CelebDF-V2 and FaceForensics++ (C23).

See [Model & Results](#model--results) below for architecture details and benchmark numbers.

## Repository layout

| Path | What it is | Where it runs | Stack |
|---|---|---|---|
| `Project.ipynb` | Full pipeline: preprocessing, training, evaluation, visualization | Kaggle / Google Colab (GPU recommended) | Python 3.9+ · PyTorch · timm · facenet-pytorch · OpenCV |
| `checkpoints/` | Saved best model weights (`.pth`) | Generated after training | — |
| `graphs/` | Loss/accuracy curves, confusion matrices, ROC & PR curves | Generated after training | — |

## Quick start (10 minutes)

### 1. Prerequisites

- Python 3.9+
- pip
- A GPU is strongly recommended (the notebook was developed on a Kaggle T4); CPU works but training will be slow
- A Kaggle account (to attach the datasets) or the datasets downloaded manually — see [Dataset](#dataset)

### 2. Clone & install

```bash
git clone https://github.com/<your-username>/DeepFake-Image-and-Video-Detection.git
cd DeepFake-Image-and-Video-Detection

pip install torch torchvision timm facenet-pytorch opencv-python numpy pandas scikit-learn matplotlib seaborn tqdm pillow jupyter
```

### 3. Get the dataset

The datasets are too large to include in this repo. Download them from Kaggle and point the notebook's config cell at the extracted folders.

| Dataset | Description | Link |
|---|---|---|
| Celeb-DF V2 | Real and deepfake celebrity videos/face images | https://www.kaggle.com/datasets/reubensuju/celeb-df-v2 |
| FaceForensics++ (C23) | Real videos + 6 manipulation methods | https://www.kaggle.com/datasets/xdxd003/ff-c23 |

In `Project.ipynb`, update these paths to wherever you extracted the datasets:

```python
CELEB_INPUT = "/path/to/Celeb_V2"
FF_INPUT    = "/path/to/FaceForensics++_C23"
```

### 4. Run the pipeline

Open `Project.ipynb` in Jupyter, Kaggle, or Colab and run the cells top to bottom:

1. **Preprocessing** — MTCNN face extraction from CelebDF images + sampled FaceForensics++ video frames, merged into `combined_data/train|val|test`
2. **Data loaders** — builds `train_loader` / `val_loader` / `test_loader` with augmentation
3. **Model definitions** — initializes `EfficientNetModel` and `HybridCNNTransformer`
4. **Training** — trains both models (20 epochs, mixed precision, cosine LR schedule); best weights saved to `checkpoints/`
5. **Evaluation & visualization** — computes accuracy/precision/recall/F1 and saves plots to `graphs/`

### 5. End-to-end smoke test

1. Run the preprocessing cell → confirms `combined_data/train|val|test` folders are populated with face crops
2. Run the training cell for a couple of epochs → confirms `checkpoints/*.pth` gets saved
3. Run the evaluation cell → confirms metrics print to console and `graphs/` fills with plots
4. Run the test-set evaluation cell → confirms final accuracy/ROC/PR plots are generated

If all four steps work, the pipeline is wired correctly end-to-end.

## Useful commands

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Convert the notebook to a plain Python script (optional, for CLI runs)
jupyter nbconvert --to script Project.ipynb

# Launch Jupyter locally
jupyter notebook
```

## Model & Results

**EfficientNet_B0 (CNN baseline)** — pretrained `efficientnet_b0` backbone (`timm`) + custom head: `Linear(1280→512) → ReLU → Dropout(0.4) → Linear(512→2)`.

**CNN_Transformer (Hybrid model)** — pretrained `efficientnet_b3` backbone → flattened patch embeddings → learnable positional embeddings → 2-layer Transformer encoder (4 heads, Pre-LN, dropout 0.3) → pooled classification head.

| Model | Test Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| EfficientNet_B0 | 95.72% | 0.9530 | 0.9615 | 0.9573 |
| CNN_Transformer (Hybrid) | 86.28% | 0.8528 | 0.8760 | 0.8642 |

Key training settings: image size 224×224, batch size 64, 20 epochs (3 warmup), AdamW with per-group learning rates, cosine LR annealing, class-weighted cross-entropy with label smoothing, mixed-precision (AMP) training.

## Where it runs

| Component | Platform | Notes |
|---|---|---|
| Training & evaluation pipeline | Kaggle Notebooks (GPU T4) or Google Colab | Attach/upload the datasets, set the input paths, run top to bottom |
| Saved checkpoints (`.pth`) & graphs | Local / Kaggle `working` directory | Not committed to the repo — regenerate by re-running the notebook |

## Licence

This repository does not yet include a license file. Add a `LICENSE` (e.g. MIT) if you plan to open-source the project.
