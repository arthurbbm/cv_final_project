# cv\_final\_project

This repository implements a simple pipeline for plant–soil segmentation using simulated datasets generated in PyBullet and a deep‐learning segmentation model. It also supports applying a style‐transfer step (via an external repository) to improve domain adaptation.

---

## 📂 Directory Structure

```
.
├── config.yml
├── environment.yml
├── requirements.txt
├── pybullet
│   ├── generate_train_dataset.py
│   └── generate_test_dataset.py
└── segmentation
    ├── train.py
    ├── inference.py
    ├── metrics.py
    └── visualization.py
```

You will also add your raw inputs here:

```
data/
└── input/
    ├── soil/      # .jpg images of soil textures
    └── plant/     # .stl mesh files of plant models
```

---

## 🔧 Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/arthurbbm/cv_final_project
   cd cv_final_project
   ```

2. **Create the conda environment**

   ```bash
   conda env create -f environment.yml
   conda activate cv_final_project
   ```

3. (Optional) You can also install via `pip` if you prefer:

   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Configuration

All configurable paths go into `config.yml`.  At minimum, set:

```yaml
pybullet_dataset: /absolute/path/to/data
segmentation_dataset: /path/to/segmentation_dataset
```

* **`pybullet_dataset`**
  Root of your data directory for PyBullet jobs.

  * Input soil images in `…/data/input/soil/`
  * Input plant meshes in `…/data/input/plant/`
  * After running PyBullet scripts, outputs will be written to:

    ```
    [pybullet_dataset]/output/images  
    [pybullet_dataset]/output/masks
    ```
  * **Important:** after each run of `generate_train_dataset.py` or `generate_test_dataset.py`, **rename** the `output/` folder (e.g. `output_train/`, `output_test/`) to prevent overwriting/mixture.

* **`segmentation_dataset`**
  Root of your segmentation dataset.  It should contain two subdirectories:

  ```
  /path/to/segmentation_dataset/
  ├── train_sim/           # ground-truth images/masks
  └── inference_train_sim/ # predicted masks
  ```

---

## 🤖 Dataset Generation with PyBullet

**What is PyBullet?**
PyBullet is a physics‐simulation library that can simulate rigid‐body dynamics and sensor rendering. Here it’s used to spawn synthetic scenes of soil + plant meshes and render:

* **RGB images** → saved to `…/output/images`
* **Ground‐truth masks** → saved to `…/output/masks`

### Usage

```bash
# Generate training dataset
python pybullet/generate_train_dataset.py

# Generate testing dataset
python pybullet/generate_test_dataset.py
```

After each run, rename the `…/output/` directory before running the other script.

---

## 🎨 (Optional) Style‑Transfer

This project does **not** implement a new style‑transfer method, but you can apply one to make your simulated images look more “real.” We recommend the DT‑MARS‑CycleGAN codebase:

> [https://github.com/UGA-BSAIL/DT-MARS-CycleGAN](https://github.com/UGA-BSAIL/DT-MARS-CycleGAN)

You can use their pretrained models or train your own, then apply to the images in your PyBullet `output/images` before training the segmentation network.

---

## 🏋️‍♀️ Training the Segmentation Model

The segmentation scripts live under `segmentation/`.  First, make sure your combined dataset (possibly style‑transferred) is arranged as:

```
/path/to/dataset/
├── images/
│   ├── img_0001.png
│   └── …
└── masks/
    ├── mask_0001.png
    └── …
```

### Command

```bash
cd segmentation
python train.py \
  --n_epochs 10 \
  --batchSize 4 \
  --dataroot /path/to/dataset \
  --outdir /path/to/save/model \
  --checkpoint_freq 5 \
  --lr 1e-4 \
  --num_labels 2 \
  [--device cuda]
```

* **Train script**: `train.py` is the correct entry point.
* **File naming requirement**: The script pairs images and masks by sorted order—make sure each RGB image and its corresponding mask share the same filename (aside from extension) so that after sorting they align correctly.
* **Output files** (in `--outdir`):

  * `checkpoint.pth` (every `checkpoint_freq` epochs)
  * `segmentation_model.pth` (final `state_dict`)

---

## 🔍 Running Inference

Once you have a trained model (`.pth`), you can predict on new images:

```bash
cd segmentation
python inference.py \
  --model_path /path/to/segmentation_model.pth \
  --num_labels 2 \
  --test_dir /path/to/test/images \
  --output_dir /path/to/save/predicted/masks \
  [--device cuda]
```

Predicted masks will be saved as PNGs in `output_dir`.

---

## 📝 Metrics & Visualization

This project includes scripts for evaluating and visualizing model predictions.

### 1. Metrics (`metrics.py`)

* **Purpose:** Compute image-wise and overall segmentation metrics (accuracy, precision, recall, F1-score, IoU).
* **Configuration:** Ensure `config.yml` contains:

  ```yaml
  segmentation_dataset: /path/to/segmentation_dataset
  ```

  with structure:

  ```
  segmentation_dataset/
  ├── train_sim/             # ground-truth images and masks
  └── inference_train_sim/   # predicted masks (same filenames)
  ```
* **Usage:**

  ```bash
  cd segmentation
  python metrics.py
  ```
* **Outputs:**

  * `metrics_imagewise.csv` – per-image metrics table
  * `metrics_overall.csv`   – aggregated metrics summary

### 2. Visualization (`visualization.py`)

* **Purpose:** Overlay predicted masks on original RGB images and save blended figures for qualitative inspection.
* **Configuration:** Uses the same `segmentation_dataset` path from above.
* **Usage:**

  ```bash
  cd segmentation
  python visualization.py
  ```
