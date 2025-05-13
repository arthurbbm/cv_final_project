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
   git clone <your‑repo‑url> cv_final_project
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
pybullet_dataset: /absolute/path/to/data/input
```

* **`pybullet_dataset`**
  Root of your data directory.

  * Input soil images in `[pybullet_dataset]/input/soil/`
  * Input plant meshes in `[pybullet_dataset]/input/plant/`
  * After running PyBullet scripts, outputs will be written to:

    ```
    [pybullet_dataset]/output/images  
    [pybullet_dataset]/output/masks
    ```
  * **Important:** after each run of `generate_train_dataset.py` or `generate_test_dataset.py`, **rename** the `output/` folder (e.g. `output_train/`, `output_test/`) to prevent overwriting/mixture.

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
  --num_labels 5 \
  [--device cuda]
```

* `--num_labels`: including background
* `--checkpoint_freq`: save model every *n* epochs
* `--device`: `cuda` or `cpu`

---

## 🔍 Running Inference

Once you have a trained model (`.pth`), you can predict on new images:

```bash
cd segmentation
python inference.py \
  --model_path /path/to/segmentation_model.pth \
  --num_labels 5 \
  --test_dir /path/to/test/images \
  --output_dir /path/to/save/predicted/masks \
  [--device cuda]
```

Predicted masks will be saved as PNGs in `output_dir`.

---

## 📝 Metrics & Visualization

* **metrics.py** — compute accuracy, IoU, precision, recall, etc.
* **visualization.py** — overlay masks on RGB for qualitative inspection.

---

## 💬 Support & Contributions

Feel free to open issues or pull requests for bug fixes, enhancements, or documentation improvements.

---

**Happy segmenting!**
