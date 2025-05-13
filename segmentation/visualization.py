import os
from PIL import Image
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def get_image_mask_pairs(image_dir, mask_dir):
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg'))])
    return [
        {
            'image_path': os.path.join(image_dir, img),
            'mask_path': os.path.join(mask_dir, msk)
        }
        for img, msk in zip(image_files, mask_files)
    ]


def visualize_predictions(image_dir, mask_dir, num_to_show=5, alpha=0.5):
    pairs = get_image_mask_pairs(image_dir, mask_dir)

    # Binary color map: 0=background (black), 1=plant (green)
    colors = np.array([
        [0, 0, 0],    # background
        [0, 255, 0],  # plant
    ], dtype=np.uint8)
    class_names = ['background', 'plant']

    # Normalize for legend patches
    colors_norm = colors / 255.0
    dpi = 100

    # make output dir if it doesn't exist
    os.makedirs('output_dir', exist_ok=True)

    # for pair in pairs[:num_to_show]:
    for pair in pairs:
        img = Image.open(pair['image_path']).convert('RGB')
        mask = Image.open(pair['mask_path']).convert('L')
        mask_np = np.array(mask)

        # Collapse all non-zero labels into 1
        mask_bin = (mask_np > 0).astype(np.uint8)

        # Apply colors
        color_mask = colors[mask_bin]
        overlay = Image.blend(
            img.convert('RGBA'),
            Image.fromarray(color_mask).convert('RGBA'),
            alpha
        )

        # Figure at true pixel size
        w_px, h_px = img.size
        figsize = (w_px / dpi, h_px / dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.imshow(overlay)
        ax.axis('off')

        # Legend
        handles = [
            Patch(facecolor=colors_norm[i], edgecolor='k', label=class_names[i])
            for i in range(2)
        ]
        ax.legend(handles=handles,
                  loc='upper right',
                  bbox_to_anchor=(0.98, 0.98),
                  framealpha=0.7,
                  fontsize='small')

        # Save or show the figure
        fig.savefig(os.path.join('output_dir', f'overlay_{os.path.basename(pair["image_path"])}'))
        plt.close(fig)  # Close the figure to free memory
    # plt.show()


if __name__ == "__main__":
    with open("../config.yml", "r") as f:
        config = yaml.safe_load(f)
    dataset_root = os.path.expanduser(config["segmentation_dataset"])

    image_dir = os.path.join(dataset_root, 'train_sim')
    mask_dir = os.path.join(dataset_root, 'inference_train_sim')
    image_dir = "/Users/arthur/Documents/ComputerVision/assignments/DT-MARS-CycleGAN/dataset/segmentation/test_sim"
    mask_dir = "/Users/arthur/Downloads/inference_test_sim_styled"
    visualize_predictions(image_dir, mask_dir, num_to_show=20, alpha=0.6)
