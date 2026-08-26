from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plantcv.plantcv as pcv

from src.data import IMAGE_EXTENSIONS, LeafLoader


class LeafTransformer:
    def __init__(self, loader=None):
        pcv.params.debug = None
        self.loader = loader or LeafLoader()

    def get_mask(self, rgb_img):
        """Return binary mask isolating the leaf."""
        rgb_img = rgb_img.astype(np.uint8) if rgb_img.dtype != np.uint8 else rgb_img
        blurred = pcv.gaussian_blur(img=rgb_img, ksize=(5, 5), sigma_x=0)
        s = pcv.rgb2gray_hsv(rgb_img=blurred, channel="s")
        mask = pcv.fill(pcv.threshold.binary(s, threshold=40, object_type="light"), 200)
        return pcv.fill_holes(bin_img=mask)

    def transform(self, rgb):
        """Run all 6 PlantCV transformations. Returns result dict."""
        blurred = pcv.gaussian_blur(img=rgb, ksize=(5, 5), sigma_x=0)
        mask = self.get_mask(rgb)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        leaf_contour = max(contours, key=cv2.contourArea)
        roi_vis = rgb.copy()
        cv2.drawContours(roi_vis, [leaf_contour], -1, (0, 255, 0), 2)

        lbl, n_lbl = pcv.create_labels(mask=mask)
        pcv.outputs.clear()
        pcv.analyze.size(img=rgb, labeled_mask=lbl, n_labels=n_lbl)
        obs = pcv.outputs.observations["default_1"]
        shape = {
            k: obs[k]["value"]
            for k in [
                "area", "perimeter", "solidity", "convex_hull_area",
                "longest_path", "ellipse_eccentricity", "width", "height",
            ]
        }
        shape["circularity"] = 4 * np.pi * shape["area"] / (shape["perimeter"] ** 2 + 1e-6)
        shape["extent"] = shape["area"] / (shape["width"] * shape["height"])

        plm = pcv.homology.x_axis_pseudolandmarks(img=rgb, mask=mask, label="leaf")
        plm_vis = rgb.copy()
        for group in plm:
            for pt in group:
                cv2.circle(plm_vis, (int(pt[0][0]), int(pt[0][1])), 3, (255, 0, 0), -1)

        pcv.outputs.clear()
        pcv.analyze.color(rgb_img=rgb, labeled_mask=lbl, n_labels=n_lbl, colorspaces="hsv")
        obs = pcv.outputs.observations["default_1"]
        hue_h = np.array(obs["hue_frequencies"]["value"])
        sat_h = np.array(obs["saturation_frequencies"]["value"])

        masked = pcv.apply_mask(img=rgb, mask=mask, mask_color="white")

        return {
            "rgb": rgb,
            "blurred": blurred,
            "mask": mask,
            "roi_vis": roi_vis,
            "plm_vis": plm_vis,
            "masked": masked,
            "hue_h": hue_h,
            "sat_h": sat_h,
            "shape": shape,
        }

    def display_results(self, results):
        """Show 2x4 panel of all transformation results."""
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        panels = [
            (axes[0, 0], results["rgb"], "Original"),
            (axes[0, 1], results["blurred"], "T1: Blur"),
            (axes[0, 2], results["mask"], "T2: Mask"),
            (axes[0, 3], results["roi_vis"], "T3: ROI Objects"),
            (axes[1, 0], results["masked"], "T3: Masked Leaf"),
            (axes[1, 1], results["plm_vis"], "T5: Pseudolandmarks"),
        ]
        for ax, img, title in panels:
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")
        axes[1, 2].bar(range(len(results["hue_h"])), results["hue_h"], color="salmon")
        axes[1, 2].set_title("T6a: Hue")
        axes[1, 3].bar(range(len(results["sat_h"])), results["sat_h"], color="dodgerblue")
        axes[1, 3].set_title("T6b: Saturation")
        plt.suptitle("Part 3 — Image Transformations", fontsize=15)
        plt.tight_layout()
        plt.show()

        print("\nShape metrics:")
        for k, v in results["shape"].items():
            print(f"  {k:20s}  {v:.4f}")

    def save_transformations(self, img_path, dst_dir):
        """Apply all transformations to one image and save results."""
        rgb = self.loader.load_rgb(img_path)
        stem = Path(img_path).stem
        ext = Path(img_path).suffix
        results = self.transform(rgb)

        files = {
            "Gaussian_Blur": results["blurred"],
            "Mask": results["mask"],
            "ROI_Objects": cv2.cvtColor(results["roi_vis"], cv2.COLOR_RGB2BGR) if len(results["roi_vis"].shape) == 3 else results["roi_vis"],
            "Analyze_Object": cv2.cvtColor(results["masked"], cv2.COLOR_RGB2BGR) if len(results["masked"].shape) == 3 else results["masked"],
            "Pseudolandmarks": cv2.cvtColor(results["plm_vis"], cv2.COLOR_RGB2BGR) if len(results["plm_vis"].shape) == 3 else results["plm_vis"],
        }
        for name, img_arr in files.items():
            cv2.imwrite(str(dst_dir / f"{stem}_{name}{ext}"), img_arr)

    def process_directory(self, src_dir, dst_dir):
        """Batch-process all images in a directory."""
        src_dir = Path(src_dir)
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        images = [f for f in src_dir.iterdir() if f.is_file() and f.suffix in IMAGE_EXTENSIONS]
        print(f"Processing {len(images)} images from {src_dir} -> {dst_dir}")
        for img_path in sorted(images):
            try:
                self.save_transformations(img_path, dst_dir)
                print(f"  [OK] {img_path.name}")
            except Exception as e:
                print(f"  [FAIL] {img_path.name}: {e}")
        print(f"Done. Transformed images saved to {dst_dir}")
