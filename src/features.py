import io
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plantcv.plantcv as pcv
from PIL import Image

from src.data import LeafLoader


class LeafTransformer:
    # PlantCV v4's analyze.color() no longer draws this classic overlay style,
    # so we rebuild it ourselves from pcv.outputs. Order/colors chosen to match
    # the reference "color Channel" legend (blue, blue-yellow, green,
    # green-magenta, hue, lightness, red, saturation, value).
    _HIST_CHANNELS = {
        "blue": ("blue_frequencies", "blue"),
        "blue-yellow": ("blue-yellow_frequencies", "gold"),
        "green": ("green_frequencies", "green"),
        "green-magenta": ("green-magenta_frequencies", "magenta"),
        "hue": ("hue_frequencies", "purple"),
        "lightness": ("lightness_frequencies", "gray"),
        "red": ("red_frequencies", "red"),
        "saturation": ("saturation_frequencies", "cyan"),
        "value": ("value_frequencies", "orange"),
    }

    def __init__(
        self,
        loader=None,
        *,
        # -- leaf segmentation tuning (used by _segment_leaf) --
        s_threshold=40,
        blur_ksize=(5, 5),
        blur_sigma=0,
        rethreshold=127,
        fill_size=200,
        # -- lesion segmentation tuning (used by _segment_lesion) --
        lesion_a_threshold=None,  # None -> Otsu auto-threshold on LAB 'a' channel
        lesion_fill_size=15,
        lesion_close_ksize=3,
        # -- T3 ROI outline --
        roi_color=(0, 255, 0),
        roi_thickness=2,
        lesion_color=(0, 0, 255),
        # -- T5 pseudolandmarks --
        landmark_radius=3,
        landmark_colors=((0, 0, 255), (255, 0, 255), (255, 128, 0)),  # top, bottom, center_v
        # -- T6 color histogram --
        hist_figsize=(10, 6),
        hist_dpi=150,
    ):
        pcv.params.debug = None
        self.loader = loader or LeafLoader()

        self.s_threshold = s_threshold
        self.blur_ksize = blur_ksize
        self.blur_sigma = blur_sigma
        self.rethreshold = rethreshold
        self.fill_size = fill_size

        self.lesion_a_threshold = lesion_a_threshold
        self.lesion_fill_size = lesion_fill_size
        self.lesion_close_ksize = lesion_close_ksize

        self.roi_color = roi_color
        self.roi_thickness = roi_thickness
        self.lesion_color = lesion_color

        self.landmark_radius = landmark_radius
        self.landmark_colors = landmark_colors

        self.hist_figsize = hist_figsize
        self.hist_dpi = hist_dpi

    # --- ROI 1: whole-leaf segmentation -------------------------------------
    def _segment_leaf(self, rgb_img):
        """Whole-leaf ROI: S channel -> threshold -> blur -> re-threshold -> fill -> fill_holes.
        Returns (blurred_binary, mask) so every T-method can reuse the same mask
        instead of recomputing it. Tunable via the constructor params set on self.
        """
        rgb_img = rgb_img.astype(np.uint8) if rgb_img.dtype != np.uint8 else rgb_img
        s = pcv.rgb2gray_hsv(rgb_img=rgb_img, channel="s")
        s_thresh = pcv.threshold.binary(s, threshold=self.s_threshold, object_type="light")
        s_blur = pcv.gaussian_blur(img=s_thresh, ksize=self.blur_ksize, sigma_x=self.blur_sigma)
        s_blur = pcv.threshold.binary(s_blur, threshold=self.rethreshold, object_type="light")
        filled = pcv.fill(s_blur, self.fill_size)
        mask = pcv.fill_holes(bin_img=filled)
        return s_blur, mask

    # Backward-compatible alias — old callers/tests using `_segment` keep working.
    _segment = _segment_leaf

    # --- ROI 2: lesion/disease segmentation ---------------------------------
    def _segment_lesion(self, rgb_img, leaf_mask):
        """Lesion ROI: necrotic/chlorotic spots *within* the leaf.

        Healthy leaf tissue skews green, which sits low on the LAB 'a'
        (green-magenta) channel. Disease lesions - browning, yellowing,
        necrotic patches - skew away from green and read as brighter on
        that same channel, so a single-channel threshold separates them
        cleanly without needing a second color model.

        The threshold is restricted to `leaf_mask` via bitwise_and so
        background pixels (which can also be "non-green") never leak in as
        false-positive lesions. A small morphological close then fill pass
        removes speckle noise from anti-aliased edges before contouring.
        """
        rgb_img = rgb_img.astype(np.uint8) if rgb_img.dtype != np.uint8 else rgb_img
        a_channel = pcv.rgb2gray_lab(rgb_img=rgb_img, channel="a")

        if self.lesion_a_threshold is None:
            a_thresh = pcv.threshold.otsu(gray_img=a_channel, object_type="light")
        else:
            a_thresh = pcv.threshold.binary(
                a_channel, threshold=self.lesion_a_threshold, object_type="light"
            )

        # Keep only lesion pixels that fall inside the leaf itself.
        lesion_mask = cv2.bitwise_and(a_thresh, a_thresh, mask=leaf_mask)

        if self.lesion_close_ksize:
            kernel = np.ones((self.lesion_close_ksize, self.lesion_close_ksize), np.uint8)
            lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, kernel)

        lesion_mask = pcv.fill(lesion_mask, self.lesion_fill_size)
        return lesion_mask

    def _segment_both(self, rgb_img):
        """Convenience: compute leaf mask and lesion mask together in one call."""
        _, leaf_mask = self._segment_leaf(rgb_img)
        lesion_mask = self._segment_lesion(rgb_img, leaf_mask)
        return leaf_mask, lesion_mask

    # --- T1 ---------------------------------------------------------------
    def gaussian_blur(self, rgb_img, mask_bin=None):
        """T1: thresholded-then-blurred binary image, before fill/fill_holes."""
        blurred, _ = self._segment_leaf(rgb_img)
        return blurred

    # --- T2 ---------------------------------------------------------------
    def mask(self, rgb_img, mask_bin=None):
        """T2: leaf isolated on a white background."""
        if mask_bin is None:
            _, mask_bin = self._segment_leaf(rgb_img)
        return pcv.apply_mask(img=rgb_img, mask=mask_bin, mask_color="white")

    # --- T3 ---------------------------------------------------------------
    def roi_objects(self, rgb_img, mask_bin=None):
        """T3: green outline of every leaf-ROI object on the original image.

        Uses pcv.create_labels on the binary mask to enumerate all objects
        (rather than picking the single largest contour), then outlines each
        labeled object on a copy of the original image.
        """
        if mask_bin is None:
            _, mask_bin = self._segment_leaf(rgb_img)
        lbl, n_lbl = pcv.create_labels(mask=mask_bin)
        roi_vis = rgb_img.copy()
        for i in range(1, n_lbl + 1):
            obj_mask = (lbl == i).astype(np.uint8)
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(roi_vis, contours, -1, self.roi_color, self.roi_thickness)
        return roi_vis

    def lesion_objects(self, rgb_img, leaf_mask=None, lesion_mask=None):
        """T3b: blue outline of every detected lesion, drawn on top of the
        green leaf-ROI outline so both ROIs are visible in one image.
        """
        if leaf_mask is None:
            _, leaf_mask = self._segment_leaf(rgb_img)
        if lesion_mask is None:
            lesion_mask = self._segment_lesion(rgb_img, leaf_mask)

        vis = self.roi_objects(rgb_img, leaf_mask)
        lbl, n_lbl = pcv.create_labels(mask=lesion_mask)
        for i in range(1, n_lbl + 1):
            obj_mask = (lbl == i).astype(np.uint8)
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, self.lesion_color, self.roi_thickness)
        return vis

    # --- T4 ---------------------------------------------------------------
    def analyze_object(self, rgb_img, mask_bin=None):
        """T4: PlantCV's own shape-analysis overlay + shape metrics dict."""
        if mask_bin is None:
            _, mask_bin = self._segment_leaf(rgb_img)
        lbl, n_lbl = pcv.create_labels(mask=mask_bin)
        pcv.outputs.clear()
        analyze_vis = pcv.analyze.size(img=rgb_img, labeled_mask=lbl, n_labels=n_lbl)
        obs = pcv.outputs.observations["default_1"]
        shape = {
            k: obs[k]["value"]
            for k in [
                "area",
                "perimeter",
                "solidity",
                "convex_hull_area",
                "longest_path",
                "ellipse_eccentricity",
                "width",
                "height",
            ]
        }
        shape["circularity"] = 4 * np.pi * shape["area"] / (shape["perimeter"] ** 2 + 1e-6)
        shape["extent"] = shape["area"] / (shape["width"] * shape["height"])
        return analyze_vis, shape

    def disease_severity(self, leaf_mask, lesion_mask):
        """Pixel-count based severity: lesion area as a percentage of leaf area."""
        leaf_area_px = int(np.count_nonzero(leaf_mask))
        lesion_area_px = int(np.count_nonzero(lesion_mask))
        severity_pct = 100.0 * lesion_area_px / leaf_area_px if leaf_area_px else 0.0
        return {
            "leaf_area_px": leaf_area_px,
            "lesion_area_px": lesion_area_px,
            "severity_pct": severity_pct,
        }

    # --- T5 ---------------------------------------------------------------
    def pseudolandmarks(self, rgb_img, mask_bin=None):
        """T5: x-axis pseudolandmarks, colored by group (top/bottom/center)."""
        if mask_bin is None:
            _, mask_bin = self._segment_leaf(rgb_img)
        top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
            img=rgb_img, mask=mask_bin, label="leaf"
        )
        plm_vis = rgb_img.copy()
        for group, color in zip((top, bottom, center_v), self.landmark_colors):
            for pt in group:
                cv2.circle(plm_vis, (int(pt[0][0]), int(pt[0][1])), self.landmark_radius, color, -1)
        return plm_vis

    # --- T6 ---------------------------------------------------------------
    def color_histogram(self, rgb_img, mask_bin=None):
        """T6: combined RGB + HSV + LAB channel histogram, as an RGB image array.

        PlantCV v4's pcv.analyze.color() renders a ridgeline plot of a single
        colorspace by default and no longer produces the classic overlaid
        multi-channel line chart. We still call it (colorspaces="all") to
        populate pcv.outputs with all 9 channel frequency distributions, draw
        the overlay chart ourselves, then immediately rasterize it to a numpy
        array and close the source figure.

        Rasterizing here (rather than returning the live Figure) matters:
        an open matplotlib Figure is auto-displayed by Jupyter's inline
        backend at the end of whichever cell created it - so calling
        transform()/color_histogram() on its own would show the histogram
        immediately, before display_results() ever runs it in the intended
        spot. Closing the figure inside this method means it can only ever
        appear where the caller explicitly imshow()s the returned array -
        same as every other T-method here.
        """
        if mask_bin is None:
            _, mask_bin = self._segment_leaf(rgb_img)
        lbl, n_lbl = pcv.create_labels(mask=mask_bin)
        pcv.outputs.clear()
        pcv.analyze.color(rgb_img=rgb_img, labeled_mask=lbl, n_labels=n_lbl, colorspaces="all")
        obs = pcv.outputs.observations["default_1"]

        fig, ax = plt.subplots(figsize=self.hist_figsize)
        for label, (key, color) in self._HIST_CHANNELS.items():
            vals = obs[key]["value"]  # already a percentage distribution (sums to 100)
            ax.plot(range(len(vals)), vals, label=label, color=color, linewidth=1)

        ax.set_xlabel("Pixel intensity")
        ax.set_ylabel("Proportion of pixels (%)")
        ax.set_xlim(0, 255)
        ax.legend(title="color Channel", bbox_to_anchor=(1.02, 0.5), loc="center left")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.hist_dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return np.array(Image.open(buf).convert("RGB"))

    # --- Orchestration ------------------------------------------------------
    def transform(self, rgb):
        """Run all transformations, now including both ROIs. Returns result dict."""
        blurred, leaf_mask = self._segment_leaf(rgb)
        lesion_mask = self._segment_lesion(rgb, leaf_mask)

        masked = self.mask(rgb, leaf_mask)
        roi_vis = self.roi_objects(rgb, leaf_mask)
        lesion_vis = self.lesion_objects(rgb, leaf_mask, lesion_mask)
        analyze_vis, shape = self.analyze_object(rgb, leaf_mask)
        plm_vis = self.pseudolandmarks(rgb, leaf_mask)
        color_hist = self.color_histogram(rgb, leaf_mask)
        severity = self.disease_severity(leaf_mask, lesion_mask)

        return {
            "rgb": rgb,
            "blurred": blurred,
            "mask": leaf_mask,
            "lesion_mask": lesion_mask,
            "masked": masked,
            "roi_vis": roi_vis,
            "lesion_vis": lesion_vis,
            "analyze_vis": analyze_vis,
            "plm_vis": plm_vis,
            "color_hist": color_hist,
            "shape": shape,
            "severity": severity,
        }

    def display_results(self, results):
        """Show order: T1-T5(+lesion) grid -> shape/severity metrics -> T6 color histogram.

        Uses plt.show() throughout, which renders inline under Jupyter and opens
        a blocking window under a plain `python script.py` run - no branching needed.
        """
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        panels = [
            (axes[0, 0], results["rgb"], "Original"),
            (axes[0, 1], results["blurred"], "T1: Gaussian Blur"),
            (axes[0, 2], results["masked"], "T2: Mask"),
            (axes[0, 3], results["roi_vis"], "T3: Leaf ROI"),
            (axes[1, 0], results["lesion_vis"], "T3b: Leaf ROI + Lesion ROI"),
            (axes[1, 1], results["analyze_vis"], "T4: Analyze Object"),
            (axes[1, 2], results["plm_vis"], "T5: Pseudolandmarks"),
        ]
        for ax, img, title in panels:
            cmap = "gray" if img.ndim == 2 else None
            ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.axis("off")
        axes[1, 3].axis("off")
        fig.suptitle("Part 3 — Image Transformations", fontsize=15)
        fig.tight_layout()
        plt.show()
        plt.close(fig)

        print("\nShape metrics (leaf ROI):")
        for k, v in results["shape"].items():
            print(f"  {k:20s}  {v:.4f}")

        print("\nDisease severity (leaf ROI vs lesion ROI):")
        for k, v in results["severity"].items():
            print(f"  {k:20s}  {v:.4f}" if isinstance(v, float) else f"  {k:20s}  {v}")

        # T6: color_hist is a plain RGB array (see color_histogram) - shown
        # last, deliberately, so it always renders at the bottom of the output.
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.imshow(results["color_hist"])
        ax2.axis("off")
        ax2.set_title("T6: Color Histogram")
        fig2.tight_layout()
        plt.show()
        plt.close(fig2)

    def save_transformations(self, img_path, dst_dir):
        """Apply all transformations to one image and save results."""
        rgb = self.loader.load_rgb(img_path)
        stem = Path(img_path).stem
        ext = Path(img_path).suffix
        results = self.transform(rgb)

        files = {
            "Gaussian_Blur": results["blurred"],
            "Mask": results["masked"],
            "ROI_Objects": cv2.cvtColor(results["roi_vis"], cv2.COLOR_RGB2BGR),
            "Lesion_ROI": cv2.cvtColor(results["lesion_vis"], cv2.COLOR_RGB2BGR),
            "Analyze_Object": cv2.cvtColor(results["analyze_vis"], cv2.COLOR_RGB2BGR),
            "Pseudolandmarks": cv2.cvtColor(results["plm_vis"], cv2.COLOR_RGB2BGR),
            "Color_Histogram": cv2.cvtColor(results["color_hist"], cv2.COLOR_RGB2BGR),
        }
        for name, img_arr in files.items():
            cv2.imwrite(str(dst_dir / f"{stem}_{name}{ext}"), img_arr)

    def process_directory(self, src_dir, dst_dir):
        """Batch-process all images in a directory."""
        src_dir = Path(src_dir)
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        images = [f for f in src_dir.iterdir() if f.is_file()]
        print(f"Processing {len(images)} images from {src_dir} -> {dst_dir}")
        for img_path in sorted(images):
            try:
                self.save_transformations(img_path, dst_dir)
                print(f"  [OK] {img_path.name}")
            except Exception as e:
                print(f"  [FAIL] {img_path.name}: {e}")
        print(f"Done. Transformed images saved to {dst_dir}")