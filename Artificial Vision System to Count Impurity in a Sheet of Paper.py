import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Impurity:
    index: int
    area: float
    perimeter: float
    x: int
    y: int
    width: int
    height: int
    center_x: int
    center_y: int


def build_parser():
    parser = argparse.ArgumentParser(description="Multi-image impurity detection")

    parser.add_argument(
        "--images",
        nargs="+",
        default=[
            "C:\\Users\\honey\\OneDrive\\Documents\\New folder (2)\\.vscode\\Artificial Vision System to Count Impurity in a Sheet of Paper\\paper1.jpg",
            "C:\\Users\\honey\\OneDrive\\Documents\\New folder (2)\\.vscode\\Artificial Vision System to Count Impurity in a Sheet of Paper\\paper2.jpg",
            "C:\\Users\\honey\\OneDrive\\Documents\\New folder (2)\\.vscode\\Artificial Vision System to Count Impurity in a Sheet of Paper\\paper3.jpg"
        ],
        help="List of image paths"
    )

    parser.add_argument("--output-dir", default="analysis_output")
    parser.add_argument("--min-area", type=float, default=50.0)
    parser.add_argument("--threshold-mode", choices=("adaptive", "binary"), default="adaptive")
    parser.add_argument("--binary-threshold", type=int, default=127)

    # 🔥 AUTO SHOW WINDOWS (no need to pass argument)
    parser.add_argument("--show-windows", action="store_true", default=True)

    return parser


def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot load {path}")
    return img


def create_mask(gray, mode, binary_threshold):
    if mode == "adaptive":
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 10
        )
    _, mask = cv2.threshold(gray, binary_threshold, 255, cv2.THRESH_BINARY_INV)
    return mask


def detect(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    impurities = []
    kept = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w//2, y + h//2

        imp = Impurity(len(impurities)+1, area, cv2.arcLength(c, True),
                       x, y, w, h, cx, cy)

        impurities.append(imp)
        kept.append(c)

    return impurities, kept


def annotate(img, impurities, contours):
    out = img.copy()

    for imp, c in zip(impurities, contours):
        cv2.drawContours(out, [c], -1, (0,0,255), 2)
        cv2.rectangle(out, (imp.x, imp.y),
                      (imp.x+imp.width, imp.y+imp.height),
                      (0,255,0), 2)

    return out


def main():
    parser = build_parser()
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    for img_path in args.images:
        img_path = Path(img_path)
        print(f"\nProcessing: {img_path}")

        img = load_image(img_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        mask = create_mask(blur, args.threshold_mode, args.binary_threshold)

        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        impurities, contours = detect(mask, args.min_area)

        annotated = annotate(img, impurities, contours)

        # 💾 SAVE
        cv2.imwrite(str(out_dir / f"{img_path.stem}_annotated.jpg"), annotated)
        cv2.imwrite(str(out_dir / f"{img_path.stem}_mask.jpg"), mask)

        print(f"Detected impurities: {len(impurities)}")

        # 🔥 SHOW BOTH WINDOWS (FIXED)
        if args.show_windows:
            cv2.imshow(f"{img_path.name} - Annotated", annotated)
            cv2.imshow(f"{img_path.name} - Mask", mask)

    if args.show_windows:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()