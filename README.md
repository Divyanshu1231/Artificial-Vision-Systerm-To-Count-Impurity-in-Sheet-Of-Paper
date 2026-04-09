# Artificial Vision System to Count Impurity in a Sheet of Paper

This project detects visible impurities in a paper-sheet image using OpenCV and exports the detection results for inspection.

## Features

- Counts impurities from a selected image
- Highlights contours, centers, and bounding boxes
- Supports `adaptive` and `binary` threshold modes
- Filters out small noise using configurable minimum area
- Saves annotated image, mask image, CSV table, and JSON report

## Setup

```powershell
pip install -r requirements.txt
```

## Usage

```powershell
python "Artificial Vision System to Count Impurity in a Sheet of Paper.py" --image "input_image.jpg"
```

Optional arguments:

- `--output-dir analysis_output`
- `--min-area 50`
- `--blur-kernel 5`
- `--threshold-mode adaptive`
- `--binary-threshold 127`
- `--show-windows`

## Output Files

The script creates these files inside the output directory:

- `annotated_image.jpg`
- `impurity_mask.jpg`
- `impurities.csv`
- `report.json`
