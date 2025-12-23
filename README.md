# Icarus

AI-powered photo cropping CLI that automatically crops images around detected people using ONNX models.

## Features

- **Smart Person Detection**: Uses YOLOv8, YOLO-NAS, or R-CNN models to detect people in photos
- **Aspect Ratio Presets**: Portrait (9:16), mobile (9:20), and landscape (21:9) with automatic best-fit selection
- **Batch Processing**: Process single images, directories, or recursive traversals
- **Flexible Workflows**: Dry-run, visualization, and auto-aspect modes
- **Configurable Cropping**: Adjust detection thresholds, center weighting, and aspect-specific constraints

## Quick Start

```bash
# Crop a single image
icarus crop photo.jpg --output cropped.jpg

# Process a directory with auto-aspect selection
icarus crop-dir ./photos --output ./cropped --auto-aspect

# Generate all valid aspect ratios
icarus crop photo.jpg --all-aspects --output ./results

# Dry run to preview cropping decisions
icarus crop photo.jpg --dry-run

# Visualize detections
icarus crop photo.jpg --visualize
```

## Model Management

```bash
# List available models
icarus models list

# Download a model
icarus models download yolov8n
```

## Configuration

Settings are managed via TOML configuration file:
- Detection confidence and NMS thresholds
- Center weight mode (area, confidence, or equal)
- Aspect ratio definitions and minimum dimensions
- Model provider settings (CPU-only in MVP)

## Output Organization

When using `--all-aspects`, crops are organized into subdirectories:
```
output/
├── portrait/
│   └── photo.jpg
├── mobile/
│   └── photo.jpg
└── landscape/
    └── photo.jpg
```

Only aspects that meet automatic minimum dimension requirements are generated.

## Requirements

- ONNX Runtime (CPU)
- Supported image formats: JPEG, PNG, WebP, TIFF

## Performance

- Target: <500ms per image on CPU
- Batch processing: 10+ images/sec with parallelization

## License

See LICENSE file for details.
