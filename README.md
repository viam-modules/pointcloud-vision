# Point Cloud Classification Module

A Viam module that implements a Vision service for point cloud classification using machine learning models. This module enables robots to classify 3D scenes and objects by processing point cloud data from depth cameras.

## Overview

The `viam:vision:pointcloud-vision` Vision service captures point cloud data from a camera, preprocesses it (normalization, sampling), and runs inference through an ML model to produce classification results with confidence scores.

### Key Features

- **Automatic Model Adaptation**: Extracts model requirements from ML model metadata (input shape, feature count, class labels)
- **Flexible Feature Support**: Works with models expecting XYZ coordinates only, XYZ+RGB, or XYZ+RGB+normals
- **Point Cloud Preprocessing**: Normalization to unit sphere and configurable sampling methods
- **Seamless Integration**: Implements the standard Viam Vision service API for easy integration with existing Viam robots

## Prerequisites

- Python 3.11 or higher
- A Viam robot configuration with:
  - A camera component that supports point cloud capture
  - An ML model service configured with a point cloud classification model

## Installation

### From Viam Registry

Add this module to your robot configuration through the Viam app:

1. Navigate to your robot's config page
2. Click the **Services** tab
3. Click **Create service**
4. Search for `viam:vision:pointcloud-vision`
5. Click **Add module**
6. Configure the service attributes (see Configuration section)

### Local Development

1. Clone this repository:
```bash
git clone <repository-url>
cd pointcloud-classification
```

2. Run the setup script to install dependencies:
```bash
./setup.sh
```

This will install `uv` (if needed), create a virtual environment, and sync dependencies.

## Configuration

Add the classifier as a Vision service in your robot configuration:

```json
{
  "name": "my_classifier",
  "type": "vision",
  "namespace": "rdk",
  "model": "viam:vision:pointcloud-vision",
  "attributes": {
    "mlmodel_name": "my_pointcloud_model",
    "camera_name": "my_depth_camera",
    "sampling_method": "random"
  }
}
```

### Configuration Attributes

| Name              | Type   | Required | Description |
|-------------------|--------|----------|-------------|
| `mlmodel_name`    | string | Yes      | Name of the ML model service to use for inference |
| `camera_name`     | string | No       | Default camera to use when not specified in method calls |
| `sampling_method` | string | No       | Sampling method: "random", "voxel", or "fps" (default: "random") |

## Usage

### Python SDK Example

```python
from viam.robot.client import RobotClient
from viam.services.vision import VisionClient

async def classify_scene():
    # Connect to robot
    robot = await RobotClient.at_address('<robot-address>')

    # Get the vision service
    classifier = VisionClient.from_robot(robot, "my_classifier")

    # Get classifications from camera
    classifications = await classifier.get_classifications_from_camera(
        camera_name="my_depth_camera",
        count=5
    )

    # Print results
    for c in classifications:
        print(f"{c.class_name}: {c.confidence:.2%}")

    await robot.close()
```

### Using capture_all_from_camera

```python
# Get both image and classifications in one call
result = await classifier.capture_all_from_camera(
    camera_name="my_depth_camera",
    return_image=True,
    return_classifications=True,
    extra={"count": 3}
)

if result.image:
    print(f"Captured image: {result.image.width}x{result.image.height}")

for c in result.classifications:
    print(f"{c.class_name}: {c.confidence:.2%}")
```

## ML Model Requirements

Your ML model should meet these requirements:

### Input
- Shape: `[N, F]` or `[1, N, F]` where:
  - `N` = number of points (fixed)
  - `F` = 3 (XYZ), 6 (XYZ+RGB), or 9 (XYZ+RGB+normals)

### Output
- Shape: `[num_classes]` or `[1, num_classes]`
- Format: Classification logits (will be converted to probabilities via softmax)

### Metadata (Optional)
Include class labels in the output metadata's `extra` field:
```json
{
  "labels": "/path/to/labels.txt"
}
```

Where `labels.txt` contains one class name per line.

## Development

### Running Locally

```bash
./run.sh
```

This runs the module locally using the virtual environment. The module will be available through the Viam SDK module registry.

### Building for Distribution

```bash
./build.sh
```

This uses PyInstaller to create a standalone executable with all dependencies bundled, then packages it as `dist/archive.tar.gz` for Viam module deployment.

### Testing

```bash
uv run pytest
```

## Architecture

The module is structured as follows:

- **src/main.py**: Entry point that runs the module using `Module.run_from_registry()`
- **src/models/classifier.py**: Core Vision service implementation
  - Implements `get_classifications_from_camera()` for point cloud classification
  - Implements `capture_all_from_camera()` for combined image and classification capture
  - Handles point cloud preprocessing (normalization, sampling, feature extraction)
  - Manages ML model inference and result conversion

### Preprocessing Pipeline

1. **Point Cloud Capture**: Retrieves point cloud from camera
2. **Feature Extraction**: Extracts XYZ, RGB (if needed), and normals (if needed)
3. **Sampling**: Resamples to match model's expected point count
4. **Normalization**: Normalizes XYZ coordinates to unit sphere
5. **Inference**: Runs ML model inference
6. **Post-processing**: Converts logits to classifications with confidence scores

## Supported Platforms

- linux/amd64
- linux/arm64
- darwin/arm64
- windows/amd64

## Models

This module provides the following Vision service:

- [`viam:vision:pointcloud-vision`](viam_pointcloud-vision.md) - Vision service for point cloud classification using ML models

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or pull request.
