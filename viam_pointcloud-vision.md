# Model viam:vision:pointcloud-vision

A Vision service that performs point cloud classification using machine learning models. This service captures point cloud data from a camera, preprocesses it (normalization, sampling), and runs inference through an ML model to classify the scene or object.

## Features

- Automatic extraction of model requirements from ML model metadata (input shape, feature count, class labels)
- Support for models expecting XYZ coordinates (3 features), XYZ+RGB (6 features), or XYZ+RGB+normals (9 features)
- Point cloud normalization to unit sphere
- Configurable sampling methods for matching model's expected point count
- Returns top N classifications with confidence scores
- Reads class names from model metadata when available

## Configuration
The following attribute template can be used to configure this model:

```json
{
  "mlmodel_name": "<string>",
  "camera_name": "<string>",
  "sampling_method": "<string>"
}
```

### Attributes

The following attributes are available for this model:

| Name               | Type   | Inclusion | Description                                                                                     |
|--------------------|--------|-----------|------------------------------------------------------------------------------------------------|
| `mlmodel_name`     | string | Required  | Name of the ML model service to use for inference                                              |
| `camera_name`      | string | Optional  | Default camera to use when no camera is specified in method calls                              |
| `sampling_method`  | string | Optional  | Point cloud sampling method: "random", "voxel", or "fps". Default: "random". Only "random" is currently fully implemented. |

### Example Configuration

```json
{
  "mlmodel_name": "my_pointcloud_classifier",
  "camera_name": "my_camera",
  "sampling_method": "random"
}
```

## Usage

### Getting Classifications from a Camera

The primary method for this service is `get_classifications_from_camera()`, which:

1. Retrieves point cloud data from the specified camera
2. Parses and preprocesses the point cloud according to model requirements
3. Runs inference through the ML model
4. Returns the top N classifications with confidence scores

Example usage via the Viam SDK:

```python
from viam.services.vision import VisionClient

# Get the vision service
vision = VisionClient.from_robot(robot, "my_classifier")

# Get top 5 classifications from a camera
classifications = await vision.get_classifications_from_camera(
    camera_name="my_camera",
    count=5
)

for c in classifications:
    print(f"{c.class_name}: {c.confidence:.2%}")
```

### Capture All from Camera

The `capture_all_from_camera()` method supports retrieving both images and classifications in a single call:

```python
result = await vision.capture_all_from_camera(
    camera_name="my_camera",
    return_image=True,
    return_classifications=True,
    extra={"count": 3}
)

# Access image
if result.image:
    print(f"Got image: {result.image.width}x{result.image.height}")

# Access classifications
for c in result.classifications:
    print(f"{c.class_name}: {c.confidence:.2%}")
```

## Model Requirements

Your ML model should:

1. **Input**: Accept point cloud data with shape `[N, F]` or `[1, N, F]` where:
   - `N` = number of points (fixed, will be enforced by sampling)
   - `F` = number of features per point (3, 6, or 9)

2. **Output**: Return classification logits with shape `[num_classes]` or `[1, num_classes]`

3. **Metadata**: Optionally include class labels in the output metadata's `extra` field under the key `"labels"` pointing to a text file with one class name per line

### Supported Feature Configurations

- **3 features**: XYZ coordinates only
- **6 features**: XYZ coordinates + RGB colors
- **9 features**: XYZ coordinates + RGB colors + normals

The service automatically extracts these requirements from your model's metadata and validates that the point cloud has the necessary data.

## Preprocessing Details

The service performs the following preprocessing steps:

1. **Feature Extraction**: Extracts XYZ, RGB (if needed), and normals (if needed) from the point cloud
2. **Sampling**: Resamples to match the model's expected point count using the configured sampling method
3. **Normalization**: Normalizes XYZ coordinates to a unit sphere (centered at origin, scaled by max distance)
4. **Batch Dimension**: Adds batch dimension if required by the model

## Limitations

The following Vision service methods are **not implemented**:

- `get_classifications()` - Point cloud classification requires camera input; use `get_classifications_from_camera()` instead
- `get_detections()` and `get_detections_from_camera()` - Not supported
- `get_object_point_clouds()` - Not supported
- `do_command()` - Not supported

