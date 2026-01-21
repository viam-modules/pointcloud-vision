import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.models.classifier import Classifier
from src.utils.pointcloud import PointCloud


def _create_pcd_bytes(points: np.ndarray, colors: np.ndarray = None) -> bytes:
    """Helper to create PCD format bytes from numpy arrays"""
    num_points = points.shape[0]
    has_colors = colors is not None

    # Create header
    header_lines = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
    ]

    if has_colors:
        header_lines.append("FIELDS x y z rgb")
        header_lines.append("SIZE 4 4 4 4")
        header_lines.append("TYPE F F F F")
        header_lines.append("COUNT 1 1 1 1")
    else:
        header_lines.append("FIELDS x y z")
        header_lines.append("SIZE 4 4 4")
        header_lines.append("TYPE F F F")
        header_lines.append("COUNT 1 1 1")

    header_lines.extend(
        [
            f"WIDTH {num_points}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {num_points}",
            "DATA ascii",
        ]
    )

    # Create data lines
    data_lines = []
    for i in range(num_points):
        x, y, z = points[i]
        if has_colors:
            # Convert RGB [0,1] to packed integer
            r, g, b = colors[i]
            rgb_int = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
            data_lines.append(f"{x} {y} {z} {rgb_int}")
        else:
            data_lines.append(f"{x} {y} {z}")

    # Combine and encode
    pcd_text = "\n".join(header_lines + data_lines) + "\n"
    return pcd_text.encode("ascii")


def test_parse_metadata_no_batch_dim():
    """Test parsing metadata with shape [N, F]"""
    # Create mock metadata
    mock_input = MagicMock()
    mock_input.name = "points"
    mock_input.shape = [1024, 3]
    mock_input.data_type = "float32"

    mock_output = MagicMock()
    mock_output.name = "logits"
    mock_output.shape = [10]
    mock_output.associated_files = []

    mock_metadata = MagicMock()
    mock_metadata.input_info = [mock_input]
    mock_metadata.output_info = [mock_output]

    classifier = MagicMock(spec=Classifier)
    result = Classifier._parse_metadata(classifier, mock_metadata)

    (
        input_name,
        target_points,
        target_features,
        has_batch_dim,
        output_name,
        class_names,
    ) = result

    assert input_name == "points"
    assert target_points == 1024
    assert target_features == 3
    assert has_batch_dim == False
    assert output_name == "logits"
    assert class_names is None


def test_parse_metadata_with_batch_dim():
    """Test parsing metadata with shape [1, N, F]"""
    mock_input = MagicMock()
    mock_input.name = "input_points"
    mock_input.shape = [1, 2048, 6]
    mock_input.data_type = "float32"

    mock_output = MagicMock()
    mock_output.name = "output_logits"
    mock_output.shape = [1, 5]
    mock_output.associated_files = []

    mock_metadata = MagicMock()
    mock_metadata.input_info = [mock_input]
    mock_metadata.output_info = [mock_output]

    classifier = MagicMock(spec=Classifier)
    result = Classifier._parse_metadata(classifier, mock_metadata)

    (
        input_name,
        target_points,
        target_features,
        has_batch_dim,
        output_name,
        class_names,
    ) = result

    assert input_name == "input_points"
    assert target_points == 2048
    assert target_features == 6
    assert has_batch_dim == True
    assert output_name == "output_logits"


def test_parse_metadata_dynamic_shape_raises():
    """Test that dynamic shapes raise ValueError"""
    mock_input = MagicMock()
    mock_input.name = "points"
    mock_input.shape = [-1, 3]
    mock_input.data_type = "float32"

    mock_metadata = MagicMock()
    mock_metadata.input_info = [mock_input]

    classifier = MagicMock(spec=Classifier)

    with pytest.raises(ValueError, match="dynamic input shape"):
        Classifier._parse_metadata(classifier, mock_metadata)


def test_logits_to_classifications_with_labels():
    """Test converting logits to classifications with class labels"""
    logits = np.array([2.0, 1.0, 3.0, 0.5, 0.1])
    class_names = ["cat", "dog", "bird", "fish", "lizard"]
    count = 3

    classifier = MagicMock(spec=Classifier)
    result = Classifier._logits_to_classifications(
        classifier, logits, class_names, count
    )

    assert len(result) == 3
    assert result[0].class_name == "bird"  # Highest logit (3.0)
    assert result[1].class_name == "cat"  # Second highest (2.0)
    assert result[2].class_name == "dog"  # Third highest (1.0)

    # Check confidences are valid probabilities
    assert 0 <= result[0].confidence <= 1
    assert 0 <= result[1].confidence <= 1
    assert 0 <= result[2].confidence <= 1

    # Check highest confidence corresponds to highest logit
    assert result[0].confidence > result[1].confidence > result[2].confidence


def test_logits_to_classifications_no_labels():
    """Test converting logits without class labels (use indices)"""
    logits = np.array([1.0, 3.0, 2.0])
    class_names = None
    count = 2

    classifier = MagicMock(spec=Classifier)
    result = Classifier._logits_to_classifications(
        classifier, logits, class_names, count
    )

    assert len(result) == 2
    assert result[0].class_name == "1"  # Index of highest logit
    assert result[1].class_name == "2"  # Index of second highest


def test_normalize_point_cloud():
    """Test point cloud normalization to unit sphere"""
    # Create test points: cube from -10 to 10
    points = np.array(
        [
            [10.0, 10.0, 10.0],
            [-10.0, -10.0, -10.0],
            [5.0, 5.0, 5.0],
            [0.0, 0.0, 0.0],
        ]
    )

    classifier = MagicMock(spec=Classifier)
    result = Classifier._normalize_point_cloud(classifier, points)

    # Check centered (mean near zero)
    assert np.allclose(result.mean(axis=0), [0, 0, 0], atol=1e-10)

    # Check scaled (max distance from center is 1)
    distances = np.sqrt((result**2).sum(axis=1))
    assert np.allclose(distances.max(), 1.0, atol=1e-6)

    # Check shape unchanged
    assert result.shape == points.shape


def test_sample_point_cloud_downsample():
    """Test downsampling to fewer points"""
    points = np.random.rand(2000, 3)
    target_count = 1024

    classifier = MagicMock(spec=Classifier)
    result = Classifier._sample_point_cloud(classifier, points, target_count, "random")

    assert result.shape == (1024, 3)
    # Check all result points exist in original
    for point in result[:5]:  # Check first few
        # Should be close to one of the original points
        distances = np.sqrt(((points - point) ** 2).sum(axis=1))
        assert distances.min() < 1e-6


def test_sample_point_cloud_upsample():
    """Test upsampling to more points (with duplication)"""
    points = np.random.rand(100, 3)
    target_count = 500

    classifier = MagicMock(spec=Classifier)
    result = Classifier._sample_point_cloud(classifier, points, target_count, "random")

    assert result.shape == (500, 3)
    # All result points should match one of the original points
    for point in result[:10]:  # Check first few
        distances = np.sqrt(((points - point) ** 2).sum(axis=1))
        assert distances.min() < 1e-6


def test_sample_point_cloud_exact_count():
    """Test when point count already matches"""
    points = np.random.rand(1024, 3)
    target_count = 1024

    classifier = MagicMock(spec=Classifier)
    result = Classifier._sample_point_cloud(classifier, points, target_count, "random")

    assert result.shape == (1024, 3)


def test_parse_point_cloud_pcd_format():
    """Test parsing PCD format point cloud bytes"""
    # Create test points
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    pcd_bytes = _create_pcd_bytes(points)

    classifier = MagicMock(spec=Classifier)
    result = Classifier._parse_point_cloud(classifier, pcd_bytes, "application/pcd")

    assert isinstance(result, PointCloud)
    assert result.points.shape == (3, 3)
    assert np.allclose(result.points[0], [1.0, 2.0, 3.0])


def test_preprocess_point_cloud_xyz_only():
    """Test preprocessing with XYZ features only"""
    # Create PointCloud
    points = np.random.rand(2000, 3) * 10
    pcd = PointCloud(points=points)

    target_points = 1024
    target_features = 3

    classifier = MagicMock(spec=Classifier)
    classifier._sample_point_cloud = Classifier._sample_point_cloud.__get__(
        classifier, Classifier
    )
    classifier._normalize_point_cloud = Classifier._normalize_point_cloud.__get__(
        classifier, Classifier
    )

    result = Classifier._preprocess_point_cloud(
        classifier, pcd, target_points, target_features, "random"
    )

    assert result.shape == (1024, 3)
    # Check normalized (max value should be ~1)
    assert np.abs(result).max() <= 1.0 + 1e-6


def test_preprocess_point_cloud_xyz_rgb():
    """Test preprocessing with XYZ+RGB features"""
    # Create PointCloud with colors
    points = np.random.rand(500, 3) * 10
    colors = np.random.rand(500, 3)
    pcd = PointCloud(points=points, colors=colors)

    target_points = 256
    target_features = 6

    classifier = MagicMock(spec=Classifier)
    classifier._sample_point_cloud = Classifier._sample_point_cloud.__get__(
        classifier, Classifier
    )
    classifier._normalize_point_cloud = Classifier._normalize_point_cloud.__get__(
        classifier, Classifier
    )

    result = Classifier._preprocess_point_cloud(
        classifier, pcd, target_points, target_features, "random"
    )

    assert result.shape == (256, 6)
    # XYZ should be normalized
    assert np.abs(result[:, :3]).max() <= 1.0 + 1e-6
    # RGB should be in [0, 1]
    assert result[:, 3:].min() >= 0.0
    assert result[:, 3:].max() <= 1.0


def test_preprocess_point_cloud_missing_rgb():
    """Test error when RGB required but not available"""
    # Create point cloud without colors
    points = np.random.rand(100, 3)
    pcd = PointCloud(points=points)

    classifier = MagicMock(spec=Classifier)

    with pytest.raises(ValueError, match="requires RGB"):
        Classifier._preprocess_point_cloud(classifier, pcd, 100, 6, "random")


@pytest.mark.asyncio
async def test_get_classifications_from_camera_integration():
    """Integration test for get_classifications_from_camera"""
    # Setup mocks
    mock_camera = AsyncMock()
    mock_mlmodel = AsyncMock()

    # Create test point cloud bytes
    points = np.random.rand(2000, 3) * 10
    pcd_bytes = _create_pcd_bytes(points)

    mock_camera.get_point_cloud = AsyncMock(return_value=(pcd_bytes, "application/pcd"))

    # Mock metadata
    mock_input = MagicMock()
    mock_input.name = "points"
    mock_input.shape = [1024, 3]

    mock_output = MagicMock()
    mock_output.name = "logits"
    mock_output.shape = [3]
    mock_output.associated_files = []

    mock_metadata = MagicMock()
    mock_metadata.input_info = [mock_input]
    mock_metadata.output_info = [mock_output]
    mock_mlmodel.metadata = AsyncMock(return_value=mock_metadata)

    # Mock inference output
    output_logits = np.array([2.0, 1.0, 3.0])
    mock_mlmodel.infer = AsyncMock(return_value={"logits": output_logits})

    # Create classifier instance
    from viam.proto.app.robot import ComponentConfig
    from google.protobuf.struct_pb2 import Struct

    config = ComponentConfig()
    config.name = "test_classifier"
    attrs = Struct()
    attrs["mlmodel_name"] = "test_model"
    attrs["camera_name"] = "test_camera"
    config.attributes.CopyFrom(attrs)

    # Create mock dependencies
    from viam.components.camera import Camera
    from viam.services.mlmodel import MLModel

    dependencies = {
        Camera.get_resource_name("test_camera"): mock_camera,
        MLModel.get_resource_name("test_model"): mock_mlmodel,
    }

    classifier = Classifier.new(config, dependencies)

    # Test
    results = await classifier.get_classifications_from_camera("test_camera", count=2)

    assert len(results) == 2
    assert results[0].class_name == "2"  # Highest logit
    assert results[1].class_name == "0"  # Second highest
    assert results[0].confidence > results[1].confidence


@pytest.mark.asyncio
async def test_capture_all_from_camera_with_classifications():
    """Test capture_all_from_camera returns classifications"""
    # Setup mocks (similar to previous test)
    mock_camera = AsyncMock()
    mock_mlmodel = AsyncMock()

    points = np.random.rand(1000, 3) * 5
    pcd_bytes = _create_pcd_bytes(points)

    mock_camera.get_point_cloud = AsyncMock(return_value=(pcd_bytes, "application/pcd"))

    mock_input = MagicMock()
    mock_input.name = "points"
    mock_input.shape = [512, 3]

    mock_output = MagicMock()
    mock_output.name = "logits"
    mock_output.shape = [3]
    mock_output.associated_files = []

    mock_metadata = MagicMock()
    mock_metadata.input_info = [mock_input]
    mock_metadata.output_info = [mock_output]
    mock_mlmodel.metadata = AsyncMock(return_value=mock_metadata)
    mock_mlmodel.infer = AsyncMock(return_value={"logits": np.array([1, 2, 3, 4, 5])})

    from viam.proto.app.robot import ComponentConfig
    from google.protobuf.struct_pb2 import Struct
    from viam.components.camera import Camera
    from viam.services.mlmodel import MLModel

    config = ComponentConfig()
    config.name = "test"
    attrs = Struct()
    attrs["mlmodel_name"] = "model"
    attrs["camera_name"] = "camera"
    config.attributes.CopyFrom(attrs)

    dependencies = {
        Camera.get_resource_name("camera"): mock_camera,
        MLModel.get_resource_name("model"): mock_mlmodel,
    }

    classifier = Classifier.new(config, dependencies)

    # Test with return_classifications=True
    result = await classifier.capture_all_from_camera(
        "camera", return_classifications=True
    )

    assert result.classifications is not None
    assert len(result.classifications) > 0
    assert result.classifications[0].class_name == "4"  # Highest logit (5.0 at index 4)


@pytest.mark.asyncio
async def test_get_classifications_raises_not_implemented():
    """Test get_classifications raises helpful error"""
    from viam.proto.app.robot import ComponentConfig
    from google.protobuf.struct_pb2 import Struct
    from viam.services.mlmodel import MLModel

    config = ComponentConfig()
    config.name = "test"
    attrs = Struct()
    attrs["mlmodel_name"] = "model"
    config.attributes.CopyFrom(attrs)

    dependencies = {
        MLModel.get_resource_name("model"): AsyncMock(),
    }

    classifier = Classifier.new(config, dependencies)

    mock_image = MagicMock()

    with pytest.raises(NotImplementedError, match="not supported"):
        await classifier.get_classifications(mock_image, count=5)


def test_reconfigure_with_sampling_method():
    """Test that sampling_method is stored from config"""
    from viam.proto.app.robot import ComponentConfig
    from google.protobuf.struct_pb2 import Struct
    from viam.services.mlmodel import MLModel

    config = ComponentConfig()
    config.name = "test"
    attrs = Struct()
    attrs["mlmodel_name"] = "model"
    attrs["sampling_method"] = "voxel"
    config.attributes.CopyFrom(attrs)

    dependencies = {
        MLModel.get_resource_name("model"): MagicMock(),
    }

    classifier = Classifier.new(config, dependencies)

    assert classifier.sampling_method == "voxel"


def test_reconfigure_default_sampling_method():
    """Test default sampling_method is random"""
    from viam.proto.app.robot import ComponentConfig
    from google.protobuf.struct_pb2 import Struct
    from viam.services.mlmodel import MLModel

    config = ComponentConfig()
    config.name = "test"
    attrs = Struct()
    attrs["mlmodel_name"] = "model"
    config.attributes.CopyFrom(attrs)

    dependencies = {
        MLModel.get_resource_name("model"): MagicMock(),
    }

    classifier = Classifier.new(config, dependencies)

    assert classifier.sampling_method == "random"
