import sys
from typing import ClassVar, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import open3d as o3d
from typing_extensions import Self

from src.utils.pointcloud import (
    PointCloud,
    parse_pcd_bytes,
    new_from_array,
    pcd_to_array,
)
from viam.components.camera import Camera
from viam.services.mlmodel import MLModel, Metadata
from viam.media.video import ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.vision import Vision, CaptureAllResult
from viam.utils import ValueTypes, struct_to_dict


class Classifier(Vision, EasyResource):
    # To enable debug-level logging, either run viam-server with the --debug option,
    # or configure your resource/machine to display debug logs.
    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "vision"), "pointcloud-vision")

    DEFAULT_VOXEL_SIZE = 0.05  # Default voxel size for voxel downsampling (if chosen)
    mlmodel: MLModel
    default_camera: str
    labels: Optional[List[str]]

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """This method creates a new instance of this Vision service.
        The default implementation sets the name from the `config` parameter.

        Args:
            config (ComponentConfig): The configuration for this resource
            dependencies (Mapping[ResourceName, ResourceBase]): The dependencies (both required and optional)

        Returns:
            Self: The resource
        """
        self = super().new(config, dependencies)
        self.reconfigure(config, dependencies)
        return self

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """This method allows you to validate the configuration object received from the machine,
        as well as to return any required dependencies or optional dependencies based on that `config`.

        Args:
            config (ComponentConfig): The configuration for this resource

        Returns:
            Tuple[Sequence[str], Sequence[str]]: A tuple where the
                first element is a list of required dependencies and the
                second element is a list of optional dependencies
        """
        attrs = struct_to_dict(config.attributes)
        deps = []

        if mlmodel := attrs.get("mlmodel_name", None):
            deps.append(mlmodel)
        else:
            raise ValueError("mlmodel_name field is required in configuration")

        if camera := attrs.get("camera_name", None):
            deps.append(camera)

        return deps, []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        attrs = struct_to_dict(config.attributes)

        def getCamera(name: str) -> Camera:
            return cast(Camera, dependencies.get(Camera.get_resource_name(name)))

        self.getCamera = getCamera
        self.default_camera = str(attrs.get("camera_name"))
        self.sampling_method = str(attrs.get("sampling_method", "random"))

        if self.sampling_method not in ["random", "voxel", "fps"]:
            self.logger.warning(
                f"Invalid sampling_method '{self.sampling_method}', using 'random'"
            )
            self.sampling_method = "random"

        try:
            self.mlmodel = cast(
                MLModel,
                dependencies.get(
                    MLModel.get_resource_name(str(attrs.get("mlmodel_name")))
                ),
            )
        except Exception as err:
            self.logger.error(
                f"Unable to configure pointcloud classifier vision service: {err}"
            )
            raise err

    def _parse_metadata(self, metadata: Metadata):
        """
        Parse metadata to extract model requirements.

        Args:
            metadata: MLModel metadata object

        Returns:
            Tuple of (input_name, target_points, target_features, has_batch_dim,
                     output_name, class_names)

        Raises:
            ValueError: If shape is dynamic or unexpected format
        """
        # Get first input tensor info
        input_info = metadata.input_info[0]
        input_name = input_info.name
        shape = list(input_info.shape)

        # Parse shape: [N, F] or [1, N, F]
        if len(shape) == 2:
            has_batch_dim = False
            target_points, target_features = shape
        elif len(shape) == 3 and (shape[0] == 1 or shape[0] == -1):
            has_batch_dim = True
            target_points, target_features = shape[1], shape[2]
        else:
            raise ValueError(f"Unexpected input shape: {shape}")

        # Check for dynamic shapes
        if target_points == -1 or target_features == -1:
            raise ValueError(
                f"Model has dynamic input shape {shape}; cannot determine requirements"
            )

        # Get output info
        output_info = metadata.output_info[0]
        output_name = output_info.name
        output_extras = struct_to_dict(output_info.extra)

        # Try to load class labels from extra
        class_names = None
        if label_path := output_extras.get("labels"):
            with open(label_path, "r") as f:
                class_names = f.read().splitlines()

        return (
            input_name,
            int(target_points),
            int(target_features),
            has_batch_dim,
            output_name,
            class_names,
        )

    def _logits_to_classifications(
        self, logits: "np.ndarray", class_names: Optional[List[str]], count: int
    ) -> List[Classification]:
        """
        Convert model logits to Classification objects.

        Args:
            logits: Raw model output (1D array)
            class_names: List of class names (or None to use indices)
            count: Number of top classifications to return

        Returns:
            List[Classification] sorted by confidence descending
        """
        # Apply softmax: probabilities = exp(logits) / sum(exp(logits))
        # Subtract max for numerical stability
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # Sort by probability descending
        indices = np.argsort(probs)[::-1]

        # Take top count items
        top_indices = indices[:count]

        # Create Classification objects
        if class_names is None:
            class_names = [str(i) for i in range(len(logits))]

        return [
            Classification(class_name=class_names[i], confidence=float(probs[i]))
            for i in top_indices
        ]

    def _normalize_point_cloud(self, points: "np.ndarray") -> "np.ndarray":
        """
        Normalize point cloud to unit sphere.

        Args:
            points: Nx3 array of XYZ coordinates

        Returns:
            Normalized points centered at origin, scaled to unit sphere
        """
        # Center at origin
        centered = points - points.mean(axis=0)

        # Scale to unit sphere by max distance from origin
        distances = np.sqrt((centered**2).sum(axis=1))
        max_dist = distances.max()
        if max_dist > 0:
            normalized = centered / max_dist
        else:
            normalized = centered

        return normalized

    def _sample_point_cloud(
        self, points: "np.ndarray", target_count: int, method: str
    ) -> "np.ndarray":
        """
        Sample point cloud to target number of points.
        Args:
            points: Nx3 (or NxF) array of point features
            target_count: Desired number of points
            method: Sampling method ("random", "voxel", or "fps")
        Returns:
            Sampled points with shape [target_count, F]
        """
        # If we have the correct number of points, return as-is
        current_count = points.shape[0]
        if current_count == target_count:
            return points

        # If they pick an invalid method, default to random
        if method not in ["random", "voxel", "fps"]:
            self.logger.warning(
                f"Sampling method '{method}' not yet implemented, using random"
            )
            method = "random"

        cloud = new_from_array(points)
        if method == "random":
            ratio = target_count / current_count
            if ratio > 1.0:
                # Upsampling here.
                indices = np.random.choice(current_count, target_count, replace=True)
                sampled_points = points[indices]
                return sampled_points

            downsampled = cloud.orig.random_down_sample(ratio)
            if np.asarray(downsampled.points).shape[0] != target_count:
                self.logger.warning(
                    f"Random sampling did not yield exact target count {target_count}, "
                    f"got {np.asarray(downsampled.points).shape[0]}"
                )
        elif method == "voxel":
            downsampled = cloud.orig.voxel_down_sample(self.DEFAULT_VOXEL_SIZE)
        else:
            downsampled = cloud.orig.farthest_point_down_sample(target_count)

        return pcd_to_array(downsampled)

    def _parse_point_cloud(self, pcd_bytes: bytes, mimetype: str) -> PointCloud:
        """
        Parse point cloud bytes into PointCloud object.

        Args:
            pcd_bytes: Raw point cloud bytes from camera
            mimetype: MIME type of the point cloud data
                     TODO: Use mimetype to support multiple formats in future

        Returns:
            PointCloud object

        Raises:
            RuntimeError: If parsing fails
        """
        try:
            pcd = parse_pcd_bytes(pcd_bytes)

            if len(pcd.points) == 0:
                raise RuntimeError("Parsed point cloud is empty")

            return pcd

        except Exception as e:
            raise RuntimeError(f"Failed to parse point cloud data: {e}")

    def _preprocess_point_cloud(
        self,
        cloud: PointCloud,
        target_points: int,
        target_features: int,
        sampling_method: str,
    ) -> "np.ndarray":
        """
        Preprocess point cloud for model inference.

        Args:
            cloud: PointCloud object
            target_points: Number of points required by model (N)
            target_features: Number of features per point (3, 6, or 9)
            sampling_method: "random", "voxel", or "fps"

        Returns:
            numpy array of shape [target_points, target_features]

        Raises:
            ValueError: If required features are missing from cloud
        """

        # Extract XYZ (always present)
        points = cloud.points

        # Check and extract additional features
        features = [points]

        # Extract RGB if needed (target_features >= 6)
        if target_features >= 6:
            if not cloud.has_colors():
                raise ValueError(
                    f"Model requires RGB data (shape [N,{target_features}]) "
                    "but point cloud has no colors"
                )
            colors = cloud.colors
            features.append(colors)

        # Extract normals if needed (target_features >= 9)
        if target_features >= 9:
            if not cloud.has_normals():
                raise ValueError(
                    f"Model requires normals (shape [N,{target_features}]) "
                    "but point cloud has none"
                )
            normals = cloud.normals
            features.append(normals)

        # Concatenate features
        combined = np.concatenate(features, axis=1)

        # Resample to target_points
        sampled = self._sample_point_cloud(combined, target_points, sampling_method)

        # Normalize XYZ coordinates only (first 3 columns)
        xyz_normalized = self._normalize_point_cloud(sampled[:, :3])

        # Combine normalized XYZ with other features
        if target_features == 3:
            result = xyz_normalized
        else:
            # Keep RGB/normals as-is, replace XYZ with normalized
            result = np.concatenate([xyz_normalized, sampled[:, 3:]], axis=1)

        return result

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        result = CaptureAllResult()

        if camera_name == "" and self.default_camera == "":
            raise ValueError(
                "No camera name provided and no default camera name configured"
            )
        elif camera_name == "":
            camera_name = self.default_camera
        camera = self.getCamera(camera_name)

        if return_image:
            images, _ = await camera.get_images()

            if len(images) > 0:
                result.image = images[0]

        if return_classifications:
            # Get count from extra or use default
            count = 5  # Default count
            if extra and "count" in extra:
                count = int(extra["count"])

            classifications = await self.get_classifications_from_camera(
                camera_name, count, extra=extra, timeout=timeout
            )
            result.classifications = classifications

        return result

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        self.logger.error("`get_detections_from_camera` is not implemented")
        raise NotImplementedError()

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        self.logger.error("`get_detections` is not implemented")
        raise NotImplementedError()

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        """Get classifications from point cloud captured by camera."""
        # Get metadata
        metadata = await self.mlmodel.metadata()
        (
            input_name,
            target_points,
            target_features,
            has_batch_dim,
            output_name,
            class_names,
        ) = self._parse_metadata(metadata)

        # Get camera
        if camera_name == "" and self.default_camera == "":
            raise ValueError(
                "No camera name provided and no default camera name configured"
            )
        elif camera_name == "":
            camera_name = self.default_camera
        camera = self.getCamera(camera_name)

        # Get point cloud
        pcd_bytes, mimetype = await camera.get_point_cloud(timeout=timeout)

        # Parse with Open3D
        cloud = self._parse_point_cloud(pcd_bytes, mimetype)

        # Preprocess
        sampling_method = getattr(self, "sampling_method", "random")
        preprocessed = self._preprocess_point_cloud(
            cloud, target_points, target_features, sampling_method
        )

        # Add batch dimension if needed
        if has_batch_dim:
            preprocessed = preprocessed[np.newaxis, ...]

        # Inference
        input_tensors = {input_name: preprocessed}
        output_tensors = await self.mlmodel.infer(input_tensors, timeout=timeout)

        # Extract output
        logits = output_tensors[output_name]
        self.logger.debug(f"Raw output shape: {logits.shape}, dtype: {logits.dtype}")

        # Handle output shape correctly by inspecting actual dimensions
        # Output could be [num_classes] or [1, num_classes] or [batch, num_classes]
        if logits.ndim == 2:
            # Has batch dimension: [batch, num_classes]
            if logits.shape[0] != 1:
                raise ValueError(
                    f"Expected single sample output, got batch size {logits.shape[0]}"
                )
            logits = logits[0]  # Remove batch dimension -> [num_classes]
        elif logits.ndim == 1:
            # Already 1D: [num_classes] - use as-is
            pass
        elif logits.ndim == 0:
            # Scalar output - this shouldn't happen for classification
            raise ValueError(
                "Model returned scalar output, expected array of class logits"
            )
        else:
            raise ValueError(
                f"Unexpected output shape {logits.shape}, expected 1D or 2D array"
            )

        self.logger.debug(f"Final logits shape: {logits.shape}")

        # Convert to classifications
        return self._logits_to_classifications(logits, class_names, count)

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        self.logger.error("Point cloud classification requires camera input")
        raise NotImplementedError(
            "get_classifications() not supported for point clouds; "
            "use get_classifications_from_camera() instead"
        )

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[PointCloudObject]:
        self.logger.error("`get_object_point_clouds` is not implemented")
        raise NotImplementedError()

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Vision.Properties:
        properties = Vision.Properties(
            classifications_supported=True,
            detections_supported=False,
            object_point_clouds_supported=False,
        )
        return properties

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        self.logger.error("`do_command` is not implemented")
        raise NotImplementedError()
