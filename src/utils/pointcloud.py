from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class PointCloud:
    """Simple point cloud container replacing open3d.geometry.PointCloud"""

    points: np.ndarray  # (N, 3) XYZ coordinates
    colors: Optional[np.ndarray] = None  # (N, 3) RGB values in [0, 1]
    normals: Optional[np.ndarray] = None  # (N, 3) normal vectors

    def has_colors(self) -> bool:
        """Check if point cloud has color data"""
        return self.colors is not None

    def has_normals(self) -> bool:
        """Check if point cloud has normal vectors"""
        return self.normals is not None


def parse_pcd_bytes(data: bytes) -> PointCloud:
    """
    Parse PCD format bytes into PointCloud object.

    Args:
        data: Raw PCD file bytes

    Returns:
        PointCloud object with points, colors, and/or normals

    Raises:
        ValueError: If PCD format is invalid or unsupported
    """
    return parse_pcd_bytes_manually(data)


def new_from_array(input: np.ndarray) -> PointCloud:
    """
    Create PointCloud from points only.

    Args:
        input: (N, 3) (N, 6) or (N, 9) array of XYZ coordinates
    Returns
        PointCloud object with points only
    """
    if len(input.shape) != 2 or input.shape[1] not in [3, 6, 9]:
        raise ValueError("Input array must be of shape (N, 3), (N, 6), or (N, 9)")

    points = input[:, 0:3].astype(np.float32)
    colors = input[:, 3:6].astype(np.float32) if input.shape[1] >= 6 else None
    normals = input[:, 6:9].astype(np.float32) if input.shape[1] == 9 else None

    return PointCloud(points=points, colors=colors, normals=normals)


def pcd_to_array(input: PointCloud) -> "np.ndarray":
    points_np = np.asarray(input.points)

    if input.has_colors():
        colors_np = np.asarray(input.colors)
        if input.has_normals():
            normals_np = np.asarray(input.normals)
            return np.hstack((points_np, colors_np, normals_np))
        else:
            return np.hstack((points_np, colors_np))
    else:
        return points_np


def parse_pcd_bytes_manually(data: bytes) -> PointCloud:
    """
    Parse PCD format bytes into PointCloud object.

    Args:
        data: Raw PCD file bytes

    Returns:
        PointCloud object with points, colors, and/or normals

    Raises:
        ValueError: If PCD format is invalid or unsupported
    """
    import struct

    # Decode header
    lines = data.split(b"\n")
    header = {}
    data_start_idx = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith(b"#"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        key = parts[0].decode("ascii")
        value = b" ".join(parts[1:]).decode("ascii")

        if key == "DATA":
            header[key] = value
            data_start_idx = i + 1
            break
        else:
            header[key] = value

    # Parse header fields
    fields = header.get("FIELDS", "").split()
    sizes = list(map(int, header.get("SIZE", "").split()))
    types = header.get("TYPE", "").split()
    points_count = int(header.get("POINTS", "0"))
    data_format = header.get("DATA", "ascii")

    # Determine which fields are present
    has_x = "x" in fields
    has_y = "y" in fields
    has_z = "z" in fields
    has_rgb = "rgb" in fields or "rgba" in fields
    has_normal_x = "normal_x" in fields
    has_normal_y = "normal_y" in fields
    has_normal_z = "normal_z" in fields

    if not (has_x and has_y and has_z):
        raise ValueError("PCD must contain x, y, z fields")

    # Get field indices
    x_idx = fields.index("x")
    y_idx = fields.index("y")
    z_idx = fields.index("z")
    rgb_idx = (
        fields.index("rgb")
        if "rgb" in fields
        else (fields.index("rgba") if "rgba" in fields else -1)
    )

    normal_idx = -1
    if has_normal_x and has_normal_y and has_normal_z:
        normal_idx = fields.index("normal_x")

    # Parse data
    points = np.zeros((points_count, 3), dtype=np.float32)
    colors = np.zeros((points_count, 3), dtype=np.float32) if has_rgb else None
    normals = np.zeros((points_count, 3), dtype=np.float32) if has_normal_x else None

    if data_format == "ascii":
        # Parse ASCII data
        point_idx = 0
        for i in range(data_start_idx, len(lines)):
            line = lines[i].strip()
            if not line:
                continue

            values = line.split()
            if len(values) < len(fields):
                continue

            # Extract XYZ
            points[point_idx] = [
                float(values[x_idx]),
                float(values[y_idx]),
                float(values[z_idx]),
            ]

            # Extract RGB if present
            if has_rgb and colors is not None:
                rgb_value = int(float(values[rgb_idx]))
                # RGB is packed as 32-bit integer: 0xRRGGBB or 0xAARRGGBB
                r = ((rgb_value >> 16) & 0xFF) / 255.0
                g = ((rgb_value >> 8) & 0xFF) / 255.0
                b = (rgb_value & 0xFF) / 255.0
                colors[point_idx] = [r, g, b]

            # Extract normals if present
            if has_normal_x and normals is not None:
                normals[point_idx] = [
                    float(values[normal_idx]),
                    float(values[normal_idx + 1]),
                    float(values[normal_idx + 2]),
                ]

            point_idx += 1
            if point_idx >= points_count:
                break

    elif data_format == "binary":
        # Calculate total bytes per point
        bytes_per_point = sum(sizes)

        # Find where binary data starts (after header)
        header_end = data.find(b"DATA binary\n") + len(b"DATA binary\n")
        binary_data = data[header_end:]

        # Parse binary data
        offset = 0
        for point_idx in range(points_count):
            point_data = binary_data[offset : offset + bytes_per_point]

            # Unpack based on field order
            field_offset = 0
            for field_idx, (field, size, ftype) in enumerate(zip(fields, sizes, types)):
                field_bytes = point_data[field_offset : field_offset + size]

                if ftype == "F":  # Float
                    value = struct.unpack("<f", field_bytes)[0]
                elif ftype == "U":  # Unsigned int
                    if size == 4:
                        value = struct.unpack("<I", field_bytes)[0]
                    elif size == 2:
                        value = struct.unpack("<H", field_bytes)[0]
                    elif size == 1:
                        value = struct.unpack("<B", field_bytes)[0]
                elif ftype == "I":  # Signed int
                    if size == 4:
                        value = struct.unpack("<i", field_bytes)[0]
                    elif size == 2:
                        value = struct.unpack("<h", field_bytes)[0]
                    elif size == 1:
                        value = struct.unpack("<b", field_bytes)[0]

                # Store value in appropriate array
                if field == "x":
                    points[point_idx, 0] = value
                elif field == "y":
                    points[point_idx, 1] = value
                elif field == "z":
                    points[point_idx, 2] = value
                elif field in ["rgb", "rgba"] and colors is not None:
                    rgb_value = int(value)
                    r = ((rgb_value >> 16) & 0xFF) / 255.0
                    g = ((rgb_value >> 8) & 0xFF) / 255.0
                    b = (rgb_value & 0xFF) / 255.0
                    colors[point_idx] = [r, g, b]
                elif field == "normal_x" and normals is not None:
                    normals[point_idx, 0] = value
                elif field == "normal_y" and normals is not None:
                    normals[point_idx, 1] = value
                elif field == "normal_z" and normals is not None:
                    normals[point_idx, 2] = value

                field_offset += size

            offset += bytes_per_point

    else:
        raise ValueError(f"Unsupported DATA format: {data_format}")

    return PointCloud(points=points, colors=colors, normals=normals)
