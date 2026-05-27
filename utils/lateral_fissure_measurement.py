import math
from collections.abc import Sequence
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize


Point = Tuple[float, float]


def parse_pixel_spacing(value: Optional[str]) -> Optional[Tuple[float, float]]:
    if value is None or value == "":
        return None
    parts = [part.strip() for part in str(value).replace("x", ",").split(",") if part.strip()]
    if len(parts) == 1:
        spacing = float(parts[0])
        return spacing, spacing
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])
    raise ValueError("pixel spacing should be one value or row,col values")


def _largest_component(mask: np.ndarray):
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary.astype(bool), 0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas) + 1)
    return labels == largest_label, int(areas.max()), int(num_labels - 1)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return vector / norm


def _length(p1: Point, p2: Point, pixel_spacing: Optional[Tuple[float, float]] = None) -> float:
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    if pixel_spacing is None:
        return math.hypot(dx, dy)
    row_spacing, col_spacing = pixel_spacing
    return math.hypot(dx * col_spacing, dy * row_spacing)


def _trace_inside(mask: np.ndarray, center: np.ndarray, direction: np.ndarray, step: float = 0.5) -> Point:
    height, width = mask.shape
    point = center.astype(np.float32).copy()
    last_inside = point.copy()
    for _ in range(int(max(height, width) * 4)):
        point = point + direction * step
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if x < 0 or x >= width or y < 0 or y >= height or not mask[y, x]:
            break
        last_inside = point.copy()
    return float(last_inside[0]), float(last_inside[1])


def _measurement_label(value_px: float, value_mm: Optional[float], name: str) -> str:
    if value_mm is None:
        return f"{name}={value_px:.1f}px"
    return f"{name}={value_mm:.2f}mm ({value_px:.1f}px)"


def measure_lateral_fissure(
    mask: np.ndarray,
    pixel_spacing: Optional[Tuple[float, float]] = None,
    min_area: int = 8,
) -> Dict[str, object]:
    component, area, component_count = _largest_component(mask)
    empty_result = {
        "status": "empty",
        "component_count": component_count,
        "area_px": area,
        "depth_px": 0.0,
        "width_px": 0.0,
        "mean_width_px": 0.0,
        "depth_mm": None,
        "width_mm": None,
        "mean_width_mm": None,
        "orientation_deg": 0.0,
        "depth_line": None,
        "width_line": None,
        "component_mask": component,
    }
    if area < min_area:
        return empty_result

    yx = np.column_stack(np.where(component))
    xy = yx[:, ::-1].astype(np.float32)
    centroid = xy.mean(axis=0)
    centered = xy - centroid
    if len(xy) < 2:
        return empty_result

    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = _unit(eigenvectors[:, int(np.argmax(eigenvalues))].astype(np.float32))
    if major[0] < 0:
        major = -major
    minor = np.array([-major[1], major[0]], dtype=np.float32)

    projections = centered @ major
    start = centroid + major * float(projections.min())
    end = centroid + major * float(projections.max())
    depth_px = _length(tuple(start), tuple(end))
    depth_mm = _length(tuple(start), tuple(end), pixel_spacing) if pixel_spacing else None

    distance_map = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
    skeleton = skeletonize(component)
    if skeleton.any():
        weighted = distance_map * skeleton.astype(np.float32)
        width_center_y, width_center_x = np.unravel_index(int(np.argmax(weighted)), weighted.shape)
        skeleton_widths = distance_map[skeleton] * 2.0
        mean_width_px = float(np.mean(skeleton_widths)) if skeleton_widths.size else 0.0
    else:
        width_center_y, width_center_x = np.unravel_index(int(np.argmax(distance_map)), distance_map.shape)
        mean_width_px = float(distance_map[width_center_y, width_center_x] * 2.0)

    width_center = np.array([float(width_center_x), float(width_center_y)], dtype=np.float32)
    width_a = _trace_inside(component, width_center, minor)
    width_b = _trace_inside(component, width_center, -minor)
    width_px = _length(width_a, width_b)
    width_mm = _length(width_a, width_b, pixel_spacing) if pixel_spacing else None
    if pixel_spacing:
        scale = (pixel_spacing[0] + pixel_spacing[1]) / 2.0
        mean_width_mm = mean_width_px * scale
    else:
        mean_width_mm = None

    orientation_deg = math.degrees(math.atan2(float(major[1]), float(major[0])))
    return {
        "status": "ok",
        "component_count": component_count,
        "area_px": area,
        "depth_px": float(depth_px),
        "width_px": float(width_px),
        "mean_width_px": float(mean_width_px),
        "depth_mm": float(depth_mm) if depth_mm is not None else None,
        "width_mm": float(width_mm) if width_mm is not None else None,
        "mean_width_mm": float(mean_width_mm) if mean_width_mm is not None else None,
        "orientation_deg": float(orientation_deg),
        "depth_line": (tuple(start), tuple(end)),
        "width_line": (width_a, width_b),
        "component_mask": component,
    }


def measurement_to_row(measurement: Dict[str, object], prefix: str = "fissure") -> Dict[str, object]:
    row = {
        f"{prefix}_measurement_status": measurement["status"],
        f"{prefix}_component_count": measurement["component_count"],
        f"{prefix}_area_px": measurement["area_px"],
        f"{prefix}_depth_px": measurement["depth_px"],
        f"{prefix}_width_px": measurement["width_px"],
        f"{prefix}_mean_width_px": measurement["mean_width_px"],
        f"{prefix}_orientation_deg": measurement["orientation_deg"],
    }
    if measurement.get("depth_mm") is not None:
        row[f"{prefix}_depth_mm"] = measurement["depth_mm"]
        row[f"{prefix}_width_mm"] = measurement["width_mm"]
        row[f"{prefix}_mean_width_mm"] = measurement["mean_width_mm"]
    return row


def _as_int_point(point: Point) -> Tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def _draw_dashed_line(
    image: np.ndarray,
    p1: Point,
    p2: Point,
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 8,
    gap_length: int = 6,
):
    p1_array = np.array(p1, dtype=np.float32)
    p2_array = np.array(p2, dtype=np.float32)
    distance = float(np.linalg.norm(p2_array - p1_array))
    if distance <= 1e-6:
        return
    direction = (p2_array - p1_array) / distance
    cursor = 0.0
    while cursor < distance:
        segment_start = p1_array + direction * cursor
        segment_end = p1_array + direction * min(cursor + dash_length, distance)
        cv2.line(image, _as_int_point(tuple(segment_start)), _as_int_point(tuple(segment_end)), color, thickness, cv2.LINE_AA)
        cursor += dash_length + gap_length


def _put_label(image: np.ndarray, text: str, point: Point, color: Tuple[int, int, int]):
    x, y = _as_int_point(point)
    height, width = image.shape[:2]
    x = max(4, min(width - 4, x))
    y = max(18, min(height - 8, y))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(
        image,
        (x - 3, y - text_h - baseline - 3),
        (min(width - 1, x + text_w + 3), min(height - 1, y + baseline + 3)),
        (0, 0, 0),
        -1,
    )
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def annotate_lateral_fissure_measurement(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    measurement: Optional[Dict[str, object]] = None,
    pixel_spacing: Optional[Tuple[float, float]] = None,
    alpha: float = 0.25,
) -> np.ndarray:
    measurement = measurement or measure_lateral_fissure(mask, pixel_spacing=pixel_spacing)
    annotated = image_bgr.copy()

    component_mask = measurement.get("component_mask")
    if isinstance(component_mask, np.ndarray) and component_mask.any():
        color_layer = np.zeros_like(annotated, dtype=np.uint8)
        color_layer[component_mask] = (0, 0, 255)
        blended = cv2.addWeighted(annotated, 1.0, color_layer, alpha, 0)
        annotated[component_mask] = blended[component_mask]

    if measurement.get("status") != "ok":
        _put_label(annotated, "no measurable fissure", (8, 22), (0, 255, 255))
        return annotated

    depth_line = measurement.get("depth_line")
    width_line = measurement.get("width_line")
    if isinstance(depth_line, Sequence):
        p1, p2 = depth_line
        _draw_dashed_line(annotated, p1, p2, (255, 255, 0), thickness=2)
        cv2.circle(annotated, _as_int_point(p1), 3, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(annotated, _as_int_point(p2), 3, (255, 255, 0), -1, cv2.LINE_AA)
        midpoint = ((p1[0] + p2[0]) / 2.0 + 6.0, (p1[1] + p2[1]) / 2.0 - 6.0)
        _put_label(
            annotated,
            _measurement_label(float(measurement["depth_px"]), measurement.get("depth_mm"), "depth"),
            midpoint,
            (255, 255, 0),
        )
    if isinstance(width_line, Sequence):
        p1, p2 = width_line
        _draw_dashed_line(annotated, p1, p2, (0, 255, 255), thickness=2)
        cv2.circle(annotated, _as_int_point(p1), 3, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(annotated, _as_int_point(p2), 3, (0, 255, 255), -1, cv2.LINE_AA)
        midpoint = ((p1[0] + p2[0]) / 2.0 + 6.0, (p1[1] + p2[1]) / 2.0 + 16.0)
        _put_label(
            annotated,
            _measurement_label(float(measurement["width_px"]), measurement.get("width_mm"), "width"),
            midpoint,
            (0, 255, 255),
        )
    return annotated
