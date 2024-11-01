import numpy as np
from typing import Tuple
from typing import Protocol
from primitives import Direction2d, Point2d, Pose2d, Point3d


class Segment(Protocol):

    start_pose: Pose2d
    end_pose: Pose2d

    def length(self) -> float: ...

    def points(self, step: float) -> list[Point2d]: ...


class Line:

    def __init__(self, start_pose: Pose2d, end_pose: Pose2d):
        self.start_pose = start_pose
        self.end_pose = end_pose

        if not self._is_valid():
            raise ValueError('Cannot create line from provided configuration.')

    def _is_valid(self) -> bool:
        direction = self.start_pose.relative_position(self.end_pose.position)
        return (direction == Direction2d.STRAIGHT and self.start_pose.heading == self.end_pose.heading and
                self.start_pose.position != self.end_pose.position)

    def length(self) -> float:
        return self.start_pose.position.distance_to(self.end_pose.position)

    def points(self, step: float) -> list[Point2d]:
        num_points = int(np.ceil(self.length() / step)) + 1
        x_sequence = np.linspace(self.start_pose.position.x, self.end_pose.position.x, num=num_points)
        y_sequence = np.linspace(self.start_pose.position.y, self.end_pose.position.y, num=num_points)
        return [Point2d(x, y) for x, y in zip(x_sequence, y_sequence)]

    def __eq__(self, other):
        if not isinstance(other, Line):
            return False
        return self.start_pose == other.start_pose and self.end_pose == other.end_pose


class Arc:

    def __init__(self, start_pose: Pose2d, end_pose: Pose2d):
        self.start_pose = start_pose
        self.end_pose = end_pose

        if not self._is_valid():
            raise ValueError('Cannot create arc from provided configuration.')

    def _is_valid(self) -> bool:
        direction = self.direction()

        if direction == Direction2d.STRAIGHT:
            return False

        radius = self.radius()
        center_1 = self.start_pose.rotation_center(direction, radius)
        center_2 = self.end_pose.rotation_center(direction, radius)

        return center_1 == center_2

    def direction(self) -> Direction2d:
        return self.start_pose.relative_position(self.end_pose.position)

    def length(self) -> float:
        start_angle, end_angle = self._angles()
        radius = self.radius()
        return abs(end_angle - start_angle) * radius

    def radius(self) -> float:
        chord_length = self.start_pose.position.distance_to(self.end_pose.position)
        delta_heading = abs(self.end_pose.heading - self.start_pose.heading)
        return 0 if delta_heading == 0 else chord_length / (2 * np.sin(delta_heading / 2))

    def points(self, step: float) -> list[Point2d]:
        start_angle, end_angle = self._angles()
        direction = self.direction()
        radius = self.radius()
        num_points = int(np.ceil(self.length() / step)) + 1
        angles = np.linspace(start_angle, end_angle, num_points)
        center = self.start_pose.rotation_center(direction, radius)
        x_sequence = center.x + radius * np.cos(angles)
        y_sequence = center.y + radius * np.sin(angles)
        return [Point2d(x, y) for x, y in zip(x_sequence, y_sequence)]

    def _angles(self) -> Tuple[float, float]:
        start_angle = self.start_pose.heading
        end_angle = self.end_pose.heading
        direction = self.direction()

        if direction == Direction2d.RIGHT:
            start_angle += np.pi / 2
            end_angle += np.pi / 2
            end_angle = end_angle - 2 * np.pi if start_angle < end_angle else end_angle
        else:
            start_angle -= np.pi / 2
            end_angle -= np.pi / 2
            end_angle = end_angle + 2 * np.pi if end_angle < start_angle else end_angle

        return start_angle, end_angle

    def __eq__(self, other):
        if not isinstance(other, Arc):
            return False
        return self.start_pose == other.start_pose and self.end_pose == other.end_pose


class Path:

    def __init__(self, name: str, segments: list[Segment]):
        self.name = name
        self.segments = segments

    @property
    def start_pose(self) -> Pose2d:
        return self.segments[0].start_pose

    @property
    def end_pose(self) -> Pose2d:
        return self.segments[-1].end_pose

    def points(self, step: float = 0.01) -> list[Point2d]:
        return [point for segment in self.segments for point in segment.points(step)]

    def length(self) -> float:
        return sum([segment.length() for segment in self.segments])

    def __str__(self):
        return str(self.name)


class Trajectory:

    def __init__(self, path: Path, glide_straight: float, glide_curve: float):
        self.path = path
        self.glide_straight = glide_straight
        self.glide_curve = glide_curve

    def height_diff(self) -> float:
        return sum(self.segment_height_diffs())

    def segment_height_diffs(self) -> [float]:
        return [segment.length() / self._glide(segment) for segment in self.path.segments]

    def _glide(self, segment) -> float:
        return self.glide_straight if isinstance(segment, Line) else self.glide_curve
