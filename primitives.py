import numpy as np
from enum import Enum


class Direction2d(Enum):

    STRAIGHT = 'S'
    LEFT = 'L'
    RIGHT = 'R'


class Vector2d:

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def magnitude(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def normalize(self) -> "Vector2d":
        magnitude = self.magnitude()
        return Vector2d(self.x / magnitude, self.y / magnitude) if magnitude != 0 else Vector2d(0, 0)

    def invert(self) -> "Vector2d":
        return Vector2d(-self.x, -self.y)

    def heading(self) -> float:
        return np.atan2(self.y, self.x) % (2 * np.pi)

    def __str__(self):
        return f'x: {self.x}; y: {self.y}'

    def __eq__(self, other):
        if not isinstance(other, Vector2d):
            return False
        return is_close(self.x, other.x) and is_close(self.y, other.y)


class Point2d:

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def distance_to(self, point: "Point2d") -> float:
        return np.sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2)

    def vector_to(self, point: "Point2d") -> Vector2d:
        return Vector2d(point.x - self.x, point.y - self.y)

    def move(self, heading: float, distance: float) -> "Point2d":
        return Point2d(self.x + distance * np.cos(heading), self.y + distance * np.sin(heading))

    def __str__(self):
        return f'x: {self.x}; y: {self.y}'

    def __eq__(self, other):
        if not isinstance(other, Point2d):
            return False
        return is_close(self.x, other.x) and is_close(self.y, other.y)


class Pose2d:

    def __init__(self, position: Point2d, heading: float):
        self.position = position
        self.heading = _normalize_heading(heading)

    @classmethod
    def at(cls, x: float, y: float, heading: float) -> "Pose2d":
        position = Point2d(x, y)
        return cls(position, heading)

    def relative_position(self, position: Point2d) -> Direction2d:
        cross_product = np.cos(self.heading) * (position.y - self.position.y) - np.sin(self.heading) * (position.x - self.position.x)

        if np.isclose(cross_product, 0):
            return Direction2d.STRAIGHT
        elif cross_product > 0:
            return Direction2d.LEFT
        else:
            return Direction2d.RIGHT

    def rotation_center(self, direction: Direction2d, radius: float) -> Point2d:
        perpendicular_heading = self.heading + np.pi / 2 if direction == Direction2d.LEFT else self.heading - np.pi / 2
        return self.position.move(perpendicular_heading, radius)

    def __str__(self):
        return f'{self.position}; heading: {self.heading}'

    def __eq__(self, other):
        if not isinstance(other, Pose2d):
            return False
        return self.position == other.position and np.isclose(self.heading, other.heading)


class Point3d:

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def distance_to(self, point: "Point3d") -> float:
        return np.sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2 + + (self.z - point.z) ** 2)

    def xy(self) -> Point2d:
        return Point2d(self.x, self.y)

    def __eq__(self, other):
        if not isinstance(other, Point3d):
            return False
        return is_close(self.x, other.x) and is_close(self.y, other.y) and is_close(self.z, other.z)


class Pose3d:

    def __init__(self, position: Point3d, heading: float):
        self.position = position
        self.heading = _normalize_heading(heading)

    @classmethod
    def at(cls, x: float, y: float, z: float, heading: float) -> "Pose3d":
        position = Point3d(x, y, z)
        return cls(position, heading)

    def xy(self) -> Pose2d:
        return Pose2d(self.position.xy(), self.heading)

    def __eq__(self, other):
        if not isinstance(other, Pose3d):
            return False
        return self.position == other.position and np.isclose(self.heading, other.heading)


def _normalize_heading(heading: float) -> float:
    return (heading + 2 * np.pi) % (2 * np.pi)


def is_close(a, b) -> bool:
    return np.isclose(a, b, atol=1e-1)
