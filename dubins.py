import numpy as np
from enum import Enum
from log import logger
from primitives import Direction2d, Point2d, Vector2d, Pose2d, is_close
from path import Segment, Line, Arc, Path


class DubinsWord(Enum):

    LSL = (Direction2d.LEFT, Direction2d.STRAIGHT, Direction2d.LEFT)
    RSR = (Direction2d.RIGHT, Direction2d.STRAIGHT, Direction2d.RIGHT)
    LSR = (Direction2d.LEFT, Direction2d.STRAIGHT, Direction2d.RIGHT)
    RSL = (Direction2d.RIGHT, Direction2d.STRAIGHT, Direction2d.LEFT)
    LRL = (Direction2d.LEFT, Direction2d.RIGHT, Direction2d.LEFT)
    RLR = (Direction2d.RIGHT, Direction2d.LEFT, Direction2d.RIGHT)

    @classmethod
    def csc(cls) -> set["DubinsWord"]:
        return {cls.LSL, cls.RSR, cls.LSR, cls.RSL}

    @classmethod
    def ccc(cls) -> set["DubinsWord"]:
        return {cls.LRL, cls.RLR}


class DubinsPlanner:

    @classmethod
    def paths(cls, start_pose: Pose2d, end_pose: Pose2d, radius: float, shortest_only: bool = True) -> list[Path]:
        paths = []
        shortest_paths = []
        min_length = float('inf')

        for word in DubinsWord:
            try:
                path = cls._path(start_pose, end_pose, radius, word)
                paths.append(path)
                path_length = path.length()

                if path_length < min_length:
                    shortest_paths = [path]
                    min_length = path_length
                elif is_close(path_length, min_length):
                    shortest_paths.append(path)

            except ValueError as error:
                logger().debug(f'{word}-Path could not be created: {error}')

        return shortest_paths if shortest_only else paths

    @classmethod
    def _path(cls, start_pose: Pose2d, end_pose: Pose2d, radius: float, word: DubinsWord) -> Path:
        center_1 = start_pose.rotation_center(word.value[0], radius)
        center_2 = end_pose.rotation_center(word.value[2], radius)
        segments: [Segment]

        if center_1 == center_2:
            segments = [cls._arc(start_pose, end_pose)]
        elif word in DubinsWord.csc():
            segments = cls._csc(start_pose, end_pose, radius, center_1, center_2, word)
        else:
            segments = cls._ccc(start_pose, end_pose, radius, center_1, center_2, word)

        return Path(word.name, [x for x in segments if x is not None])

    @classmethod
    def _csc(cls, start_pose: Pose2d, end_pose: Pose2d, radius: float, center_1: Point2d, center_2: Point2d, word: DubinsWord) -> [Segment]:
        if word == DubinsWord.LSL:
            sign1, sign2 = 1, -1
        elif word == DubinsWord.LSR:
            sign1, sign2 = -1, -1
        elif word == DubinsWord.RSL:
            sign1, sign2 = -1, 1
        else:
            sign1, sign2 = 1, 1

        center_vector = center_1.vector_to(center_2).normalize()
        distance = center_1.distance_to(center_2)
        c = (radius - sign1 * radius) / distance

        if abs(c) > 1.0:
            raise ValueError('No valid tangent for this configuration.')

        c_2 = np.sqrt(1.0 - c ** 2)
        n = Vector2d(center_vector.x * c - sign2 * center_vector.y * c_2, center_vector.y * c + sign2 * center_vector.x * c_2)
        inter_point_1 = Point2d(center_1.x + radius * n.x, center_1.y + radius * n.y)
        inter_point_2 = Point2d(center_2.x + sign1 * radius * n.x, center_2.y + sign1 * radius * n.y)
        inter_heading = n.heading() - sign2 * np.pi / 2
        inter_pose_1 = Pose2d(inter_point_1, inter_heading)
        inter_pose_2 = Pose2d(inter_point_2, inter_heading)

        logger().debug(f'Calculated tangent for {word}-Path: ({inter_pose_1}); ({inter_pose_2})')

        return [
            cls._arc(start_pose, inter_pose_1),
            cls._line(inter_pose_1, inter_pose_2),
            cls._arc(inter_pose_2, end_pose)
        ]

    @classmethod
    def _ccc(cls, start_pose: Pose2d, end_pose: Pose2d, radius: float, center_1: Point2d, center_2: Point2d, word: DubinsWord) -> [Segment]:
        center_vector = center_1.vector_to(center_2)
        distance = center_vector.magnitude()

        if distance > 4 * radius:
            raise ValueError(f'Centers are too far apart for a CCC path ({word}).')

        sign = 1 if word.value[0] == Direction2d.LEFT else -1
        center_1_angle = np.acos((distance ** 2) / (4 * radius * distance))
        heading_c1_c3 = center_vector.heading() + sign * center_1_angle
        inter_point_1 = center_1.move(heading_c1_c3, radius)
        inter_pose_1 = Pose2d(inter_point_1, heading_c1_c3 + sign * np.pi / 2)

        center_3 = inter_point_1.move(heading_c1_c3, radius)
        heading_c3_c2 = center_3.vector_to(center_2).heading()
        inter_point_2 = center_3.move(heading_c3_c2, radius)
        inter_pose_2 = Pose2d(inter_point_2, heading_c3_c2 - sign * np.pi / 2)

        logger().debug(f'Calculated connecting arc for {word}-Path: ({inter_pose_1}); ({inter_pose_2})')

        return [
            cls._arc(start_pose, inter_pose_1),
            cls._arc(inter_pose_1, inter_pose_2),
            cls._arc(inter_pose_2, end_pose)
        ]

    @classmethod
    def _line(cls, start_pose: Pose2d, end_pose: Pose2d) -> Line | None:
        return Line(start_pose, end_pose) if start_pose != end_pose else None

    @classmethod
    def _arc(cls, start_pose: Pose2d, end_pose: Pose2d) -> Arc | None:
        return Arc(start_pose, end_pose) if start_pose != end_pose else None
