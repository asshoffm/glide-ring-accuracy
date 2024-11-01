import unittest
import numpy as np
from primitives import Direction2d, Point2d, Pose2d
from path import Line, Arc, Path


class TestLine(unittest.TestCase):

    def test_init(self):
        start_pose = Pose2d.at(0, 0, np.pi / 4)
        end_pose = Pose2d.at(3, 3, np.pi / 4)
        Line(start_pose, end_pose)

        with self.assertRaises(ValueError):
            start_pose = Pose2d.at(0, 0, np.pi / 4)
            end_pose = Pose2d.at(3, 3, 0)
            Line(start_pose, end_pose)

        with self.assertRaises(ValueError):
            Line(start_pose, start_pose)

    def test_length(self):
        start_pose = Pose2d.at(0, 0, np.pi / 4)
        end_pose = Pose2d.at(4, 4, np.pi / 4)
        line = Line(start_pose, end_pose)

        self.assertAlmostEqual(line.length(), 5.6569, places=4)

    def test_points(self):
        start_pose = Pose2d.at(0, 0, 0)
        end_pose = Pose2d.at(4, 0, 0)
        line = Line(start_pose, end_pose)

        expected_points = [start_pose.position, end_pose.position]
        self.assertEqual(line.points(6), expected_points)

        expected_points = [start_pose.position, Point2d(2, 0), end_pose.position]
        self.assertEqual(line.points(3), expected_points)

        expected_points = [start_pose.position, Point2d(1, 0), Point2d(2, 0), Point2d(3, 0), end_pose.position]
        self.assertEqual(line.points(1), expected_points)

    def test_equality(self):
        start_pose = Pose2d.at(0, 0, 5 * np.pi / 4)
        end_pose = Pose2d.at(-3, -3, 5 * np.pi / 4)
        line_1 = Line(start_pose, end_pose)

        start_pose = Pose2d.at(0, 0, 5 * np.pi / 4)
        end_pose = Pose2d.at(-3, -3, 5 * np.pi / 4)
        line_2 = Line(start_pose, end_pose)

        start_pose = Pose2d.at(0, 0, 5 * np.pi / 4)
        end_pose = Pose2d.at(-4, -4, 5 * np.pi / 4)
        line_3 = Line(start_pose, end_pose)

        self.assertEqual(line_1, line_2)
        self.assertNotEqual(line_1, line_3)
        self.assertNotEqual(line_1, "Not a Line object")


class TestArc(unittest.TestCase):

    def setUp(self):
        self.start_pose = Pose2d.at(0, 0, 0)
        self.end_pose_left = Pose2d.at(0, 1, np.pi)
        self.end_pose_right = Pose2d.at(-0.5, -0.5, np.pi / 2)

    def test_init(self):
        Arc(self.start_pose, self.end_pose_left)
        Arc(self.start_pose, self.end_pose_right)

        with self.assertRaises(ValueError):
            end_pose = Pose2d.at(0, 1, 0)
            Arc(self.start_pose, end_pose)

    def test_direction(self):
        arc_left = Arc(self.start_pose, self.end_pose_left)
        arc_right = Arc(self.start_pose, self.end_pose_right)

        self.assertEqual(arc_left.direction(), Direction2d.LEFT)
        self.assertEqual(arc_right.direction(), Direction2d.RIGHT)

    def test_length(self):
        arc_left = Arc(self.start_pose, self.end_pose_left)
        arc_right = Arc(self.start_pose, self.end_pose_right)

        self.assertAlmostEqual(arc_left.length(), np.pi / 2)
        self.assertAlmostEqual(arc_right.length(), 3 * np.pi / 4)

        with self.assertRaises(ValueError):
            Arc(self.start_pose, self.start_pose)

    def test_radius(self):
        arc_left = Arc(self.start_pose, self.end_pose_left)
        arc_right = Arc(self.start_pose, self.end_pose_right)

        self.assertAlmostEqual(arc_left.radius(), 0.5)
        self.assertAlmostEqual(arc_right.radius(), 0.5)

    def test_points(self):
        arc_left = Arc(self.start_pose, self.end_pose_left)

        expected_points = [self.start_pose.position, self.end_pose_left.position]
        actual_points = arc_left.points(2)

        self.assertEqual(actual_points, expected_points)

        expected_points = [self.start_pose.position, Point2d(0.5, 0.5), self.end_pose_left.position]
        actual_points = arc_left.points(1)

        self.assertEqual(actual_points, expected_points)

    def test_equality(self):
        arc_1 = Arc(self.start_pose, self.end_pose_left)
        arc_2 = Arc(self.start_pose, self.end_pose_left)
        arc_3 = Arc(self.start_pose, self.end_pose_right)

        self.assertEqual(arc_1, arc_2)
        self.assertNotEqual(arc_1, arc_3)
        self.assertNotEqual(arc_1, "Not an Arc")


class TestPath(unittest.TestCase):

    def test_points(self):
        expected_points = [Point2d(3, 3), Point2d(5, 5), Point2d(7, 7), Point2d(7, 0)]
        dummy_pose = Pose2d.at(0,0,0)
        segment_1 = MockSegment(start_pose=dummy_pose, end_pose=dummy_pose, length=10, points=expected_points[:2])
        segment_2 = MockSegment(start_pose=dummy_pose, end_pose=dummy_pose, length=5, points=expected_points[-2:])

        path = Path('', [segment_1, segment_2])

        self.assertEqual(path.points(), expected_points)

    def test_start_pose(self):
        dummy_pose = Pose2d.at(0,0,0)
        start_pose = Pose2d.at(10,2, np.pi)
        segment_1 = MockSegment(start_pose=start_pose, end_pose=dummy_pose, length=0, points=[])
        segment_2 = MockSegment(start_pose=dummy_pose, end_pose=dummy_pose, length=0, points=[])

        path = Path('', [segment_1, segment_2])

        self.assertEqual(path.start_pose, start_pose)

    def test_end_pose(self):
        dummy_pose = Pose2d.at(0,0,0)
        end_pose = Pose2d.at(-7, -9, 2 * np.pi)
        segment_1 = MockSegment(start_pose=dummy_pose, end_pose=dummy_pose, length=0, points=[])
        segment_2 = MockSegment(start_pose=dummy_pose, end_pose=end_pose, length=0, points=[])

        path = Path('', [segment_1, segment_2])

        self.assertEqual(path.end_pose, end_pose)

    def test_length(self):
        dummy_pose = Pose2d.at(0,0,0)
        segment_1 = MockSegment(start_pose=dummy_pose, end_pose=dummy_pose, length=10, points=[])
        segment_2 = MockSegment(start_pose=dummy_pose, end_pose=dummy_pose, length=5, points=[])

        path = Path('', [segment_1, segment_2])

        self.assertEqual(path.length(), 15)


class MockSegment:

    def __init__(self, start_pose: Pose2d, end_pose: Pose2d, length: float, points: list[Point2d]):
        self.start_pose = start_pose
        self.end_pose = end_pose
        self._length = length
        self._points = points

    def length(self) -> float:
        return self._length

    def points(self, step: float) -> list[Point2d]:
        return self._points


if __name__ == "__main__":
    unittest.main()
