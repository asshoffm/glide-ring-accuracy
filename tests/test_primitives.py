import unittest
import numpy as np
from primitives import Direction2d, Vector2d, Point2d, Pose2d, Point3d, Pose3d


class TestVector2d(unittest.TestCase):

    def test_magnitude(self):
        vector = Vector2d(3, 4)
        self.assertEqual(vector.magnitude(), 5)

        vector = Vector2d(0, 0)
        self.assertEqual(vector.magnitude(), 0)

    def test_normalize(self):
        vector = Vector2d(3, 4)
        normalized = vector.normalize()
        self.assertAlmostEqual(normalized.x, 0.6)
        self.assertAlmostEqual(normalized.y, 0.8)

        zero_vector = Vector2d(0, 0)
        normalized_zero = zero_vector.normalize()
        self.assertEqual(normalized_zero, Vector2d(0, 0))

    def test_heading(self):
        vector = Vector2d(1, 0)
        self.assertAlmostEqual(vector.heading(), 0)

        vector = Vector2d(0, 1)
        self.assertAlmostEqual(vector.heading(), np.pi / 2)

        vector = Vector2d(-1, 0)
        self.assertAlmostEqual(vector.heading(), np.pi)

        vector = Vector2d(0, -1)
        self.assertAlmostEqual(vector.heading(), 3 * np.pi / 2)

    def test_equality(self):
        vector_1 = Vector2d(1, 2)
        vector_2 = Vector2d(1, 2)
        vector_3 = Vector2d(2, 3)

        self.assertEqual(vector_1, vector_2)
        self.assertNotEqual(vector_1, vector_3)
        self.assertNotEqual(vector_1, "Not a Vector object")


class TestPoint2d(unittest.TestCase):

    def test_distance_to(self):
        point1 = Point2d(0, 0)
        point2 = Point2d(3, 4)
        self.assertEqual(point1.distance_to(point2), 5)

        point3 = Point2d(1, 1)
        self.assertAlmostEqual(point1.distance_to(point3), np.sqrt(2))

        self.assertEqual(point1.distance_to(point1), 0)

    def test_vector_to(self):
        point1 = Point2d(1, 2)
        point2 = Point2d(4, 6)
        vector = point1.vector_to(point2)
        self.assertEqual(vector, Vector2d(3, 4))

        point3 = Point2d(0, 0)
        vector = point1.vector_to(point3)
        self.assertEqual(vector, Vector2d(-1, -2))

    def test_move(self):
        point = Point2d(0, 0)
        moved_point = point.move(np.pi / 4, np.sqrt(2))
        self.assertEqual(moved_point, Point2d(1, 1))

        moved_point = point.move(0, 5)
        self.assertEqual(moved_point, Point2d(5, 0))

        moved_point = point.move(np.pi / 2, 3)
        self.assertEqual(moved_point, Point2d(0, 3))

    def test_equality(self):
        point_1 = Point2d(1, 2)
        point_2 = Point2d(1, 2)
        point_3 = Point2d(2, 3)

        self.assertEqual(point_1, point_2)
        self.assertNotEqual(point_1, point_3)
        self.assertNotEqual(point_1, "Not a Point object")


class TestPose2d(unittest.TestCase):

    def test_init(self):
        pose = Pose2d.at(1, 2, np.pi / 4)
        self.assertEqual(pose.position, Point2d(1, 2))
        self.assertAlmostEqual(pose.heading, np.pi / 4)

        pose = Pose2d.at(0, 0, -np.pi / 2)
        self.assertAlmostEqual(pose.heading, 3 * np.pi / 2)

    def test_relative_position(self):
        pose = Pose2d.at(0, 0, 0)

        left_point = Point2d(0, 1)
        right_point = Point2d(0, -1)
        straight_point = Point2d(1, 0)

        self.assertEqual(pose.relative_position(left_point), Direction2d.LEFT)
        self.assertEqual(pose.relative_position(right_point), Direction2d.RIGHT)
        self.assertEqual(pose.relative_position(straight_point), Direction2d.STRAIGHT)

    def test_rotation_center(self):
        pose = Pose2d.at(0, 0, 0)

        left_center = pose.rotation_center(Direction2d.LEFT, 5)
        right_center = pose.rotation_center(Direction2d.RIGHT, 5)

        self.assertEqual(left_center, Point2d(0, 5))
        self.assertEqual(right_center, Point2d(0, -5))

        pose = Pose2d.at(0, 0, np.pi / 4)
        left_center = pose.rotation_center(Direction2d.LEFT, 5)
        right_center = pose.rotation_center(Direction2d.RIGHT, 5)

        self.assertEqual(left_center, Point2d(-3.5355, 3.5355))
        self.assertEqual(right_center, Point2d(3.5355, -3.5355))

    def test_equality(self):
        pose_1 = Pose2d.at(1, 2, np.pi / 4)
        pose_2 = Pose2d.at(1, 2, np.pi / 4)
        pose_3 = Pose2d.at(3, 4, np.pi / 4)
        pose_4 = Pose2d.at(1, 2, np.pi / 2)

        self.assertEqual(pose_1, pose_2)
        self.assertNotEqual(pose_1, pose_3)
        self.assertNotEqual(pose_1, pose_4)
        self.assertNotEqual(pose_1, "Not a Pose object")


class TestPoint3d(unittest.TestCase):

    def test_distance_to(self):
        point_1 = Point3d(0.0, 0.0, 0.0)
        point_2 = Point3d(3.0, 4.0, 0.0)  # Distance should be 5.0 (3-4-5 triangle)
        self.assertAlmostEqual(point_1.distance_to(point_2), 5.0)

    def test_equality(self):
        point_1 = Point3d(1.0, 2.0, 3.0)
        point_2 = Point3d(1.0, 2.0, 3.0)
        point_3 = Point3d(1.0, 2.0, 3.2)
        self.assertTrue(point_1 == point_2)
        self.assertFalse(point_1 == point_3)
        self.assertFalse(point_1 == "not a point")


class TestPose3d(unittest.TestCase):

    def test_init(self):
        pose = Pose3d.at(1, 2, 0, np.pi / 4)
        self.assertEqual(pose.position, Point3d(1, 2, 0))
        self.assertAlmostEqual(pose.heading, np.pi / 4)

        pose = Pose3d.at(0, 0, 0, -np.pi / 2)
        self.assertAlmostEqual(pose.heading, 3 * np.pi / 2)

    def test_equality(self):
        pose_1 = Pose3d(Point3d(1.0, 2.0, 3.0), np.pi / 4)
        pose_2 = Pose3d(Point3d(1.0, 2.0, 3.0), np.pi / 4)
        pose_3 = Pose3d(Point3d(1.0, 2.0, 3.0), np.pi / 2)
        self.assertTrue(pose_1 == pose_2)
        self.assertFalse(pose_1 == pose_3)
        self.assertFalse(pose_1 == "not a pose")


if __name__ == "__main__":
    unittest.main()
