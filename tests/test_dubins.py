import unittest
import numpy as np
from primitives import Pose2d, Point2d
from path import Arc, Line
from dubins import DubinsPlanner, DubinsWord


class TestDubinsPlanner(unittest.TestCase):

    def test_paths_shortest(self):
        start_pose = Pose2d(position=Point2d(x=-1, y=0), heading=np.pi / 2)
        end_pose = Pose2d(position=Point2d(x=5, y=0), heading=3 * np.pi / 2)

        shortest_paths = DubinsPlanner.paths(start_pose, end_pose, 1)
        actual_words = {path.name for path in shortest_paths}
        self.assertEqual(actual_words, {DubinsWord.RSR.name})

        rsr_path = shortest_paths[0]

        expected_segments = [
            Arc(start_pose, Pose2d(position=Point2d(x=0, y=1), heading=0)),
            Line(Pose2d(position=Point2d(x=0, y=1), heading=0), Pose2d(position=Point2d(x=4, y=1), heading=0)),
            Arc(Pose2d(position=Point2d(x=4, y=1), heading=0), end_pose)
        ]

        self.assertEqual(rsr_path.segments, expected_segments)

    def test_paths_all(self):
        start_pose = Pose2d(position=Point2d(x=-1, y=0), heading=np.pi / 2)
        end_pose = Pose2d(position=Point2d(x=5, y=0), heading=3 * np.pi / 2)

        all_paths = DubinsPlanner.paths(start_pose, end_pose, 1, shortest_only=False)
        expected_words = {word.name for word in DubinsWord.csc().union({DubinsWord.RLR})}
        actual_words = {path.name for path in all_paths}
        self.assertEqual(actual_words, expected_words)

        rlr_path = [path for path in all_paths if path.name == DubinsWord.RLR.name][0]

        expected_segments = [
            Arc(start_pose, Pose2d(position=Point2d(x=1, y=0), heading=3 * np.pi / 2)),
            Arc(Pose2d(position=Point2d(x=1, y=0), heading=3 * np.pi / 2), Pose2d(position=Point2d(x=3, y=0), heading=np.pi / 2)),
            Arc(Pose2d(position=Point2d(x=3, y=0), heading=np.pi / 2), end_pose)
        ]

        self.assertEqual(rlr_path.segments, expected_segments)

    def test_paths_null_segments(self):
        start_pose = Pose2d.at(0, 0, 0)
        end_pose = Pose2d.at(5, 0, 0)
        self.assertEqual(DubinsPlanner.paths(start_pose, end_pose, 1)[0].segments, [Line(start_pose, end_pose)])

        start_pose = Pose2d.at(0, 0, 0)
        end_pose = Pose2d.at(0, 2, np.pi)
        self.assertEqual(DubinsPlanner.paths(start_pose, end_pose, 1)[0].segments, [Arc(start_pose, end_pose)])

        end_pose = start_pose
        self.assertEqual(DubinsPlanner.paths(start_pose, end_pose, 1)[0].segments, [])


if __name__ == "__main__":
    unittest.main()
