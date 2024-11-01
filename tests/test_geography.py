import unittest
import numpy as np
from unittest.mock import patch
from geography import GeoCoordinates, GeoHeading, GeoPose
from primitives import Point3d


class TestGeoCoordinates(unittest.TestCase):

    @patch('geography.ecef2enu')
    @patch('geography.geodetic2ecef')
    def test_to_enu(self, mock_geodetic2ecef, mock_ecef2enu):
        mock_geodetic2ecef.return_value = (4166107.29, 860479.90, 4736942.82)
        mock_ecef2enu.return_value = (8224.34, 2964.17, -14.98)
        ref_coords = GeoCoordinates(48.239, 11.559, 486)
        coords = GeoCoordinates(48.265602, 11.669937, 477)

        location = coords.to_enu(ref=ref_coords, ignore_earth_curvature=False)

        self.assertAlmostEqual(location.x, 8224.34)
        self.assertAlmostEqual(location.y, 2964.17)
        self.assertAlmostEqual(location.z, -14.98)


class TestGeoHeading(unittest.TestCase):

    @patch('geography.declination')
    def test_magnetic_to_true(self, mock_declination):
        mock_declination.return_value = 13
        coords = 37.779259, -122.419329, 20

        true_heading = GeoHeading.magnetic_to_true(magnetic_heading=350, coords=GeoCoordinates(*coords))

        self.assertEqual(true_heading, 3)
        mock_declination.assert_called_once_with(*coords)

    def test_to_enu(self):
        self.assertEqual(GeoHeading.to_enu(0), np.pi / 2)
        self.assertEqual(GeoHeading.to_enu(90), 0)
        self.assertEqual(GeoHeading.to_enu(135), 7 * np.pi / 4)


class TestGeoPose(unittest.TestCase):

    @patch('geography.GeoHeading.to_enu')
    @patch('geography.GeoCoordinates.to_enu')
    def test_to_enu(self, mock_coords_to_enu, mock_heading_to_enu):
        expected_point = Point3d(100.0, 200.0, 300.0)
        mock_coords_to_enu.return_value = expected_point
        expected_heading = np.pi / 4
        mock_heading_to_enu.return_value = expected_heading
        coords = GeoCoordinates(40, 11, 100)
        ref_coords = GeoCoordinates(41, 12, 122)
        geo = GeoPose(coords, 90.0)

        enu = geo.to_enu(ref_coords)

        self.assertEqual(enu.position, expected_point)
        self.assertEqual(enu.heading, expected_heading)
        mock_coords_to_enu.assert_called_once_with(ref_coords)
        mock_heading_to_enu.assert_called_once_with(90.0)


if __name__ == "__main__":
    unittest.main()
