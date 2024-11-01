import numpy as np
from pymap3d import geodetic2ecef, ecef2enu
from geomag import declination
from primitives import Pose3d, Point3d


class GeoCoordinates:

    def __init__(self, lat: float, lon: float, alt: float):
        self.lat = lat
        self.lon = lon
        self.alt = alt

    def to_enu(self, ref: "GeoCoordinates", ignore_earth_curvature: bool = True) -> Point3d:
        x, y, z = geodetic2ecef(self.lat, self.lon, self.alt)
        e, n, u = ecef2enu(x, y, z, ref.lat, ref.lon, ref.alt)
        return Point3d(e, n, u if not ignore_earth_curvature else self.alt - ref.alt)


class GeoHeading:

    @classmethod
    def magnetic_to_true(cls, magnetic_heading: float, coords: GeoCoordinates) -> float:
        magnetic_declination = declination(coords.lat, coords.lon, coords.alt)
        true_heading = (magnetic_heading + magnetic_declination) % 360
        return true_heading

    @classmethod
    def to_enu(cls, heading: float):
        return np.radians((90 - heading) % 360)


class GeoPose:

    def __init__(self, coords: GeoCoordinates, heading: float):
        self.coords = coords
        self.heading = heading

    def to_enu(self, ref: GeoCoordinates) -> Pose3d:
        return Pose3d(self.coords.to_enu(ref), GeoHeading.to_enu(self.heading))
