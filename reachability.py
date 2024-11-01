import os
import numpy as np
import rasterio
import geojson
from geopy.distance import geodesic
from scipy.optimize import brentq
from dubins import DubinsPlanner
from path import Line, Path, Trajectory
from primitives import Direction2d, Vector2d, Point3d, Pose2d, Pose3d, is_close
from log import logger
from multiprocessing import Pool
from enum import Enum
from geography import GeoCoordinates


class GridType(Enum):

    PLANE_CENTERED = 1
    RUNWAY_CENTERED = 2


class Reachability:

    @classmethod
    def glide_radius(cls, glide_straight, plane_height: float) -> float:
        return plane_height * glide_straight

    @classmethod
    def glide_ring_center(cls, glide_straight, plane_position: Point3d, wind: Vector2d) -> Point3d:
        offset = cls.glide_ring_offset(glide_straight, plane_position.z, wind)
        return Point3d(plane_position.x + offset.x, plane_position.y + offset.y, plane_position.z)

    @classmethod
    def glide_ring_offset(cls, sink_rate_straight, plane_height: float, wind: Vector2d) -> Vector2d:
        t = plane_height / sink_rate_straight
        return Vector2d(wind.x * t, wind.y * t)

    @classmethod
    def rendezvous_paths(
            cls, turn_radius: float, glide_straight: float, sink_rate_straight: float, plane_pose: Pose2d,
            runway_pose: Pose2d, available_height: float, wind: Vector2d, include_runway_path: bool = True
    ) -> list[Path]:
        v = wind.invert()
        v_mag = v.magnitude()
        v_norm = v.normalize()
        v_heading = v.heading()

        def calculate_discontinuities(c1, c2):
            p = -2 * (v_norm.x * (c1.x - c2.x) + v_norm.y * (c1.y - c2.y))
            q = (c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2 - 4 * turn_radius ** 2
            discriminant = (p / 2) ** 2 - q

            if discriminant < 0:
                return []

            sqrt_discriminant = np.sqrt(discriminant)
            solution_1 = (-p / 2) + sqrt_discriminant
            solution_2 = (-p / 2) - sqrt_discriminant

            return [solution_1, solution_2] if solution_1 != solution_2 else [solution_1]

        def paths(d):
            target = Pose2d(runway_pose.position.move(v_heading, d), runway_pose.heading)
            return DubinsPlanner.paths(plane_pose, target, turn_radius, True)

        def g(d):
            t_plane = paths(d)[0].length() / (glide_straight * sink_rate_straight)
            t_target = abs(d) / v_mag
            return t_plane - t_target

        RSL_centers = [
            plane_pose.rotation_center(Direction2d.RIGHT, turn_radius),
            runway_pose.rotation_center(Direction2d.LEFT, turn_radius)
        ]

        LSR_centers = [
            plane_pose.rotation_center(Direction2d.LEFT, turn_radius),
            runway_pose.rotation_center(Direction2d.RIGHT, turn_radius)
        ]

        discontinuities = (
            calculate_discontinuities(RSL_centers[0], RSL_centers[1]) +
            calculate_discontinuities(LSR_centers[0], LSR_centers[1])
        )

        min_value = 0
        max_value = v_mag * available_height / sink_rate_straight
        valid_discontinuities = [d for d in discontinuities if min_value < d < max_value]
        intervals = [min_value] + sorted(set(valid_discontinuities)) + [max_value]
        roots = []

        for i in range(len(intervals) - 1):
            d1, d2 = intervals[i], intervals[i + 1]

            d1_ = d1 + 0.0001
            d2_ = d2 - 0.0001

            if g(d1_) * g(d2_) > 0:
                continue

            try:
                root = brentq(g, d1_, d2_, xtol=1e-3)
                roots.append(root)
            except ValueError:
                pass

        if not roots:
            return []  # No rendezvous with dubins paths

        d_opt = min(roots)
        target_end_position = runway_pose.position.move(v_heading, d_opt)
        result = paths(d_opt)

        if not include_runway_path:
            return result

        runway_path = Line(Pose2d(runway_pose.position, v_heading), Pose2d(target_end_position, v_heading))
        result += [runway_path]
        return result

    @classmethod
    def plane_grid(
            cls, turn_radius: float, glide_straight: float, glide_curve: float, sink_rate_straight: float,
            plane_pose: Pose3d, runway_heading: float, wind: Vector2d = Vector2d(0, 0), res: int = 100
    ) -> np.ndarray:
        return cls._grid(
            turn_radius, glide_straight, glide_curve, sink_rate_straight, plane_pose.xy(),
            runway_heading, plane_pose.position.z, GridType.PLANE_CENTERED, wind, res
        )[0]

    @staticmethod
    def runway_grid(
            cls, turn_radius: float, glide_straight: float, glide_curve: float, sink_rate_straight: float, runway_pose: Pose2d,
            plane_heading: float, plane_height: float, wind: Vector2d = Vector2d(0, 0), res: int = 100
    ) -> np.ndarray:
        return cls._grid(
            turn_radius, glide_straight, glide_curve, sink_rate_straight, runway_pose,
            plane_heading,  plane_height, GridType.RUNWAY_CENTERED, wind, res
        )[0]

    @classmethod
    def _grid(
            cls, turn_radius: float, glide_straight: float, glide_curve: float, sink_rate_straight: float, center_pose: Pose2d,
            grid_heading: float, available_height: float, grid_type: GridType, wind: Vector2d, res: int
    ) -> (np.ndarray, float):
        max_range = cls.glide_radius(glide_straight, available_height)
        positive_values = np.linspace(0, max_range, res)
        negative_values = -1 * positive_values[1:]
        values = np.concatenate((negative_values[::-1], positive_values))
        glide_ring_offset = cls.glide_ring_offset(sink_rate_straight, available_height, wind)

        num_processes = os.cpu_count()
        array_length = len(values)
        num_chunks = num_processes
        chunk_size = array_length // num_chunks
        chunks = [(i * chunk_size, i * chunk_size + chunk_size if i < num_chunks - 1 else array_length) for i in range(num_chunks)]

        args = [(
            values, max_range, center_pose, grid_heading, turn_radius, glide_straight, glide_curve,
            sink_rate_straight, available_height, grid_type, chunk, glide_ring_offset, wind
        ) for chunk in chunks]

        logger().debug(f'Evaluating {len(values)**2} points for reachability in {num_processes} processes.')

        with Pool(num_processes) as pool:
            results = pool.map(cls._calculate_reachability, args)

        grid = np.vstack(results)
        glide_count = sum(1 for y in values for x in values if x ** 2 + y ** 2 <= max_range ** 2)
        dubins_count = np.sum(grid)
        fraction = dubins_count/glide_count

        logger().info(f'Points within glide ring: {glide_count}. Reachable via Dubins path: {dubins_count}. Relative: {fraction}')

        return grid, fraction

    @classmethod
    def _calculate_reachability(cls, args) -> (np.ndarray, float, float):
        (values, max_range, center_pose, grid_heading, turn_radius, glide_straight, glide_curve,
         sink_rate_straight, available_height, grid_type, chunk, glide_ring_offset, wind) = args

        min_row, max_row = chunk

        chunk_length = max_row - min_row
        grid_shape = (max_row - min_row, len(values))
        grid = np.zeros(grid_shape, dtype=np.uint8)

        for y_index in range(0, chunk_length):
            y = values[min_row + y_index]

            for x_index, x in np.ndenumerate(values):
                if x ** 2 + y ** 2 > max_range ** 2:
                    continue

                start_pose = Pose2d.at(
                    x + center_pose.position.x - glide_ring_offset.x,
                    y + center_pose.position.y - glide_ring_offset.y,
                    grid_heading
                ) if grid_type == GridType.RUNWAY_CENTERED else center_pose

                end_pose = Pose2d.at(
                    x + center_pose.position.x + glide_ring_offset.x,
                    y + center_pose.position.y + glide_ring_offset.y,
                    grid_heading
                ) if grid_type == GridType.PLANE_CENTERED else center_pose

                if end_pose == start_pose:
                    continue

                if wind.magnitude() > 0:
                    paths = cls.rendezvous_paths(
                        turn_radius, glide_straight, sink_rate_straight, start_pose,
                        end_pose, available_height, wind, include_runway_path=False
                    )
                else:
                    paths = DubinsPlanner.paths(start_pose, end_pose, turn_radius, shortest_only=True)

                if not paths:
                    continue

                height_diff = min([Trajectory(path, glide_straight, glide_curve).height_diff() for path in paths])
                grid[y_index, x_index] = 1 if is_close(height_diff, available_height) or height_diff < available_height else 0

        return grid

    @classmethod
    def terrain_glide_ring(
            cls, glide_straight, coordinates: GeoCoordinates, srtm_file: str,
            output_file: str, radial_step: int = 50, polar_res: int = 180
    ):
        dataset = rasterio.open(srtm_file)
        terrain = dataset.read(1)

        angles = np.linspace(0, 2 * np.pi, polar_res)
        glide_ring_coords = []

        for i, theta in enumerate(angles):
            d = 0
            max_lat = coordinates.lat
            max_lon = coordinates.lon

            while True:
                new_point = geodesic(meters=d).destination((coordinates.lat, coordinates.lon), np.degrees(theta))
                y_raster, x_raster = dataset.index(new_point.longitude, new_point.latitude)

                if 0 <= x_raster < dataset.width and 0 <= y_raster < dataset.height:
                    terrain_height = terrain[y_raster, x_raster]
                else:
                    raise RuntimeError('Should not happen, increase size of elevation dataset')

                remaining_altitude = coordinates.alt - d / glide_straight

                if remaining_altitude <= terrain_height:
                    break

                max_lat = new_point.latitude
                max_lon = new_point.longitude

                d += radial_step

            glide_ring_coords.append((max_lon, max_lat))

        glide_ring_geojson = geojson.Feature(
            geometry=geojson.Polygon(glide_ring_coords),
            properties={"name": "Glide Ring"}
        )

        with open(output_file, "w") as f:
            geojson.dump(geojson.FeatureCollection([glide_ring_geojson]), f)
