import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Tuple
from matplotlib.patches import Circle, Polygon
from matplotlib.ticker import FuncFormatter
from path import Path
from primitives import Point2d


def plot_grid(grid: np.ndarray, glide_radius: float, center_position: Point2d):
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('Reachability')

    circle = Circle((center_position.x, center_position.y), glide_radius, edgecolor='red', fill=False, linewidth=2)
    ax.add_patch(circle)

    grid_length = len(grid)
    cell_size = glide_radius / (grid_length // 2)

    contours, hierarchy = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    offset = -np.array([grid_length // 2, grid_length // 2])
    contours = [((np.vstack((contour[:, 0], contour[0, 0])) + offset) * cell_size) + np.array([center_position.x, center_position.y]) for contour in contours]

    if hierarchy is None:
        return

    hierarchy = hierarchy[0]

    for i, contour in enumerate(contours):
        level = 0
        parent = hierarchy[i, 3]

        while parent != -1:
            level += 1
            parent = hierarchy[parent, 3]

        if level % 2 == 0:
            polygon = Polygon(contour, edgecolor='#0000ff', facecolor='#dfe9f6', linewidth=2, zorder=level, alpha=0.8)
        else:
            polygon = Polygon(contour, edgecolor='#0000ff', facecolor='white', linewidth=1, zorder=level)

        ax.add_patch(polygon)

    formatter = FuncFormatter(lambda x, _: f'{x/1000:.1f}' if x != 0 else '0')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    glide_radius_x_min, glide_radius_x_max = center_position.x - glide_radius, center_position.x + glide_radius
    glide_radius_y_min, glide_radius_y_max = center_position.y - glide_radius, center_position.y + glide_radius
    lim_min, lim_max = min(glide_radius_x_min, glide_radius_y_min, 0), max(glide_radius_x_max, glide_radius_y_max, 0)

    margin = 0.1 * glide_radius
    ax.set_xlim(lim_min-margin, lim_max+margin)
    ax.set_ylim(lim_min-margin, lim_max+margin)

    ax.set(xlabel='E (km)', ylabel='N (km)')
    ax.set_aspect('equal')
    ax.set_box_aspect(1)

    ax.set_xticks([glide_radius_x_min, 0, glide_radius_x_max])
    ax.set_yticks([glide_radius_y_min, 0, glide_radius_y_max])

    ax.xaxis.set_label_coords(0.75, -0.01)
    ax.yaxis.set_label_coords(-0.01, 0.75)
    ax.tick_params(axis='x', pad=2)
    ax.tick_params(axis='y', pad=2)

    plt.grid(which='both', color='#d4d4d4', linestyle='-', linewidth=0.2)
    plt.show()


def plot_paths(paths_with_air_speeds: list[Tuple[Path, float]], labels: list[str]):
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('Rendezvous')
    x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')
    lines, x_coordinates, y_coordinates = [], [], []

    for i, path in enumerate(paths_with_air_speeds):
        path_points = path[0].points(step=path[1]/100)
        end_heading = path[0].end_pose.heading
        x_coordinates.append([point.x for point in path_points])
        y_coordinates.append([point.y for point in path_points])
        x_min, x_max = min(x_min, min(x_coordinates[i])), max(x_max, max(x_coordinates[i]))
        y_min, y_max = min(y_min, min(y_coordinates[i])), max(y_max, max(y_coordinates[i]))
        lines.append(ax.plot([], [], label=labels[i], color=f'C{i}', alpha=0.8)[0])

    margin = 1.4
    center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
    max_range = max(x_max - x_min, y_max - y_min) * margin

    ax.set_xlim(center_x - max_range / 2, center_x + max_range / 2)
    ax.set_ylim(center_y - max_range / 2, center_y + max_range / 2)
    ax.set(xlabel='E (km)', ylabel='N (km)', aspect='equal')
    ax.legend()

    lengths = [len(series) for series in x_coordinates]
    max_frames = max(lengths)

    def update(frame):
        for j in range(len(lines)):
            if frame < lengths[j]:
                lines[j].set_xdata(x_coordinates[j][:frame])
                lines[j].set_ydata(y_coordinates[j][:frame])
            else:
                lines[j].set_xdata(x_coordinates[j])
                lines[j].set_ydata(y_coordinates[j])

        return []

    _ = animation.FuncAnimation(fig, update, frames=max_frames, blit=False, interval=10, repeat=False)
    plt.show()
