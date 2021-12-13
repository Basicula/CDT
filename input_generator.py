import math
import random
from typing import List

def generate_polygon_points(number_of_points : int, center_x : float, center_y : float):
    radius = 100
    radius_noise = radius / 4

    points = []
    angle = 0
    angle_step = 2 * math.pi / number_of_points
    for _ in range(number_of_points):
        r = radius + 2 * random.random() * radius_noise - radius_noise
        x = center_x + r * math.cos(angle)
        y = center_y + r * math.sin(angle)
        points.append([x, y])
        angle += angle_step

    return points

def random_points(number_of_points : int, min_x : float, min_y : float, max_x : float, max_y : float):
    points = []
    for _ in range(number_of_points):
        x = random.random() * (max_x - min_x) + min_x
        y = random.random() * (max_y - min_y) + min_y
        points.append([x, y])
    return points

def circle(number_of_points : int, radius : float, center_x : float, center_y : float):
    points = []
    angle = 0
    angle_step = 2 * math.pi / number_of_points
    for _ in range(number_of_points):
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        points.append([x, y])
        angle += angle_step
    return points

def dump_to_obj(points : List[List[float]]):
    with open("test.obj", 'w') as f:
        for point in points:
            f.write(f"v {point[0]} {point[1]} 0\n")
        face = "f "
        for i in range(len(points)):
            face += f"{i+1} "
        face += '\n'
        f.write(face)