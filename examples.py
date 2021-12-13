import matplotlib.pyplot as plt
from typing import List
import numpy as np

from input_generator import *
from delaunay_triangulation import *

def render_polygon_points(points : List[List[float]], axis):
    points = np.array(points)
    #axis.plot(points[:,0], points[:,1])
    axis.scatter(points[:,0], points[:, 1], c='r')

def render_triangulation(points : List[List[float]], triangles : List[List[int]], axis):
    points = np.array(points)
    axis.scatter(points[:,0], points[:, 1], c='r')
    for triangle in triangles:
        triangle_points = np.array([points[triangle[0]], points[triangle[1]], points[triangle[2]], points[triangle[0]]])
        axis.plot(triangle_points[:,0], triangle_points[:,1], c='k')

def render(points : List[List[float]], triangles : List[List[int]]):
    fig = plt.figure()
    ax_points = fig.add_subplot(2, 2, (1, 3))
    ax_triangulation = fig.add_subplot(2, 2, (2,4))

    render_polygon_points(points, ax_points)
    render_triangulation(points, triangles, ax_triangulation)

    plt.show()

def render_steps(steps):
    fig = plt.figure()
    ax_points = fig.add_subplot(2, 2, (1, 3))
    ax_triangulation = fig.add_subplot(2, 2, (2,4))

    for step in steps:
        ax_points.clear()
        ax_triangulation.clear()
        #print(step[2])
        #print(step[1])
        #print(step[0])
        render_polygon_points(step[0], ax_points)
        render_triangulation(step[0], step[1], ax_triangulation)

        plt.draw()
        plt.pause(0.05)
        
    ax_points.clear()
    ax_triangulation.clear()
    render_polygon_points(steps[-1][0], ax_points)
    render_triangulation(steps[-1][0], steps[-1][1], ax_triangulation)
    plt.show()

def smile():
    cdt = CDT()
    head = circle(20, 1, 0, 0)
    left_eye = list(reversed(circle(10, 0.1, -.25, .25)))
    right_eye = list(reversed(circle(10, 0.1, .25, .25)))
    smile = list(reversed(circle(20, 0.25, 0, -.25)[11:]))
    cdt.add_constraint_closed_region(head)
    cdt.add_constraint_closed_region(left_eye)
    cdt.add_constraint_closed_region(right_eye)
    cdt.add_constraint_closed_region(smile)
    cdt.triangulate()
    #render_steps(cdt.steps)
    render(cdt.points, cdt.triangles)

def random_polygon():
    points = generate_polygon_points(50, 0, 0)
    cdt = CDT()
    cdt.add_constraint_closed_region(points)
    cdt.triangulate()
    #render_steps(cdt.steps)
    render(cdt.points, cdt.triangles)

def random():
    points = random_points(15, 0, 0, 1, 1)
    cdt = CDT()
    cdt.add_points(points)
    cdt.triangulate()
    #render_steps(cdt.steps)
    render(cdt.points, cdt.triangles)

if __name__ == "__main__":
    #points = circle(15, 1, 0, 0)
    #points = random_points(10, -10, -10, 10, 10)
    #dump_to_obj(points[:-1])
    #render(points)
    #random_polygon()
    #random()
    smile()