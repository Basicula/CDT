import enum
from typing import List, Tuple

import numpy as np
import copy

class SuperGeometryType(enum.Enum):
    TRIANGLE = 3
    SQUARE = 4

class PointLocation(enum.Enum):
    INSIDE = -1
    OUTSIDE = 1
    ON_EDGE = 0
    ON_VERTEX = 2

class CDT:
    '''
    Constraint Delaunay triangulation algorithm
    '''
    triangle_edge_ids = [(0, 1), (1, 2), (2, 0)]

    def __init__(self):
        self.points = []
        self.points_in_triangulation = 0
        self.steps = []
        self.triangles = []
        self.constraints = []
        self.constraints_segments = []
        self.neighbors = {}

    def add_points(self, points : List[List[float]]) -> None:
        '''
        Adds points to the future triangulation

        points - List[List[float]], required
            List of the 2-dimensional points where List[float] is just a pair of float numbers
        '''
        if len(self.points) == 0:
            self.points = np.array(points)
        else:
            self.points = np.concatenate((self.points, points), axis=0)

    def add_constraint_segment(self, segment_constraint : List[List[float]]) -> None:
        '''
        Adds the segment constraint to the future triangulation

        segment_constraint - List[List[float]], required
            The segment that is represented by two 2d points
        '''
        assert(len(segment_constraint) == 2)
        start_id = len(self.points)
        constraint_segments = [[start_id, start_id + 1]]
        self.__add_constraint(segment_constraint, constraint_segments)

    def add_constraint_line(self, constraint : List[List[float]]) -> None:
        '''
        Adds the line constraint to the future triangulation

        constraint - List[List[float]], required
            The sequence of the 2d points
        '''
        start_id = len(self.points)
        end_id = start_id + len(constraint) - 1
        constraint_segments = []
        for start_segment_id in range(start_id, end_id):
            constraint_segments.append([start_segment_id, start_segment_id + 1])
        self.__add_constraint(constraint, constraint_segments)

    def add_constraint_closed_region(self, constraint : List[List[float]]) -> None:
        '''
        Adds the closed region constraint to the future triangulation
        closed region is represented by sequence of 2d points where
        first and last point isn't equal, but it'll be treated as closed polyline

        constraint - List[List[float]], required
            The sequence of the 2d points
        '''
        start_id = len(self.points)
        end_id = start_id + len(constraint) - 1
        constraint_segments = []
        for start_segment_id in range(start_id, end_id):
            constraint_segments.append([start_segment_id, start_segment_id + 1])
        constraint_segments.append([end_id, start_id])
        self.__add_constraint(constraint, constraint_segments)

    def __add_constraint(self, constraint : List[List[float]], constraint_segments : List[List[int]]) -> None:
        '''
        Private function that stores common input from constraints input functions

        constraint - List[List[float]], required
            The sequence of the 2d points that represent constraint
        constraint_segments - List[List[int]], required
            The indexed segments built on top of the constraint points sequence
        '''
        self.constraints.append(constraint)
        self.constraints_segments.append(constraint_segments)
        self.add_points(constraint)

    def triangulate(self) -> None:
        '''
        Triangulates after all desired input is set
        '''
        if len(self.points) < 3:
            return
        self.triangles = []
        self.neighbors = {}
        self.__normalize_input()
        self.__add_super_geometry(SuperGeometryType.TRIANGLE)
        self.__add_points_to_triangulation()
        self.__process_constraints()
        self.__mark_inner_outer_triangles()
        self.__remove_outer_triangles()
        self.__remove_super_geometry()

    def __append_step(self, step_name : str = "Undefined") -> None:
        '''
        Appends step that is represented by (points, triangles, name) during triangulation process

        step_name - str, optional
            The name of the step
        '''
        self.steps.append([copy.deepcopy(self.points), copy.deepcopy(self.triangles), step_name])

    def __check_segment_intersection(self, segment1 : List[List[int]], segment2 : List[List[int]]) -> bool:
        '''
        Checks segment intersection, help function

        Returns True if segments are intersected, otherwise - False

        segment1 - List[List[int]]
            First indexed segment
        segment2 - List[List[int]]
            Second indexed
        '''
        p1 = self.points[segment1[0]]
        q1 = self.points[segment1[1]]
        p2 = self.points[segment2[0]]
        q2 = self.points[segment2[1]]
        orientations = [
            np.cross(q1 - p1, p2 - q1), # (p1, q1, p2)
            np.cross(q1 - p1, q2 - q1), # (p1, q1, q2)
            np.cross(q2 - p2, p1 - q2), # (p2, q2, p1)
            np.cross(q2 - p2, q1 - q2), # (p2, q2, q1)
        ]
        return orientations[0] * orientations[1] < 0 and orientations[2] * orientations[3] < 0

    def __process_constraints(self) -> None:
        '''
        Private help function that processes all segments
        built on top of the input constraints

        Inserts segment to the triangulation if it's not yet exists
        '''
        for constraint_segments in self.constraints_segments:
            for segment in constraint_segments:
                segment_present = False
                for triangle in self.triangles:
                    if segment_present:
                        break
                    for triangle_edge in self.triangle_edge_ids:
                        if segment[0] == triangle[triangle_edge[0]] and segment[1] == triangle[triangle_edge[1]] or\
                           segment[1] == triangle[triangle_edge[0]] and segment[0] == triangle[triangle_edge[1]]:
                            segment_present = True
                            break
                if not segment_present:
                    triangles_to_check = self.__add_segment_to_triangulation(segment)
                    self.__check_circumcircles(triangles_to_check)

    def __traverse_segment(self, segment : List[List[int]]) -> Tuple[List[int], List[int], List[int]]:
        '''
        Checks triangles that is intersected with given segment

        Returns [triangles that was intersected, list of the left point ids, list of the right points ids]

        segment - List[List[int]], required
            The segment to check before insertion to the triangulation
        '''

        right_region = [segment[0]]
        left_region = [segment[0]]
        triangles_to_check = []
        for triangle_id, triangle in enumerate(self.triangles):
            if segment[0] in triangle:
                triangles_to_check.append(triangle_id)
        triangle_to_remove = []
        while len(triangles_to_check) > 0:
            curr_triangle_id = triangles_to_check.pop()
            curr_triangle = self.triangles[curr_triangle_id]
            if segment[1] in curr_triangle:
                right_region.append(segment[1])
                left_region.append(segment[1])
                break
            for triangle_edge_id in self.triangle_edge_ids:
                triangle_edge = [curr_triangle[triangle_edge_id[0]], curr_triangle[triangle_edge_id[1]]]
                if self.__check_segment_intersection(segment, triangle_edge):
                    triangle_to_remove.append(curr_triangle_id)
                    next_triangle_id = self.neighbors[curr_triangle_id][triangle_edge_id[0]]
                    if next_triangle_id in triangle_to_remove:
                        continue
                    triangles_to_check.append(next_triangle_id)
                    if not triangle_edge[0] in right_region:
                        right_region.append(triangle_edge[0])
                    if not triangle_edge[1] in left_region:
                        left_region.append(triangle_edge[1])
                    break
        return triangle_to_remove, left_region, right_region

    def __add_segment_to_triangulation(self, segment : List[List[int]]) -> List[int]:
        '''
        Private help function that adds indexed segment to the triangulation
        i.e. if there is no triangle edge with ids defined by segment inserts it to the triangulation

        Returns list of triangles which circumcircles need to check
        '''

        triangle_to_remove, left_region, right_region = self.__traverse_segment(segment)
        right_region.reverse()

        for triangle_to_remove_id in triangle_to_remove:
            self.triangles[triangle_to_remove_id] = [0, 0, 0]

        triangles_to_check = []
        for region in [left_region, right_region]:
            point_id = 1
            while len(region) > 2:
                a = self.points[region[point_id - 1]]
                b = self.points[region[point_id]]
                c = self.points[region[point_id + 1]]
                if np.cross(b - a, c - a) >= 0:
                    point_id += 1
                else:
                    triangle_id = -1
                    if len(triangle_to_remove) > 0:
                        triangle_id = triangle_to_remove.pop()
                    self.__insert_triangle(
                        [region[point_id - 1], region[point_id + 1], region[point_id]],
                        triangle_id
                    )
                    if triangle_id == -1:
                        triangle_id = len(self.triangles) - 1
                    triangles_to_check.append(triangle_id)
                    region.pop(point_id)
                    point_id = 1
        
        if len(triangle_to_remove) > 0:
            triangle_to_remove.sort()
            for triangle_id in reversed(triangle_to_remove):
                self.__remove_triangle(triangle_id)

        return triangles_to_check

    def __add_points_to_triangulation(self) -> None:
        '''
        Adds all free points to the triangulation
        '''
        self.points_in_triangulation = len(self.points) - self.super_geometry_points_cnt
        for point_id in range(self.points_in_triangulation):
            self.__add_point_to_triangulation(point_id)

    def __add_point_to_triangulation(self, point_id : int) -> None:
        '''
        Adds point to the triangulation if point inside some triangle it'll be splitted
        if point lies on some triangle edge it also will be splitted
        after each splitting operation new triangles will apppear in triangulation

        point_id - int, required
            The id of the point that is going to be inserted
        '''
        for triangle_id, triangle in enumerate(self.triangles):
            point_location = self.__locate_point(triangle, point_id)
            if point_location == PointLocation.OUTSIDE:
                continue
            triangles_to_check = []
            if point_location == PointLocation.ON_EDGE:
                triangles_to_check = self.__split_edge(triangle_id, point_id)

            if point_location == PointLocation.INSIDE:
                triangles_to_check = self.__split_triangle(triangle_id, point_id)

            self.__check_circumcircles(triangles_to_check)
            break

    def __check_circumcircles(self, triangles_to_check : List[int]) -> None:
        '''
        Checks Delaunay condition i.e. checks that triangle circumcircle hasn't any other points

        triangles_to_check - List[int]
            The list with the triangle ids that need to be checked
        '''
        while len(triangles_to_check) > 0:
            triangle_id = triangles_to_check.pop()
            triangle_neighbors = self.neighbors[triangle_id]
            for triangle_edge_id, neighbor_triangle_id in enumerate(triangle_neighbors):
                if neighbor_triangle_id == None:
                    continue
                neighbor_triangle = self.triangles[neighbor_triangle_id]
                neighbor_triangle_edge_id = self.__find_neighbor_triangle_edge_id(triangle_id, triangle_edge_id)
                opposite_point_id = neighbor_triangle[(neighbor_triangle_edge_id + 2) % 3]
                if not self.__is_in_circumcircle(triangle_id, opposite_point_id):
                    continue

                for new_to_check in self.neighbors[neighbor_triangle_id]:
                    if new_to_check != triangle_id and new_to_check != None:
                        triangles_to_check.append(new_to_check)

                self.__flip_edge(triangle_id, neighbor_triangle_id, triangle_edge_id, neighbor_triangle_edge_id)

    def __flip_edge(self, triangle_id : int, neighbor_id : int, triangle_edge_id : int, neighbor_edge_id : int) -> None:
        '''
        Flips edge for pair of triangles that shares its edge

        triangle_id - int, required
            The first triangle
        neighbor_id - int, required
            The second triangle that is neighbor of the first triangle
        triangle_edge_id - int, required
            The edge id (0, 1 or 2) that is shared in first triangle
        neighbor_edge_id - int, required
            The edge id (0, 1 or 2) that is shared in second triangle
        '''
        triangle = self.triangles[triangle_id]
        if self.__is_constraint_segment([triangle[triangle_edge_id], triangle[(triangle_edge_id + 1) % 3]]):
            return
        triangle_point_id = (triangle_edge_id + 2) % 3
        opposite_point_id = (neighbor_edge_id + 2) % 3
        self.triangles[triangle_id][triangle_edge_id] = self.triangles[neighbor_id][opposite_point_id]
        self.triangles[neighbor_id][neighbor_edge_id] = self.triangles[triangle_id][triangle_point_id]
        self.neighbors[neighbor_id][neighbor_edge_id] = self.neighbors[triangle_id][triangle_point_id]
        self.__update_neighbors(self.neighbors[triangle_id][triangle_point_id], triangle_id, neighbor_id)
        self.neighbors[triangle_id][triangle_point_id] = neighbor_id
        self.neighbors[triangle_id][triangle_edge_id] = self.neighbors[neighbor_id][opposite_point_id]
        self.__update_neighbors(self.neighbors[neighbor_id][opposite_point_id], neighbor_id, triangle_id)
        self.neighbors[neighbor_id][opposite_point_id] = triangle_id

    def __is_constraint_segment(self, triangle_segment : List[int]) -> bool:
        '''
        Checks whether the given triangle segment matches any segment from the input constraints

        triangle_segment - List[int], required
            Triangle segment to be checked it's represented by two point ids
        '''
        for constraint_segments in self.constraints_segments:
            for constraint_segment in constraint_segments:
                if triangle_segment[0] == constraint_segment[0] and triangle_segment[1] == constraint_segment[1] or\
                   triangle_segment[1] == constraint_segment[0] and triangle_segment[0] == constraint_segment[1]:
                    return True
        return False

    def __is_in_circumcircle(self, triangle_id : int, point_id : int) -> bool:
        '''
        Checks whether the point with the given id lies in the triangle's circumcircle with the given triangle id

        triangle_id - int, required
            The id of the triangle to be checked
        point_id - int, required
            The id of the point that is going to be checked
        '''
        triangle = self.triangles[triangle_id]
        point = self.points[point_id]
        ax = self.points[triangle[0]][0]
        ay = self.points[triangle[0]][1]
        bx = self.points[triangle[1]][0]
        by = self.points[triangle[1]][1]
        cx = self.points[triangle[2]][0]
        cy = self.points[triangle[2]][1]
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
        r_sqr = (ax - ux) * (ax - ux) + (ay - uy) * (ay - uy)
        return (point[0] - ux) * (point[0] - ux) + (point[1] - uy) * (point[1] - uy) <= r_sqr

    def __split_edge(self, triangle_id : int, point_id : int) -> List[int]:
        '''
        Splits edge if the new point lies on the triangle edge
        produces 2 new triangles and updates 2 other triangles that
        shares edge where target point lies or 1 if triangle edge is
        outer edge of the triangulation

        Returns list of the ids of the triangles that was updated and added

        triangle_id - int, required
            The id of the triangle which edge will be splitted
        point_id - int, required
            The id of the point that splits triangle edge
        '''
        triangle = self.triangles[triangle_id]
        point = self.points[point_id]
        edge_id = -1
        for triangle_edge in self.triangle_edge_ids:
            a = np.array(self.points[triangle[triangle_edge[0]]])
            b = np.array(self.points[triangle[triangle_edge[1]]])
            if np.linalg.norm(a - b) == np.linalg.norm(point - a) + np.linalg.norm(point - b):
                edge_id = triangle_edge[0]
                break
        
        neighbor_triangle_id = self.neighbors[triangle_id][edge_id]

        t1_id = len(self.triangles)
        t2_id = t1_id + 1
        if neighbor_triangle_id == None:
            t2_id = None

        triangle_opposite_point_id = (edge_id + 2) % 3
        t1 = [point_id, triangle[triangle_opposite_point_id], triangle[edge_id]]
        t1_adj = [triangle_id, self.neighbors[triangle_id][triangle_opposite_point_id], t2_id]
        self.__update_neighbors(self.neighbors[triangle_id][triangle_opposite_point_id], triangle_id, t1_id)
        self.triangles[triangle_id][edge_id] = point_id
        self.neighbors[triangle_id][triangle_opposite_point_id] = t1_id
        self.neighbors[t1_id] = t1_adj
        self.triangles.append(t1)

        if neighbor_triangle_id != None:
            neighbor_triangle_edge_id = self.__find_neighbor_triangle_edge_id(triangle_id, edge_id)
            neighbor_triangle_edge_end_id = (neighbor_triangle_edge_id + 1) % 3
            neighbor_triangle_opposite_point_id = (neighbor_triangle_edge_id + 2) % 3
            neighbor_triangle = self.triangles[neighbor_triangle_id]
            t2 = [point_id, neighbor_triangle[neighbor_triangle_edge_end_id], neighbor_triangle[neighbor_triangle_opposite_point_id]]
            t2_adj = [t1_id, self.neighbors[neighbor_triangle_id][neighbor_triangle_edge_end_id], neighbor_triangle_id]
            self.__update_neighbors(self.neighbors[neighbor_triangle_id][neighbor_triangle_edge_end_id], neighbor_triangle_id, t2_id)
            self.triangles[neighbor_triangle_id][neighbor_triangle_edge_end_id] = point_id
            self.neighbors[neighbor_triangle_id][neighbor_triangle_edge_end_id] = t2_id
            self.neighbors[t2_id] = t2_adj
            self.triangles.append(t2)

            return [triangle_id, neighbor_triangle_id, t1_id, t2_id]
        
        return [triangle_id, t1_id]

    def __split_triangle(self, triangle_id : int, point_id : int) -> List[int]:
        '''
        Splits triangle if the new point lies in it
        produces 2 new triangles and updates given triangle

        Returns list of the ids of the triangles that was updated and added

        triangle_id - int, required
            The id of the triangle which edge will be splitted
        point_id - int, required
            The id of the point that splits triangle
        '''
        triangle = self.triangles[triangle_id]
        t1_id = len(self.triangles)
        t2_id = t1_id + 1
        
        t1 = [point_id, triangle[0], triangle[1]]
        t1_adj = [triangle_id, self.neighbors[triangle_id][0], t2_id]
        self.triangles.append(t1)
        self.neighbors[t1_id] = t1_adj
        self.__update_neighbors(self.neighbors[triangle_id][0], triangle_id, t1_id)
        
        t2 = [point_id, triangle[1], triangle[2]]
        t2_adj = [t1_id, self.neighbors[triangle_id][1], triangle_id]
        self.triangles.append(t2)
        self.neighbors[t2_id] = t2_adj
        self.__update_neighbors(self.neighbors[triangle_id][1], triangle_id, t2_id)

        self.triangles[triangle_id] = [point_id, triangle[2], triangle[0]]
        self.neighbors[triangle_id] = [t2_id, self.neighbors[triangle_id][2], t1_id]

        return [t1_id, t2_id, triangle_id]

    def __find_neighbor_triangle_edge_id(self, triangle_id : int, edge_id : int) -> int:
        '''
        Finds id (0, 1, 2) for sharing edge in neighbor triangle

        Returns searching id or -1 if no such edge or triangle

        triangle_id - int, required
            The id of the triangle which neighbor need to process
        edge_id - int, required
            The id of the edge in the given triangle
        '''
        neighbor_id = self.neighbors[triangle_id][edge_id]
        if neighbor_id == None:
            return -1
        neighbors_of_neighbor = self.neighbors[neighbor_id]
        for edge_id, neighbor_of_neighbor in enumerate(neighbors_of_neighbor):
            if neighbor_of_neighbor == triangle_id:
                return edge_id
        return -1

    def __update_neighbors(self, triangle_id : int, old_neighbor_id : int, new_neighbor_id : int) -> None:
        '''
        Updates neighbor for triangle by searching old neighbor id within all neighbors and
        replaces it with the new one

        triangle_id - int, required
            The id of the triangle which neighbor will be updated
        old_neighbor_id - int, required
            The id of the old neighbor triangle
        ne_neighbor_id - int, required
            The id of the new neighbor triangle
        '''
        if triangle_id == None:
            return
        for i, neighbor in enumerate(self.neighbors[triangle_id]):
            if neighbor == old_neighbor_id:
                self.neighbors[triangle_id][i] = new_neighbor_id
                break

    def __insert_triangle(self, triangle : List[int], triangle_id : int = -1) -> None:
        '''
        Inserts new triangle to the triangulation

        triangle - List[int], required
            The triangle that is going to be inserted represented by 3 point ids
        triangle_id - int, required
            The id for new triangle if it doesn't set appends new triangle
        '''
        if triangle_id == -1:
            triangle_id = len(self.triangles)
            self.triangles.append(triangle)
        else:
            self.triangles[triangle_id] = triangle
        self.neighbors[triangle_id] = [None, None, None]
        for neighbor_triangle_id, neighbor_triangle in enumerate(self.triangles):
            edge_pairs = [
                [(0, 1), (0, 1)], [(0, 1), (1, 2)], [(0, 1), (2, 0)],
                [(1, 2), (0, 1)], [(1, 2), (1, 2)], [(1, 2), (2, 0)],
                [(2, 0), (0, 1)], [(2, 0), (1, 2)], [(2, 0), (2, 0)],
            ]
            for edge_pair in edge_pairs:
                triangle_edge = [triangle[edge_pair[0][0]], triangle[edge_pair[0][1]]]
                neighbor_triangle_edge = [neighbor_triangle[edge_pair[1][0]], neighbor_triangle[edge_pair[1][1]]]
                if triangle_edge[0] == neighbor_triangle_edge[1] and triangle_edge[1] == neighbor_triangle_edge[0]:
                    self.neighbors[triangle_id][edge_pair[0][0]] = neighbor_triangle_id
                    self.neighbors[neighbor_triangle_id][edge_pair[1][0]] = triangle_id
                    break

    def __locate_point(self, triangle : List[int], point_id : int) -> PointLocation:
        '''
        Finds where point is located relatively to triangle

        Returns PointLocation

        triangle - List[int], required
            The triangle that is checked
        point_id - int, required
            The id of the point which location is searching
        '''
        cross = np.array([])
        for i in range(3):
            a = self.points[triangle[i]]
            b = self.points[triangle[i + 1]] if i < 2 else self.points[triangle[0]]
            p = self.points[point_id]
            cross = np.append(cross, np.cross(p - a, b - a))
        if np.all(cross < 0):
            return PointLocation.INSIDE
        elif np.any(cross > 0):
                return PointLocation.OUTSIDE
        return PointLocation.ON_EDGE

    def __add_super_triangle(self) -> None:
        '''
        Adds super triangle that is needed for Delaunay triangulation process
        i.e. the first Delaunay triangle
        '''
        min_corner = np.min(self.points, axis=0) - 1
        max_corner = np.max(self.points, axis=0)
        delta = np.max(max_corner - min_corner) * 2
        self.points = np.concatenate((
            self.points,
            [
                min_corner,
                [min_corner[0] + delta, min_corner[1]],
                [min_corner[0], min_corner[1] + delta]
            ]), axis=0)
        points_cnt = len(self.points)
        self.super_geometry = [[points_cnt - 3, points_cnt - 2, points_cnt - 1]]
        self.triangles.append(self.super_geometry[0])
        self.neighbors[0] = [None, None, None]

    def __add_super_square(self) -> None:
        '''
        Adds super square that is needed for Delaunay triangulation process
        i.e. the first two Delaunay triangle
        '''
        min_corner = np.min(self.points, axis=0)
        max_corner = np.max(self.points, axis=0)
        min_corner -= 1
        max_corner += 1
        self.points = np.concatenate((self.points, [min_corner,
                                                    [max_corner[0],min_corner[1]],
                                                    max_corner,
                                                    [min_corner[0], max_corner[1]]]), axis=0)
        point_cnt = len(self.points)
        self.super_geometry = [[point_cnt - 4, point_cnt - 3, point_cnt - 1],
                               [point_cnt - 3, point_cnt - 2, point_cnt - 1]]
        self.triangles.append(copy.copy(self.super_geometry[0]))
        self.neighbors[0] = [None, 1, None]
        self.triangles.append(copy.copy(self.super_geometry[1]))
        self.neighbors[1] = [None, None, 0]

    def __add_super_geometry(self, super_geometry_type : SuperGeometryType) -> None:
        '''
        Adds specified by enum super geometry for the triangulation process

        super_geometry_type - SuperGeometryType
            The desired type of the super geometry
        '''
        self.super_geometry_points_cnt = super_geometry_type.value
        if super_geometry_type == SuperGeometryType.TRIANGLE:
            self.__add_super_triangle()
        elif super_geometry_type == SuperGeometryType.SQUARE:
            self.__add_super_square()

    def __remove_super_geometry(self) -> None:
        '''
        Removes all triangles that was generated by super geometry
        '''
        points_to_remove = np.array([], dtype=int)
        for super_triangle in self.super_geometry:
            points_to_remove = np.append(points_to_remove, super_triangle)
            triangles_to_remove = []
            for triangle_id, triangle in enumerate(self.triangles):
                for point_id in triangle:
                    if point_id in super_triangle:
                        triangles_to_remove.append(triangle_id)
                        break

            for triangle_id in reversed(triangles_to_remove):
                self.triangles.pop(triangle_id)
        self.points = np.delete(self.points, np.unique(points_to_remove), axis=0)

    def __remove_triangle(self, triangle_id : int) -> None:
        '''
        Removes triangle from the triangulation

        triangle_id - int, required
            The id of the triangle that will be deleted
        '''
        for neighbor_triangle_id in self.neighbors[triangle_id]:
            if neighbor_triangle_id == None:
                continue
            for i in range(3):
                if self.neighbors[neighbor_triangle_id][i] == triangle_id:
                    self.neighbors[neighbor_triangle_id][i] == None
                    break
        for id in self.neighbors:
            for i in range(3):
                if self.neighbors[id][i] != None and self.neighbors[id][i] > triangle_id:
                    self.neighbors[id][i] -= 1
        self.triangles.pop(triangle_id)

    def __mark_inner_outer_triangles(self) -> None:
        '''
        Sets status for each triangle whether it's outer or inner depending on constraints
        '''
        self.triangle_region_id = [0] * len(self.triangles)
        for constraint_segments in self.constraints_segments:
            segment = constraint_segments[0]
            triangle_id = -1
            for id, triangle in enumerate(self.triangles):
                if triangle_id != -1:
                    break
                for triangle_edge_id in self.triangle_edge_ids:
                    if segment[0] == triangle[triangle_edge_id[0]] and segment[1] == triangle[triangle_edge_id[1]]:
                        triangle_id = id
                        break
            inner_triangles = [triangle_id]
            while len(inner_triangles) > 0:
                inner_triangle_id = inner_triangles.pop()
                self.triangle_region_id[inner_triangle_id] = 1
                inner_triangle = self.triangles[inner_triangle_id]
                for triangle_edge_id in self.triangle_edge_ids:
                    if self.__is_constraint_segment([inner_triangle[triangle_edge_id[0]], inner_triangle[triangle_edge_id[1]]]):
                        continue
                    next_triangle = self.neighbors[inner_triangle_id][triangle_edge_id[0]]
                    if next_triangle != None and self.triangle_region_id[next_triangle] != 1:
                        inner_triangles.append(next_triangle)

    def __remove_outer_triangles(self) -> None:
        '''
        Removes all triangles that is marked as outer
        '''
        to_remove = []
        for triangle_id in range(len(self.triangles)):
            if self.triangle_region_id[triangle_id] == 0:
                to_remove.append(triangle_id)
        for triangle_id in reversed(to_remove):
            self.__remove_triangle(triangle_id)

    def __normalize_input(self) -> None:
        '''
        Optional step fits input points to the unit box
        '''
        min_point = np.min(self.points, axis=0)
        max_point = np.max(self.points, axis=0)
        delta = max_point - min_point
        self.points = np.array([(point - min_point) / delta for point in self.points])