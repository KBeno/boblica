from typing import Union, List
import copy
import math
import numpy as np


"""
Principles:

- geometry objects are defined by the minimum required information
- Points are made of coordinates (floats), everything else is based on Points except for Vectors
"""


class Point:

    def __init__(self, x: float, y: float, z: float = 0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __str__(self):
        return self.pretty_print()

    def pretty_print(self, indentation=''):
        return "{ind}{x}, {y}, {z} (Point)".format(x=self.x, y=self.y, z=self.z, ind=indentation)

    def coordinates(self):
        return self.x, self.y, self.z

    def __sub__(self, other):
        if isinstance(other, Point):
            return Vector(x=self.x - other.x,
                          y=self.y - other.y,
                          z=self.z - other.z)
        elif isinstance(other, Vector):
            return Point(x=self.x - other.x,
                         y=self.y - other.y,
                         z=self.z - other.z)

    def __add__(self, other):
        if isinstance(other, Vector):
            return Point(x=self.x + other.x,
                         y=self.y + other.y,
                         z=self.z + other.z)

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y and self.z == other.z:
            return True
        else:
            return False


class Vector:

    def __init__(self, x, y, z: float = 0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __str__(self):
        return self.pretty_print()

    def pretty_print(self, indentation=''):
        return "{ind}{x}, {y}, {z} (Vector)".format(x=self.x, y=self.y, z=self.z, ind=indentation)

    def coordinates(self):
        return self.x, self.y, self.z

    def length(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def unitize(self):
        return Vector(self.x / self.length(), self.y / self.length(), self.z / self.length())

    def cross_product(self, vector2):
        product_x = self.y * vector2.z - self.z * vector2.y
        product_y = -self.x * vector2.z + self.z * vector2.x
        product_z = self.x * vector2.y - self.y * vector2.x
        return Vector(product_x, product_y, product_z)

    def scalar_product(self, vector2):
        product = 0
        for xyz in [0, 1, 2]:
            product += self.coordinates()[xyz] * vector2.coordinates()[xyz]
        return product

    def __mul__(self, other):
        if isinstance(other, Vector):
            # scalar (dot) product
            product = 0
            for xyz in [0, 1, 2]:
                product += self.coordinates()[xyz] * other.coordinates()[xyz]
            return product
        elif isinstance(other, (float, int)):
            return Vector(self.x * other, self.y * other, self.z * other)

    def angle(self, vector2):
        # angle between the instance vector and the given vector in degrees
        # always positive and smaller or equal to 180°
        return math.degrees(math.acos(self.scalar_product(vector2) / self.length() / vector2.length()))

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __truediv__(self, other: float):
        return self * other ** -1

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y and self.z == other.z:
            return True
        else:
            return False


class Plane:
    def __init__(self, normal: Vector, point: Point):
        self.normal = normal
        self.point = point

    def __str__(self):
        return self.pretty_print()

    def pretty_print(self, indentation=''):
        return '{ind}Plane:\n'.format(ind=indentation) +\
               '{ind}|--Normal: {s}\n'.format(s=self.normal.pretty_print(), ind=indentation) +\
               '{ind}`--Point: {e}\n'.format(e=self.point.pretty_print(), ind=indentation)

    def intersect(self, other: Union['Ray', 'Plane']):
        if isinstance(other, Ray):
            # solve the linear equation system aX = b
            plane_eq, plane_ord = self.get_equation(standardize=True)
            ray_eq, ray_ord = other.get_equation(standardize=True)
            a = np.append(plane_eq, ray_eq, axis=0)
            b = np.append(plane_ord, ray_ord, axis=0)

            try:
                solution = np.linalg.solve(a, b)
            except np.linalg.LinAlgError:
                # parallel
                return None

            return Point(
                x=solution[0, 0],
                y=solution[1, 0],
                z=solution[2, 0]
            )
        if isinstance(other, Plane):
            # direction of intersection ray
            vector = self.normal.cross_product(other.normal)
            if vector == Vector(0, 0, 0):
                # parallel
                return None
            else:
                # get largest absolute coordinate value
                xyz = [abs(vector.x), abs(vector.y), abs(vector.z)]
                set_0_coord = xyz.index(max(xyz))

                # set this coordinate to 0 to solve the equation of the two planes
                eq1, ord1 = self.get_equation(standardize=True)
                eq2, ord2 = other.get_equation(standardize=True)

                a = np.append(eq1, eq2, axis=0)
                b = np.append(ord1, ord2, axis=0)
                # delete the corresponding column from the matrix
                i = [True, True, True]
                i[set_0_coord] = False
                a = a[:, i]

                # we should be able to solve this, because parallel case was checked already
                solution = np.linalg.solve(a, b)

                if set_0_coord == 0:
                    point = Point(0, solution[0, 0], solution[1, 0])
                elif set_0_coord == 1:
                    point = Point(solution[0, 0], 0, solution[1, 0])
                else:
                    point = Point(solution[0, 0], solution[1, 0], 0)

            return Ray(
                vector=vector,
                point=point
            )

    def get_equation(self, standardize=False):
        # http://tutorial.math.lamar.edu/Classes/CalcIII/EqnsOfPlanes.aspx
        a = self.normal.x
        b = self.normal.y
        c = self.normal.z
        d = a * self.point.x + b * self.point.y + c * self.point.z
        if standardize:
            # return the coefficients of the equation in this form aX + bY + cZ = d
            return (
                np.array([
                    [a, b, c]
                ]),
                np.array([
                    [d]
                ])
            )
        return {
            'a': a, 'b': b, 'c': c, 'd': d
        }

    def print_equation(self):
        return '{a}x + {b}y + {c}z = {d}'.format(**self.get_equation())


class Ray:
    def __init__(self, vector: Vector, point: Point):
        self.vector = vector
        self.point = point

    def get_equation(self, standardize=False):
        # http://tutorial.math.lamar.edu/Classes/CalcIII/EqnsOfLines.aspx

        x0 = self.point.x
        y0 = self.point.y
        z0 = self.point.z
        a = self.vector.x
        b = self.vector.y
        c = self.vector.z
        if standardize:
            # return the coefficients of the equations in this form aX + bY + cZ + d = 0
            if a == 0:
                # 1X + 0Y + 0Z = x0
                a1, b1, c1, d1 = 1, 0, 0, x0
                if b == 0:
                    # 0X + 1Y + 0Z = y0
                    a2, b2, c2, d2 = 0, 1, 0, y0
                elif c == 0:
                    # 0X + 0Y + 1Z = z0
                    a2, b2, c2, d2 = 0, 0, 1, z0
                else:
                    # 0X + cY - bZ = y0*c - z0*b
                    a2, b2, c2, d2 = 0, c, -b, y0 * c - z0 * b
            elif b == 0:
                # 0X + 1Y + 0Z = y0
                a1, b1, c1, d1 = 0, 1, 0, y0
                if c == 0:
                    # 0X + 0Y + 1Z = z0
                    a2, b2, c2, d2 = 0, 0, 1, z0
                else:
                    # cX + 0Y - aZ = x0*c - z0*a
                    a2, b2, c2, d2 = c, 0, -a, x0 * c - z0 * a
            else:
                # bX - aY + 0Z = x0*b - y0*a
                a1, b1, c1, d1 = b, -a, 0, x0 * b - y0 * a
                if c == 0:
                    # 0X + 0Y + 1Z = z0
                    a2, b2, c2, d2 = 0, 0, 1, z0
                else:
                    # cX + 0Y - aZ = x0*c - z0*a
                    a2, b2, c2, d2 = c, 0, -a, x0 * c - z0 * a
            return (
                np.array([
                    [a1, b1, c1],
                    [a2, b2, c2]
                ]),
                np.array([
                    [d1],
                    [d2]
                ])
            )
        else:
            return {
                'x0': x0, 'y0': y0, 'z0': z0, 'a': a, 'b': b, 'c': c,
            }

    def print_equation(self):
        coeffs = self.get_equation()
        if coeffs['a'] == 0:
            eq1 = 'x = {x0}'.format(**coeffs)
            if coeffs['b'] == 0:
                eq2 = 'y = {y0}, '.format(**coeffs)
            elif coeffs['c'] == 0:
                eq2 = 'z = {z0}, '.format(**coeffs)
            else:
                eq2 = '(y - {y0}) / {b} = (z - {z0}) / {c}'.format(**coeffs)
        elif coeffs['b'] == 0:
            eq1 = 'y = {y0}'.format(**coeffs)
            if coeffs['c'] == 0:
                eq2 = 'z = {z0}, '.format(**coeffs)
            else:
                eq2 = '(x - {x0}) / {a} = (z - {z0}) / {c}'.format(**coeffs)
        else:
            eq1 = '(x - {x0}) / {a} = (y - {y0}) / {b}'.format(**coeffs)
            if coeffs['c'] == 0:
                eq2 = 'z = {z0}, '.format(**coeffs)
            else:
                eq2 = '(x - {x0}) / {a} = (z - {z0}) / {c}'.format(**coeffs)

        return eq1 + '\n' + eq2

    def intersect(self, other: Plane) -> Point:
        return other.intersect(self)


class Line:

    def __init__(self, start: Point, end: Point):
        self.start = start
        self.end = end

    def __str__(self):
        return self.pretty_print()

    def pretty_print(self, indentation=''):
        return '{ind}Line:\n'.format(ind=indentation) +\
               '{ind}|--Start: {s}\n'.format(s=self.start.pretty_print(), ind=indentation) +\
               '{ind}`--End: {e}\n'.format(e=self.end.pretty_print(), ind=indentation)

    def length(self):
        return self.to_vector().length()

    def to_points(self):
        return [self.start, self.end]

    def to_vector(self, reverse=False):
        if reverse:
            return Vector(x=self.start.x - self.end.x,
                          y=self.start.y - self.end.y,
                          z=self.start.z - self.end.z)
        else:
            return Vector(x=self.end.x - self.start.x,
                          y=self.end.y - self.start.y,
                          z=self.end.z - self.start.z)

    def midpoint(self) -> Point:
        return Point(
            x=(self.start.x + self.end.x) / 2,
            y=(self.start.y + self.end.y) / 2,
            z=(self.start.z + self.end.z) / 2,
        )

    def __eq__(self, other):
        if self.start == other.start and self.end == other.end:
            return True
        elif self.start == other.end and self.end == other.start:
            return True
        else:
            return False

    def to_ray(self) -> Ray:
        return Ray(
            vector=self.to_vector(),
            point=self.start
        )

    def flip(self) -> 'Line':
        return Line(start=self.end, end=self.start)


class Rectangle:

    def __init__(self, side: Line, external_point: Point):
        self.side = side
        self.external_point = external_point

    def __str__(self):
        return self.pretty_print()

    def pretty_print(self, indentation=''):
        return '{ind}Rectangle:\n'.format(ind=indentation) +\
               '{ind}|--Side:\n'.format(ind=indentation) +\
               self.side.pretty_print(indentation=indentation + '|   ') +\
               '{ind}`--External Point: {p}\n'.format(p=self.external_point.pretty_print(), ind=indentation)

    def height(self):
        side_vector = self.side.to_vector()
        ext_vector = self.external_point - self.side.start
        return ext_vector.cross_product(side_vector).length() / side_vector.length()

    def height_vector(self):
        s = self.side.to_vector()
        e = self.external_point - self.side.start
        proj = s * ((e * s) / (s * s))
        return e - proj

    def normal_vector(self):
        return self.side.to_vector().cross_product(self.height_vector()).unitize()

    def area(self):
        return self.side.length() * self.height()

    def to_points(self) -> List[Point]:
        """

        :return: a list of all vertices as Point instances
        """
        return self.side.to_points() + [point + self.height_vector() for point in self.side.to_points()[::-1]]

    def to_lines(self) -> List[Line]:
        """

        :return: a list of all edges as Line instances
        """
        points = self.to_points()
        return [Line(s, e) for s, e in zip(points, points[1:] + points[:1])]

    def center(self) -> Point:
        return self.side.midpoint() + (self.height_vector() / 2)


class Box:

    def __init__(self, base: Rectangle, external_point: Point):
        self.base = base
        self.external_point = external_point

    def __str__(self):
        return self.pretty_print()

    def pretty_print(self, indentation=''):
        return '{ind}Box:\n'.format(ind=indentation) +\
               '{ind}|--Base:\n'.format(ind=indentation) +\
               self.base.pretty_print(indentation=indentation + '|   ') +\
               '{ind}`--External Point: {p}\n'.format(p=self.external_point.pretty_print(), ind=indentation)

    def height(self):
        external_vector = self.external_point - self.base.side.start
        return external_vector * self.base.normal_vector()

    def height_vector(self) -> Vector:
        return self.base.normal_vector() * self.height()

    def to_rects(self) -> List[Rectangle]:
        """

        :return: a list of all faces of the box as Rectangle instances [bottom, sides..., top]
        """
        return [self.base] + [Rectangle(s, move(s.start, self.height_vector())) for s in self.base.to_lines()] +\
               [move(self.base, self.height_vector())]


class Face:
    """
    General type of face with any number of points
    Face is treated as the projection of its points to te plane defined by
        the first 2 points and the last point in the list of vertices
    """

    def __init__(self, points: List[Point]):
        self.vertices = points

    def __str__(self):
        return self.pretty_print()

    def pretty_print(self, indentation=''):
        return '{ind}Face:\n'.format(ind=indentation) +\
               ''.join([
                   '{ind}|--{p}\n'.format(p=po.pretty_print(), ind=indentation)
                   for po in self.vertices[:-1]
               ]) + \
               '{ind}`--{p}\n'.format(p=self.vertices[-1].pretty_print(), ind=indentation)

    def normal_vector(self) -> Vector:
        """
        Normal vector of the projection plane of the face
        If we see the vertices in counter-clockwise order, the normal
            is pointing towards us

        Note: we assume VertexEntryDirection == "CounterClockWise" in the idf
        Note: if vertices are in random order we don't know what will happen :-)

        :return: Vector
        """

        # TODO normal should be flipped if the three points represent a concave edge

        # look for two lines in the face that are not parallel
        for i in range(len(self.vertices)):
            vector1 = self.vertices[i+1] - self.vertices[0]
            vector2 = self.vertices[i+2] - self.vertices[0]
            normal = vector1.cross_product(vector2)
            if normal != Vector(0, 0, 0):
                return normal.unitize()

    def area(self, signed=False) -> float:
        """
        returns the area of the specified surface
        method described here: http://geomalgorithms.com/a01-_area.html

        :return: area of the face
        """

        # close the loop of vertices without modifying the object itself
        point_vectors = [Vector(v.x, v.y, v.z) for v in self.vertices]
        # add the first point
        point_vectors += point_vectors[:1]
        normal_vector = self.normal_vector()
        area = 0
        for point_count in range(0, len(point_vectors) - 1):
            area += normal_vector.scalar_product(
                point_vectors[point_count].cross_product(point_vectors[point_count + 1]))
        area /= 2
        if signed:
            return area
        else:
            return abs(area)

    def perimeter(self) -> float:
        return sum([side.length() for side in self.to_lines()])

    def to_lines(self) -> List[Line]:
        return [Line(s, e) for s, e in zip(self.vertices, self.vertices[1:] + self.vertices[:1])]

    def __eq__(self, other):
        if self.vertices[0] in other.vertices:
            start_index = other.vertices.index(self.vertices[0])
            if self.vertices == other.vertices[start_index:] + other.vertices[:start_index]:
                return True
            elif self.vertices == other.vertices[start_index::-1] + other.vertices[:start_index:-1]:
                return True
            else:
                return False
        else:
            return False

    def centroid(self) -> Point:
        # https://math.stackexchange.com/questions/90463/how-can-i-calculate-the-centroid-of-polygon
        # triangulation with signed areas and centroids
        start_corner = self.vertices[0]
        triangle_centroids = []
        areas = []
        for k in range(len(self.vertices) - 2):
            # get vectors from first corner point pointing to next two corner points
            a_k = self.vertices[k + 1] - start_corner
            a_l = self.vertices[k + 2] - start_corner
            # get centroid of the triangle between the two vectors
            triangle_centroids.append(start_corner + (a_k + a_l) / 3)
            # get signed area of the triangle
            areas.append(self.normal_vector() * a_k.cross_product(a_l) / 2)
        # total area
        area = sum(areas)
        # return weighted average of centroids (centroid of face)
        return Point(
            x=sum([c.x * w for c, w in zip(triangle_centroids, areas)]) / area,
            y=sum([c.y * w for c, w in zip(triangle_centroids, areas)]) / area,
            z=sum([c.z * w for c, w in zip(triangle_centroids, areas)]) / area,
        )

    def to_plane(self) -> Plane:
        return Plane(
            normal=self.normal_vector(),
            point=self.vertices[0]
        )


def move(obj: Union[Point, Line, Rectangle, Box, Face], vector: Vector, inplace=False):
    if isinstance(obj, Point):
        return obj + vector
    else:
        if inplace:
            new_obj = obj
        else:
            new_obj = copy.deepcopy(obj)
        for param, val in new_obj.__dict__.items():
            if isinstance(val, (Point, Line, Rectangle, Box, Face)):
                # love recursion
                new_obj.__dict__[param] = move(val, vector)
            elif isinstance(val, list):
                new_obj.__dict__[param] = [move(p, vector) for p in val]
        return new_obj


def rotate_xy(obj: Union[Point, Line, Rectangle, Box, Face], angle: float,
              center: Point = Point(0, 0, 0), inplace=False):
    """
    Rotate objects in the xy plane (around z axis)

    :param obj: object to rotate
    :param angle: angle to rotate with
    :param center: center to rotate around
    :param inplace: set True to modify the object instance itself
    :return: rotated object
    """
    if isinstance(obj, Point):
        # move point to origin
        obj_origin = move(obj, Point(0, 0, 0) - center)
        # apply rotation around origin
        new_point = Point(
            x=obj_origin.x * math.cos(math.radians(angle)) - obj_origin.y * math.sin(math.radians(angle)),
            y=obj_origin.x * math.sin(math.radians(angle)) + obj_origin.y * math.cos(math.radians(angle)),
            z=obj_origin.z
        )
        # move back
        return move(new_point, center - Point(0, 0, 0))
    else:
        if inplace:
            new_obj = obj
        else:
            new_obj = copy.deepcopy(obj)
        for param, val in new_obj.__dict__.items():
            if isinstance(val, (Point, Line, Rectangle, Box, Face)):
                # love recursion
                new_obj.__dict__[param] = rotate_xy(val, angle, center)
            elif isinstance(val, list):
                new_obj.__dict__[param] = [rotate_xy(p, angle, center) for p in val]
        return new_obj
