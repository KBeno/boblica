from typing import Union, List
import copy
import math


"""
Principles:

- geometry objects are defined by the minimum required information
- Points are made of coordinates (floats), everything else is based on Points except for Vectors
"""


class Point:

    def __init__(self, x=None, y=None, z=None):
        self.x = x
        self.y = y
        self.z = z

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

    def __init__(self, x=None, y=None, z=None):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return self.pretty_print()

    def pretty_print(self, indentation=''):
        return "{ind}{x}, {y}, {z} (Vector)".format(x=self.x, y=self.y, z=self.z, ind=indentation)

    def coordinates(self):
        return self.x, self.y, self.z

    def length(self):
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
            # scalar product
            product = 0
            for xyz in [0, 1, 2]:
                product += self.coordinates()[xyz] * other.coordinates()[xyz]
            return product
        elif isinstance(other, float):
            return Vector(self.x * other, self.y * other, self.z * other)

    def angle(self, vector2):
        # angle between the instance vector and the given vector in degrees
        # always positive and smaller or equal to 180°
        return math.degrees(math.acos(self.scalar_product(vector2) / self.length() / vector2.length()))

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)


class Line:

    def __init__(self, start: Point = None, end: Point = None):
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

    def points(self):
        return [self.start, self.end]

    def to_vector(self):
        return Vector(x=self.end.x - self.start.x,
                      y=self.end.y - self.start.y,
                      z=self.end.z - self.start.z)


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
        return self.side.points() + [point + self.height_vector() for point in self.side.points()[::-1]]

    def to_lines(self) -> List[Line]:
        """

        :return: a list of all edges as Line instances
        """
        points = self.to_points()
        return [Line(s, e) for s, e in zip(points, points[1:] + points[:1])]


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

        :return: Vector
        """

        vector1 = self.vertices[1] - self.vertices[0]
        vector2 = self.vertices[-1] - self.vertices[0]

        # TODO normal should be flipped if first corner is concave

        return vector1.cross_product(vector2).unitize()

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

    def perimeter(self):
        return sum([side.length() for side in self.to_lines()])

    def to_lines(self):
        return [Line(s, e) for s, e in zip(self.vertices, self.vertices[1:] + self.vertices[:1])]


def move(obj: Union[Point, Line, Rectangle, Box], vector: Vector, inplace=False):
    if isinstance(obj, Point):
        return obj + vector
    else:
        if inplace:
            new_obj = obj
        else:
            new_obj = copy.deepcopy(obj)
        for param, val in new_obj.__dict__.items():
            if not isinstance(val, float):
                # love recursion
                new_obj.__dict__[param] = move(val, vector)
        return new_obj
