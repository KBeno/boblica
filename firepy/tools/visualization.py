from typing import Union, List, Tuple, Mapping, MutableMapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
from eppy.modeleditor import IDF

from firepy.calculation.cost import CostCalculation
from firepy.model.geometry import Point, Vector, Line, Ray, Plane, Rectangle, Box, Face
from firepy.model.building import BuildingSurface, FenestrationSurface, NonZoneSurface, Zone, Building, Ref
from firepy.model.building import ObjectLibrary, Construction

from firepy.calculation.lca import LCACalculation

Color = Union[str, Tuple[int, int, int, float], Tuple[float, float, float, float]]


class GeometryViewer:
    def __init__(self):
        self.figure = go.Figure()

    def view(self):
        self.figure.update_layout(
            scene_aspectmode='data',  # for equal axis aspect ratio
            showlegend=False
        )
        self.figure.show()

    def add(self, obj: Union[List, Point, Vector, Line, Rectangle, Box, Face], **kwargs):
        if isinstance(obj, List):
            for e in obj:
                self.add(e, **kwargs)
        elif isinstance(obj, Point):
            self.point(obj, **kwargs)
        elif isinstance(obj, Vector):
            self.vector(obj, **kwargs)
        elif isinstance(obj, Line):
            self.line(obj, **kwargs)
        elif isinstance(obj, Rectangle):
            self.rectangle(obj, **kwargs)
        elif isinstance(obj, Box):
            self.box(obj, **kwargs)
        elif isinstance(obj, Face):
            self.face(obj, **kwargs)

    @staticmethod
    def eval_color(color: Color):
        if isinstance(color, Tuple):
            return 'rgba({}, {}, {}, {})'.format(color[0], color[1], color[2], color[3])
        else:  # str
            return color

    def point(self, point: Point,
              color: Color = 'darkgreen',
              tag: str = "",
              text_only=False):
        if tag != "":
            if text_only:
                mode = 'text'
                text_position = 'middle center'
            else:
                mode = 'markers+text'
                text_position = 'top center'
        else:
            mode = 'markers'
            text_position = 'top center'
        trace = go.Scatter3d(
            x=[point.x],
            y=[point.y],
            z=[point.z],
            text=tag,
            textposition=text_position,
            mode=mode,
            marker=dict(
                color=self.eval_color(color)
            )
        )
        self.figure.add_trace(trace)

    def vector(self, vector: Vector, start: Point = Point(0, 0, 0),
               scale: float = 1,
               color='blue'):
        end = start + vector * scale
        trace_line = go.Scatter3d(
            x=[start.x, end.x],
            y=[start.y, end.y],
            z=[start.z, end.z],
            mode='lines',
            line=dict(
                width=2,
                color=self.eval_color(color)
            )
        )
        trace_end = go.Scatter3d(
            x=[end.x],
            y=[end.y],
            z=[end.z],
            mode='markers',
            marker=dict(
                symbol='circle',
                color=self.eval_color(color),
                size=4
            )
        )
        self.figure.add_trace(trace_line)
        self.figure.add_trace(trace_end)

    def line(self, line: Line,
             color='darkred'):
        trace = go.Scatter3d(
            x=[line.start.x, line.end.x],
            y=[line.start.y, line.end.y],
            z=[line.start.z, line.end.z],
            mode='lines',
            line=dict(
                width=2,
                color=self.eval_color(color)
            )
        )
        self.figure.add_trace(trace)

    def rectangle(self, rect: Rectangle,
                  lines: bool = True,
                  face: bool = False):
        if lines:
            for line in rect.to_lines():
                self.line(line)
        if face:
            trace = go.Mesh3d(
                x=[p.x for p in rect.to_points()],
                y=[p.y for p in rect.to_points()],
                z=[p.z for p in rect.to_points()],
            )
            self.figure.add_trace(trace)

    def box(self, box: Box, lines: bool = True, face: bool = False):
        for rect in box.to_rects():
            self.rectangle(rect, lines=lines, face=face)

    def face(self, fac: Face,
             lines: bool = True,
             face: bool = False,
             triangulation: Tuple[List[int], List[int], List[int]] = None,
             line_color='darkred',
             face_color='lightblue',
             opacity=1):
        if lines:
            for line in fac.to_lines():
                self.line(line, color=self.eval_color(line_color))
        if face:
            # get the closest axis to the normal vector of the face
            normal = fac.normal_vector()
            xyz = [abs(normal.x), abs(normal.y), abs(normal.z)]
            delaunay_axis = 'xyz'[xyz.index(max(xyz))]
            mesh_dict = dict(
                x=[p.x for p in fac.vertices],
                y=[p.y for p in fac.vertices],
                z=[p.z for p in fac.vertices],
                color=self.eval_color(face_color),
                delaunayaxis=delaunay_axis,
                opacity=opacity
            )
            if triangulation is not None:
                mesh_dict.update(dict(
                    i=triangulation[0],
                    j=triangulation[1],
                    k=triangulation[2]
                ))
            trace = go.Mesh3d(**mesh_dict)
            self.figure.add_trace(trace)


class BuildingViewer(GeometryViewer):

    def view(self):
        self.format_layout()
        self.figure.show()

    def format_layout(self):
        self.figure.update_layout(
            scene=dict(
                aspectmode='data',  # for equal axis aspect ratio
                xaxis=dict(
                    showgrid=False,
                    showbackground=False,
                    showline=True,
                    linewidth=2,
                    linecolor='red',
                    showaxeslabels=False,
                    showticklabels=False
                ),
                yaxis=dict(
                    showgrid=False,
                    showbackground=False,
                    showline=True,
                    linewidth=2,
                    linecolor='green',
                    showaxeslabels=False,
                    showticklabels=False
                ),
                zaxis=dict(
                    showgrid=False,
                    showbackground=False,
                    showline=True,
                    linewidth=2,
                    linecolor='blue',
                    showaxeslabels=False,
                    showticklabels=False
                )
            ),
            showlegend=False
        )

    def add(self, obj: Union[List, Point, Vector, Line, Rectangle, Box, Face,
                             BuildingSurface, FenestrationSurface, NonZoneSurface, Zone, Building], **kwargs):
        if isinstance(obj, List):
            for e in obj:
                self.add(e, **kwargs)
        elif isinstance(obj, BuildingSurface):
            self.building_surface(obj, **kwargs)
        elif isinstance(obj, FenestrationSurface):
            self.fenestration_surface(obj, **kwargs)
        elif isinstance(obj, NonZoneSurface):
            self.non_zone_surface(obj, **kwargs)
        elif isinstance(obj, Zone):
            self.zone(obj, **kwargs)
        elif isinstance(obj, Building):
            self.building(obj, **kwargs)
        else:
            super().add(obj, **kwargs)

    def building_surface(self, bs: BuildingSurface,
                         lines: bool = True,
                         face: bool = True,
                         line_color: Color = 'grey',
                         face_color: Color = 'darkorange',
                         opacity=1, tag: str = None,
                         fen_lines: bool = True,
                         fen_face: bool = True,
                         fen_line_color: Color = 'grey',
                         fen_face_color: Color = 'darkcyan',
                         fen_opacity=0.3,
                         fen_tag: str = None):
        punched_face, ijk = self.get_punched_geometry(bs)
        super().add(punched_face,
                    triangulation=ijk,
                    lines=False,
                    face=face,
                    face_color=face_color,
                    opacity=opacity)
        super().add(bs, lines=lines,
                    face=False,
                    line_color=line_color)

        if tag is not None:
            if tag in bs.__dict__.keys():
                if isinstance(bs.__dict__[tag], str):
                    self.add(bs.centroid(), tag=bs.__dict__[tag], text_only=True)

        for window in bs.Fenestration:
            self.add(window,
                     lines=fen_lines,
                     face=fen_face,
                     line_color=fen_line_color,
                     face_color=fen_face_color,
                     opacity=fen_opacity,
                     tag=fen_tag)

    def fenestration_surface(self, fs: FenestrationSurface,
                             lines: bool = True,
                             face: bool = True,
                             line_color: Color = 'grey',
                             face_color: Color = 'darkcyan',
                             opacity=0.3, tag: str = None):
        super().add(fs,
                    lines=lines,
                    face=face,
                    line_color=line_color,
                    face_color=face_color,
                    opacity=opacity)
        if tag is not None:
            if tag in fs.__dict__.keys():
                if isinstance(fs.__dict__[tag], str):
                    self.add(fs.centroid(), tag=fs.__dict__[tag], text_only=True)
                elif isinstance(fs.__dict__[tag], Ref):
                    text = str(fs.__dict__[tag])
                    self.add(fs.centroid(), tag=text, text_only=True)

    def non_zone_surface(self, nzs: NonZoneSurface,
                         lines: bool = True,
                         face: bool = True,
                         line_color: Color = 'darkgreen',
                         face_color: Color = 'darkcyan'):
        super().add(nzs,
                    lines=lines,
                    face=face,
                    line_color=line_color,
                    face_color=face_color)

    def zone(self, zone: Zone,
             lines: bool = True,
             face: bool = True,
             line_color: Color = 'grey',
             opacity=1,
             face_colors: Union[Mapping[str, Color], Color] = 'darkorange',
             fen_lines: bool = True,
             fen_face: bool = True,
             fen_line_color: Color = 'grey',
             fen_face_color: Color = 'darkcyan',
             fen_opacity=0.3,
             tag: str = None,
             fen_tag: str = None):

        for bs in zone.BuildingSurfaces:
            if isinstance(face_colors, Mapping):
                if bs.SurfaceType.lower() in face_colors.keys():
                    face_color = face_colors[bs.SurfaceType.lower()]
                else:
                    face_color = 'darkorange'
            else:
                face_color = 'darkorange'
            if bs.OutsideBoundaryCondition != 'outdoors':
                bs_opacity = opacity * 0.2
            else:
                bs_opacity = opacity
            self.add(bs,
                     lines=lines,
                     face=face,
                     line_color=line_color,
                     face_color=face_color,
                     opacity=bs_opacity,
                     tag=tag,
                     fen_lines=fen_lines,
                     fen_face=fen_face,
                     fen_line_color=fen_line_color,
                     fen_face_color=fen_face_color,
                     fen_opacity=fen_opacity,
                     fen_tag=fen_tag)

    def building(self, building: Building,
                 lines: bool = True,
                 face: bool = True,
                 line_color: Color = 'grey',
                 opacity: float = 1,
                 face_colors: Union[Mapping[str, Color], Color] = 'darkorange',
                 fen_lines: bool = True,
                 fen_face: bool = True,
                 fen_line_color: Color = 'grey',
                 fen_face_color: Color = 'darkcyan',
                 fen_opacity=0.3,
                 tag: str = None,
                 fen_tag: str = None):
        for zone in building.Zones:
            self.add(zone,
                     lines=lines,
                     face=face,
                     line_color=line_color,
                     opacity=opacity,
                     face_colors=face_colors,
                     fen_lines=fen_lines,
                     fen_face=fen_face,
                     fen_line_color=fen_line_color,
                     fen_face_color=fen_face_color,
                     fen_opacity=fen_opacity,
                     tag=tag,
                     fen_tag=fen_tag)
        for nzs in building.NonZoneSurfaces:
            # TODO add function parameters of nonzonesurfaces in the function
            self.add(nzs)

    @staticmethod
    def get_punched_geometry(bs: BuildingSurface):
        if len(bs.Fenestration) == 0:
            return bs, None

        # get lowest and highest side optionally with the corresponding vector from the centroid to the midpoint
        def get_x_sides(surface: Union[BuildingSurface, FenestrationSurface],
                        slice_vectors=False) -> Union[Tuple[Line, Line], Tuple[Line, Line, Vector, Vector]]:
            # get vector closest to vertical from vectors pointing from the centroid of the face to the side midpoints
            sides = surface.to_lines()
            slice_directions = [side.midpoint() - surface.centroid() for side in sides]

            # get index of lowest and highest side of the surface and the corresponding vector
            angles_to_vertical = [direction.angle(Vector(0, 0, -1)) for direction in slice_directions]
            min_index = angles_to_vertical.index(min(angles_to_vertical))
            max_index = angles_to_vertical.index(max(angles_to_vertical))

            slice_direction_low = slice_directions[min_index]
            slice_direction_high = slice_directions[max_index]
            lowest_side = sides[min_index]
            highest_side = sides[max_index]
            if slice_vectors:
                return lowest_side, highest_side, slice_direction_low, slice_direction_high
            else:
                return lowest_side, highest_side

        # order windows horizontally
        def window_sorter(w: FenestrationSurface) -> float:
            # a vector pointing up which the height of corresponds to the distance of the window centroid
            # from the building surface centroid
            vector = bs.normal().cross_product(w.centroid() - bs.centroid())
            # we only need the information on the z coordinate of the vector to sort by
            return vector.z

        low_side, high_side, slice_low, slice_high = get_x_sides(bs, slice_vectors=True)
        windows = sorted(bs.Fenestration, key=window_sorter)

        # list of points on the sides (surface edge points and window points
        points = []

        i_list = []
        j_list = []
        k_list = []

        def add_triangle(point_index: int, a: int, b: int, c: int):
            i_list.append(point_index - a)
            j_list.append(point_index - b)
            k_list.append(point_index - c)

        # add first three points (left side of building surface)
        left_midpoint = Line(low_side.start, high_side.end).midpoint()
        points.extend([low_side.start, left_midpoint, high_side.end])
        # initiate point index
        pi = 2  # first three points added above

        for i, window in enumerate(windows):
            # check if window and building surface are facing the same direction
            if bs.normal() * window.normal() < 0:
                # flip window
                window.vertices = window.vertices[::-1]

            # get points for lower side
            low_side_w, high_side_w = get_x_sides(window)

            # get slicing planes
            slicing_plane_low = Plane(normal=bs.normal().cross_product(slice_low), point=low_side_w.midpoint())
            slicing_plane_high = Plane(normal=bs.normal().cross_product(slice_high), point=high_side_w.midpoint())

            # get slicing point of the sides
            slice_point_low = slicing_plane_low.intersect(low_side.to_ray())
            slice_point_high = slicing_plane_high.intersect(high_side.to_ray())

            # add points to the list
            points.extend([high_side_w.end, low_side_w.start, slice_point_low,
                           low_side_w.end, high_side_w.start, slice_point_high])
            pi += 6

            # define triangles by indices
            add_triangle(pi, 8, 4, 7)  # A
            add_triangle(pi, 7, 4, 5)  # B
            add_triangle(pi, 7, 5, 6)  # C
            add_triangle(pi, 6, 5, 0)  # D
            add_triangle(pi, 0, 5, 1)  # E
            add_triangle(pi, 8, 3, 4)  # F
            add_triangle(pi, 4, 3, 2)  # G

            # if this is not the last window, add mid-window points and triangles
            try:
                # get slicing plane between windows
                cent_line = Line(window.centroid(), windows[i+1].centroid())

            except IndexError:
                # if we cannot get the center line, then this is the
                # last window so add points on the right side of the building surface
                right_midpoint = Line(low_side.end, high_side.start).midpoint()
                points.extend([low_side.end, right_midpoint, high_side.start])
                pi += 3

            else:
                # if we can get the center line, then this is not the last window
                # so create the slicing plane
                mid_slice_plane = Plane(normal=cent_line.to_vector(), point=cent_line.midpoint())

                # get slicing points
                mid_slice_point_low = mid_slice_plane.intersect(low_side.to_ray())
                mid_slice_point_high = mid_slice_plane.intersect(high_side.to_ray())

                # add points to the list
                points.extend([mid_slice_point_low, cent_line.midpoint(), mid_slice_point_high])
                pi += 3

            finally:
                # add triangles with the new points
                add_triangle(pi, 6, 2, 5)  # A
                add_triangle(pi, 5, 2, 1)  # B
                add_triangle(pi, 4, 5, 1)  # C
                add_triangle(pi, 4, 1, 0)  # D
                add_triangle(pi, 4, 0, 3)  # E

        return Face(points=points), (i_list, j_list, k_list)


class SimpleViewer:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.points = []
        self.tags = []

    @staticmethod
    def eval_color(color: Color, opacity: float = None):
        if isinstance(color, Tuple):
            for i in range(3):
                if color[i] > 1:
                    return (color[0] / 255, color[1] / 255, color[2] / 255, color[3])
            return color
        else:  # str
            return color

    def view(self):
        # create 3d figure
        points = np.array(self.points)
        self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0)

        # Create cubic bounding box to simulate equal aspect ratio

        # Get the largest size
        x_length = points[:, 0].max() - points[:, 0].min()
        y_width = points[:, 1].max() - points[:, 1].min()
        z_height = points[:, 2].max() - points[:, 2].min()

        # Create coordinates
        max_range = max(x_length, y_width, z_height)
        xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][
            0].flatten() + 0.5 * max_range
        yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][
            1].flatten() + -0.5 * max_range
        zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][
            2].flatten() + 0.5 * max_range

        # Plot box
        for x, y, z in zip(xb, yb, zb):
            self.ax.plot([x], [y], [z], 'w')

        # Add labels
        for tag in self.tags:
            self.ax.text(*tag)

        # discard the view of axes
        plt.grid(False)
        plt.axis('off')

        plt.show()

    def add(self, obj: Union[List, Point, Vector, Line, Rectangle, Box, Face,
                             BuildingSurface, FenestrationSurface, NonZoneSurface, Zone, Building], **kwargs):
        if isinstance(obj, List):
            for e in obj:
                self.add(e, **kwargs)
        elif isinstance(obj, BuildingSurface):
            self.building_surface(obj, **kwargs)
        elif isinstance(obj, FenestrationSurface):
            self.fenestration_surface(obj, **kwargs)
        elif isinstance(obj, NonZoneSurface):
            self.non_zone_surface(obj, **kwargs)
        elif isinstance(obj, Zone):
            self.zone(obj, **kwargs)
        elif isinstance(obj, Building):
            self.building(obj, **kwargs)
        # TODO geometry view

    def surface(self, surf: Union[FenestrationSurface, BuildingSurface, NonZoneSurface],
                lines: bool = True,
                face: bool = True,
                line_color: Color = 'b',
                face_color: Color = (1, 1, 0, 0.2),
                opacity=None, tag: str = None):

        surf_vertices = []
        # create a collection of faces by listing all vertices of them
        surf_vertices.append([vertex.coordinates() for vertex in surf.vertices])

        # add the points for the scatter plot
        for vertex in surf.vertices:
            self.points.append(vertex.coordinates())

        if lines:
            lw = 0.5
        else:
            lw = 0

        surf_ext_faces = Poly3DCollection(surf_vertices, linewidths=lw, edgecolors=self.eval_color(line_color))
        # we plot this only for the lines so set the face color to transparent
        surf_ext_faces.set_facecolor((0, 0, 1, 0))

        if face:
            if isinstance(surf, BuildingSurface):
                face, ijk = BuildingViewer.get_punched_geometry(surf)
                if ijk is not None:
                    triangles = []
                    for i, j, k in zip(ijk[0], ijk[1], ijk[2]):
                        p_i = face.vertices[i].coordinates()
                        p_j = face.vertices[j].coordinates()
                        p_k = face.vertices[k].coordinates()
                        triangles.append([p_i, p_j, p_k])
                    triangulated = Poly3DCollection(triangles, linewidths=0, edgecolors=self.eval_color(face_color))

                    triangulated.set_facecolor(self.eval_color(face_color))
                    if opacity is not None:
                        triangulated.set_alpha(float(opacity))
                    self.ax.add_collection(triangulated)
                else:
                    surf_ext_faces.set_facecolor(self.eval_color(face_color))
                    if opacity is not None:
                        surf_ext_faces.set_alpha(float(opacity))

            else:
                surf_ext_faces.set_facecolor(self.eval_color(face_color))
                if opacity is not None:
                    surf_ext_faces.set_alpha(float(opacity))

        self.ax.add_collection(surf_ext_faces)

        if tag is not None:
            if tag in surf.__dict__.keys():
                if isinstance(surf.__dict__[tag], str):
                    c = surf.centroid()
                    self.tags.append((c.x, c.y, c.z, surf.__dict__[tag]))

    def building_surface(self, bs: BuildingSurface,
                         lines: bool = True,
                         face: bool = True,
                         line_color: Color = 'k',
                         face_color: Color = (1, 0.55, 0, 0.2),
                         opacity: float = 1,
                         tag: str = None,
                         fen_lines: bool = True,
                         fen_face: bool = True,
                         fen_line_color: Color = 'b',
                         fen_face_color: Color = (0, 1, 1, 0.7),
                         fen_opacity=0.7,
                         fen_tag: str = None):

        self.surface(surf=bs, lines=lines, face=face, line_color=line_color, face_color=face_color, opacity=opacity,
                     tag=tag)

        # add window faces to the window collection
        for win in bs.Fenestration:
            self.add(win, lines=fen_lines, face=fen_face, line_color=fen_line_color, face_color=fen_face_color,
                     opacity=fen_opacity, tag=fen_tag)

    def fenestration_surface(self, fs: FenestrationSurface,
                             lines: bool = True,
                             face: bool = True,
                             line_color: Color = 'b',
                             face_color: Color = (0, 1, 1, 0.2),
                             opacity=None, tag: str = None):
        self.surface(surf=fs, lines=lines, face=face, line_color=line_color, face_color=face_color, opacity=opacity,
                     tag=tag)

    def non_zone_surface(self, nzs: NonZoneSurface,
                         lines: bool = True,
                         face: bool = True,
                         line_color: Color = 'k',
                         face_color: Color = (0, 1, 0, 0.2),
                         opacity=None, tag: str = None):
        self.surface(surf=nzs, lines=lines, face=face, line_color=line_color, face_color=face_color, opacity=opacity,
                     tag=tag)

    def zone(self, zone: Zone,
             lines: bool = True,
             face: bool = True,
             line_color: Color = 'k',
             opacity: float = 1,
             face_colors: Union[Mapping[str, Color], Color] = (1, 0.55, 0, 0.2),
             fen_lines: bool = True,
             fen_face: bool = True,
             fen_line_color: Color = 'k',
             fen_face_color: Color = (0, 1, 1, 0.7),
             fen_opacity=1,
             tag: str = None,
             fen_tag: str = None,
             fade_internal: bool = True):

        for surface in zone.BuildingSurfaces:
            if surface.OutsideBoundaryCondition.lower() == 'outdoors':
                if isinstance(face_colors, Mapping):
                    face_color = face_colors[surface.IuId]
                else:
                    face_color = face_colors
                self.add(surface, lines=lines, face=face, line_color=line_color, face_color=face_color,
                         opacity=opacity, tag=tag, fen_lines=fen_lines, fen_face=fen_face,
                         fen_line_color=fen_line_color, fen_face_color=fen_face_color, fen_opacity=fen_opacity,
                         fen_tag=fen_tag)
            else:
                if fade_internal:
                    internal_opacity = 0.1
                    internal_line_color = 'gray'
                else:
                    internal_opacity = opacity
                    internal_line_color = line_color
                if isinstance(face_colors, Mapping):
                    face_color = face_colors[surface.IuId]
                else:
                    face_color = face_colors
                self.add(surface, lines=lines, face=face, line_color=internal_line_color, face_color=face_color,
                         opacity=internal_opacity, tag=tag, fen_lines=fen_lines, fen_face=fen_face,
                         fen_line_color=fen_line_color, fen_face_color=fen_face_color, fen_opacity=fen_opacity,
                         fen_tag=fen_tag)

    def building(self, building: Building,
             lines: bool = True,
             face: bool = True,
             line_color: Color = 'k',
             opacity: float = 1,
             face_colors: Union[Mapping[str, Color], Color] = (1, 0.55, 0, 0.2),
             fen_lines: bool = True,
             fen_face: bool = True,
             fen_line_color: Color = 'k',
             fen_face_color: Color = (0, 1, 1, 0.7),
             fen_opacity=1,
             tag: str = None,
             fen_tag: str = None,
             fade_internal: bool = True):

        for zone in building.Zones:
            self.add(zone, lines=lines, face=face, line_color=line_color, face_colors=face_colors,
                     opacity=opacity, tag=tag, fen_lines=fen_lines, fen_face=fen_face,
                     fen_line_color=fen_line_color, fen_face_color=fen_face_color, fen_opacity=fen_opacity,
                     fen_tag=fen_tag, fade_internal=fade_internal)

        for surface in building.NonZoneSurfaces:
            self.add(surface)


class ResultViewer:
    """
    A class to visualize the results of an LCA or Cost calculation
    """

    def __init__(self, calculation: Union[LCACalculation, CostCalculation]):
        self.calculation = calculation

    def sunburst(self, model: Union[Building, Construction], indicator: str = None,
                 library: ObjectLibrary = None, cutoff: float = 0.01) -> go.Figure:
        """
        Create sunburst diagram for cost or impact of a Building or a Construction
        :param model:
        :param indicator: only needed for impact calculation
        :param library: only needed for constructions
        :param cutoff: the ratio of non-displayed values (default 0.01 -# impact less than 1% will not be displayed)
        :return:
        """
        if isinstance(self.calculation, LCACalculation):
            if indicator is None:
                message = 'Please specify indicator for to create sunburst diagram for. '
                message += 'Options: ' + ', '.join(self.calculation.impact_categories)
                raise Exception('Please specify indicator for to create sunburst diagram for')
            return self.__sunburst_lca(model=model, indicator=indicator, library=library, cutoff=cutoff)

        elif isinstance(self.calculation, CostCalculation):
            return self.__sunburst_cost(model=model, library=library, cutoff=cutoff)


    def __sunburst_lca(self, model: Union[Building, Construction], indicator: str,
                       library: ObjectLibrary = None, cutoff: float = 0.01) -> go.Figure:
        """

        :param model:
        :param indicator:
        :param library: if construction is calculated library is needed
        :param cutoff:
        :return:
        """
        import locale
        locale.setlocale(locale.LC_ALL, '')

        from firepy.calculation.lca import IMPACT_CATEGORIES, ImpactResult

        impact_unit = IMPACT_CATEGORIES[indicator]['Impact Unit']
        stage_names = self.get_stage_names()

        labels = []
        ids = []
        parents = []
        values = []

        # Create first level - total

        total_impact_result = self.calculation.calculate_impact(model)
        total_impacts = total_impact_result.impacts.loc[indicator, :]
        total_value = total_impacts.sum()

        if isinstance(model, Building):
            labels.append('{ind}<br>{val:n}<br>{un}'.format(ind=indicator, val=total_value, un=impact_unit))
        elif isinstance(model, Construction):
            labels.append('{ind}<br>{val:n}<br>{un}/{ref}'.format(
                ind=indicator, val=total_value, un=impact_unit, ref=total_impact_result.BasisUnit))
        else:
            raise Exception('Input model needs to be Building or Construction, not {t}'.format(t=type(model)))

        ids.append('Total')
        parents.append('')
        values.append(int(total_value / total_value * 1000))

        def add_by_stage(impacts: pd.Series, label: str, parent: str):
            for stage in ['A1-3', 'A4', 'A5', 'B4', 'B6', 'C1-4']:
                stage_value = impacts[stage]
                if stage_value / total_value > cutoff:
                    if parent == 'Total':
                        labels.append('{s}<br>{v:n}'.format(s=stage_names[stage], v=stage_value))
                        ids.append(stage)
                        parents.append('Total')
                        values.append(int(stage_value / total_value * 1000))
                    else:
                        labels.append('{s}<br>{v:n}'.format(s=label, v=stage_value))
                        ids.append(stage + parent + label)
                        parents.append(stage + parent)
                        values.append(int(stage_value / total_value * 1000))

        # Create second level - main life cycle stages
        add_by_stage(total_impacts, '', 'Total')

        if isinstance(model, Construction):
            if library is None:
                raise Exception('Please provide the ObjectLibrary of the model')
            life_times = self.calculation.evaluate_construction_lifetimes(model, library)
            for mat in model.Layers:
                material = library.get(mat)
                mat_imp = self.calculation.calculate_impact(material, life_time_overwrites=life_times)
                material_impact = mat_imp.impacts.loc[indicator, :]
                add_by_stage(material_impact, material.Name, '')

        if isinstance(model, Building):
            # Third level - envelope / internal / HVAC
            # Fourth level - fenestration / wall / slab // heating / cooling / lights
            # Fifth level - Materials
            envelope = ImpactResult(basis_unit='total')
            internal = ImpactResult(basis_unit='total')

            fenestration = ImpactResult(basis_unit='total')
            envelope_wall = ImpactResult(basis_unit='total')
            envelope_slab = ImpactResult(basis_unit='total')
            internal_wall = ImpactResult(basis_unit='total')
            internal_slab = ImpactResult(basis_unit='total')

            for zone in model.Zones:
                for surface in zone.BuildingSurfaces:
                    surf_impact: ImpactResult = self.calculation.calculate_impact(surface)
                    if surface.OutsideBoundaryCondition.lower() in ['outdoors', 'ground']:
                        envelope += surf_impact
                        if surface.SurfaceType.lower() == 'wall':
                            windows_impact = ImpactResult(basis_unit='total')
                            for window in surface.Fenestration:
                                window_impact: ImpactResult = self.calculation.calculate_impact(window)
                                windows_impact += window_impact
                            fenestration += windows_impact
                            envelope_wall += surf_impact - windows_impact
                        else:
                            envelope_slab += surf_impact
                    else:
                        internal += surf_impact
                        if surface.SurfaceType.lower() == 'wall':
                            internal_wall += surf_impact
                        else:
                            internal_slab += surf_impact

            # Ignore Non-zone Surfaces for now

            if not envelope.impacts.empty:
                envelope_impacts = envelope.impacts.loc[indicator, :]
                add_by_stage(envelope_impacts, 'Envelope', '')
            if not internal.impacts.empty:
                internal_impacts = internal.impacts.loc[indicator, :]
                add_by_stage(internal_impacts, 'Internal', '')

            hvac_impacts = self.calculation.calculate_impact(model.HVAC).impacts.loc[indicator, :]
            add_by_stage(hvac_impacts, 'HVAC', '')

            if not fenestration.impacts.empty:
                fenestration_impacts = fenestration.impacts.loc[indicator, :]
                add_by_stage(fenestration_impacts, 'Fenestration', 'Envelope')
            if not envelope_wall.impacts.empty:
                envelope_wall_impacts = envelope_wall.impacts.loc[indicator, :]
                add_by_stage(envelope_wall_impacts, 'Walls', 'Envelope')
            if not envelope_slab.impacts.empty:
                envelope_slab_impacts = envelope_slab.impacts.loc[indicator, :]
                add_by_stage(envelope_slab_impacts, 'Slabs', 'Envelope')

            if not internal_wall.impacts.empty:
                internal_wall_impacts = internal_wall.impacts.loc[indicator, :]
                add_by_stage(internal_wall_impacts, 'Walls', 'Internal')
            if not internal_slab.impacts.empty:
                internal_slab_impacts = internal_slab.impacts.loc[indicator, :]
                add_by_stage(internal_slab_impacts, 'Slabs', 'Internal')

            heating = self.calculation.calculate_impact(model.HVAC.Heating) * self.calculation.rsp
            cooling = self.calculation.calculate_impact(model.HVAC.Cooling) * self.calculation.rsp
            lights = self.calculation.calculate_impact(model.HVAC.Lighting) * self.calculation.rsp
            heating_impacts = heating.impacts.loc[indicator, :]
            cooling_impacts = cooling.impacts.loc[indicator, :]
            lights_impacts = lights.impacts.loc[indicator, :]

            add_by_stage(heating_impacts, 'Heating', 'HVAC')
            add_by_stage(cooling_impacts, 'Cooling', 'HVAC')
            add_by_stage(lights_impacts, 'Lights', 'HVAC')

        trace = go.Sunburst(
            labels=labels,
            ids=ids,
            parents=parents,
            values=values,
            branchvalues="total",
            # with using this attribute it is possible to add very many leafs
            # maxdepth=3
        )

        fig = go.Figure(
            data=trace
        )
        fig.update_layout(
            autosize=False,
            width=800,
            height=800,
            template='plotly_white'
        )
        return fig

    def __sunburst_cost(self, model: Union[Building, Construction],
                       library: ObjectLibrary = None, cutoff: float = 0.01) -> go.Figure:
        """

        :param model:
        :param indicator:
        :param library: if construction is calculated library is needed
        :param cutoff:
        :return:
        """
        import locale
        locale.setlocale(locale.LC_ALL, '')

        from firepy.calculation.cost import CostResult

        cost_unit = 'EUR'

        labels = []
        ids = []
        parents = []
        values = []

        # Create first level - total

        total_cost_result = self.calculation.calculate_cost(model)
        total_costs = total_cost_result.costs  # pd.Series
        total_value = total_costs.sum()

        if isinstance(model, Building):
            labels.append('{ind}<br>{val:n}<br>{un}'.format(ind='Cost', val=total_value, un=cost_unit))
        elif isinstance(model, Construction):
            labels.append('{ind}<br>{val:n}<br>{un}/{ref}'.format(
                ind='Cost', val=total_value, un=cost_unit, ref=total_cost_result.ReferenceUnit))
        else:
            raise Exception('Input model needs to be Building or Construction, not {t}'.format(t=type(model)))

        ids.append('Total')
        parents.append('')
        values.append(int(total_value / total_value * 1000))

        def add_by_stage(costs: pd.Series, label: str, parent: str):
            for stage in ['Production', 'Installation', 'Replacement', 'Operation']:
                stage_value = costs[stage]
                if stage_value / total_value > cutoff:
                    if parent == 'Total':
                        labels.append('{s}<br>{v:n}'.format(s=stage, v=stage_value))
                        ids.append(stage)
                        parents.append('Total')
                        values.append(int(stage_value / total_value * 1000))
                    else:
                        labels.append('{s}<br>{v:n}'.format(s=label, v=stage_value))
                        ids.append(stage + parent + label)
                        parents.append(stage + parent)
                        values.append(int(stage_value / total_value * 1000))

        # Create second level - main life cycle stages
        add_by_stage(total_costs, '', 'Total')

        if isinstance(model, Construction):
            if library is None:
                raise Exception('Please provide the ObjectLibrary of the model')
            life_times = self.calculation.evaluate_construction_lifetimes(model, library)
            for mat in model.Layers:
                material = library.get(mat)
                mat_cst = self.calculation.calculate_cost(material, life_time_overwrites=life_times)
                add_by_stage(mat_cst.costs, material.Name, '')

        if isinstance(model, Building):
            # Third level - envelope / internal / HVAC
            # Fourth level - fenestration / wall / slab // heating / cooling / lights
            # Fifth level - Materials
            envelope = CostResult(ref_unit='total')
            internal = CostResult(ref_unit='total')

            fenestration = CostResult(ref_unit='total')
            envelope_wall = CostResult(ref_unit='total')
            envelope_slab = CostResult(ref_unit='total')
            internal_wall = CostResult(ref_unit='total')
            internal_slab = CostResult(ref_unit='total')

            for zone in model.Zones:
                for surface in zone.BuildingSurfaces:
                    surf_cost: CostResult = self.calculation.calculate_cost(surface)
                    if surface.OutsideBoundaryCondition.lower() in ['outdoors', 'ground']:
                        envelope += surf_cost
                        if surface.SurfaceType.lower() == 'wall':
                            windows_cost = CostResult(ref_unit='total')
                            for window in surface.Fenestration:
                                window_cost: CostResult = self.calculation.calculate_cost(window)
                                windows_cost += window_cost
                            fenestration += windows_cost
                            envelope_wall += surf_cost - windows_cost
                        else:
                            envelope_slab += surf_cost
                    else:
                        internal += surf_cost
                        if surface.SurfaceType.lower() == 'wall':
                            internal_wall += surf_cost
                        else:
                            internal_slab += surf_cost

            # Ignore Non-zone Surfaces for now

            if not envelope.costs.empty:
                add_by_stage(envelope.costs, 'Envelope', '')
            if not internal.costs.empty:
                add_by_stage(internal.costs, 'Internal', '')

            hvac_costs = self.calculation.calculate_cost(model.HVAC).costs
            add_by_stage(hvac_costs, 'HVAC', '')

            if not fenestration.costs.empty:
                add_by_stage(fenestration.costs, 'Fenestration', 'Envelope')
            if not envelope_wall.costs.empty:
                add_by_stage(envelope_wall.costs, 'Walls', 'Envelope')
            if not envelope_slab.costs.empty:
                add_by_stage(envelope_slab.costs, 'Slabs', 'Envelope')

            if not internal_wall.costs.empty:
                add_by_stage(internal_wall.costs, 'Walls', 'Internal')
            if not internal_slab.costs.empty:
                add_by_stage(internal_slab.costs, 'Slabs', 'Internal')

            heating = self.calculation.calculate_cost(model.HVAC.Heating) * self.calculation.rsp
            cooling = self.calculation.calculate_cost(model.HVAC.Cooling) * self.calculation.rsp
            lights = self.calculation.calculate_cost(model.HVAC.Lighting) * self.calculation.rsp

            add_by_stage(heating.costs, 'Heating', 'HVAC')
            add_by_stage(cooling.costs, 'Cooling', 'HVAC')
            add_by_stage(lights.costs, 'Lights', 'HVAC')

        trace = go.Sunburst(
            labels=labels,
            ids=ids,
            parents=parents,
            values=values,
            branchvalues="total",
            # with using this attribute it is possible to add very many leafs
            # maxdepth=3
        )

        fig = go.Figure(
            data=trace
        )
        fig.update_layout(
            autosize=False,
            width=800,
            height=800,
            template='plotly_white'
        )
        return fig

    @staticmethod
    def get_stage_names() -> MutableMapping:
        from firepy.calculation.lca import LIFE_CYCLE_STAGES

        names = {}
        for stage_code, stage in LIFE_CYCLE_STAGES.items():
            names[stage_code] = stage['ShortName']
            for module_code, module_name in stage['Modules'].items():
                names[module_code] = module_name

        return names


class ScheduleViewer:

    @staticmethod
    def view(idf: IDF, schedule_name: str):
        """
        View all schedule profiles in an EnergyPlus compact schedule
        :param idf:  eppy IDF object
        :param schedule_name: the name of the schedule set
        :return: None
        """
        # get the schedule
        schedule = idf.getobject('Schedule:Compact'.upper(), schedule_name)

        # collect information from idf
        schedule_dict = {}
        actual_through = None
        actual_for = None
        actual_until = None
        for value in schedule.fieldvalues[3:]:
            if value.startswith('Through'):
                actual_through = value
                schedule_dict[actual_through] = {}
            elif value.startswith('For'):
                actual_for = value
                schedule_dict[actual_through][actual_for] = {}
            elif value.startswith('Until'):
                actual_until = value
                schedule_dict[actual_through][actual_for][actual_until] = None
            else:
                schedule_dict[actual_through][actual_for][actual_until] = value

        # create DataFrame from schedule
        schedule_frame = pd.DataFrame(index=[i for i in range(25)])
        for s_through, s_for in schedule_dict.items():
            for ts_for, ts_until in s_for.items():
                for_name = ts_for.split(':')[1].strip()
                for until, val in ts_until.items():
                    until_int = int(until.split(':')[1].strip())
                    schedule_frame.loc[until_int, for_name] = float(val)
        schedule_frame = schedule_frame.fillna(method='backfill')

        # create plots
        n = schedule_frame.shape[1]
        limits = (schedule_frame.stack().min(), schedule_frame.stack().max())
        fig, axs = plt.subplots(nrows=1, ncols=n)
        for i, schedule_name in enumerate(schedule_frame.columns.to_list()):
            schedule = schedule_frame[schedule_name]
            axs[i].fill_between(x=schedule.index, y1=schedule.values, step='pre', facecolor='palegoldenrod',
                                edgecolor='gray')
            axs[i].set_title(schedule_name)
            axs[i].set_xticks(schedule.index, minor=True)
            axs[i].set_xticks([0, 6, 12, 18, 24], minor=False)
            axs[i].set_xticklabels(['0:00', '6:00', '12:00', '18:00', '24:00'])
            axs[i].set_ylim(limits[0], limits[1] * 1.05)
        fig.set_size_inches(3.5 * n, 2.2)
        plt.show()

    @staticmethod
    def list(idf: IDF) -> List[str]:
        """
        List all compact schedules in the idf
        :param idf: eppy IDF
        :return: list of names
        """
        schedules = idf.idfobjects['Schedule:Compact']
        return [sch.Name for sch in schedules]


class ConstructionViewer:

    def __init__(self):
        self.colors = {}
        self.names = {}

    @staticmethod
    def list_materials(library: ObjectLibrary) -> List[str]:
        return [mat.Name for mat in library.opaque_materials.values()]

    def view(self, construction_name: str, library: ObjectLibrary, flip: bool = False):
        # TODO WindowMaterial
        if library.default_key != 'Name':
            library.change_key(to='Name')
        construction = library.constructions[construction_name]

        color_index = 0
        fig, ax = plt.subplots()
        spacing = 0
        min_text_pos = 0.0
        ax.plot([0, 1], [0, 0], color='gray', linewidth=1)
        if not flip:
            layers = construction.Layers[::-1]
        else:
            layers = construction.Layers[::1]
        for material in layers:
            material = library.get(material)

            used_colors = [c for c in self.colors.values()]
            if material.Name not in self.colors:
                color = 'C{}'.format(color_index)
                while color in used_colors:
                    color_index += 1
                    color = 'C{}'.format(color_index)
                color_index += 1
                self.colors[material.Name] = color
            else:
                color = self.colors[material.Name]

            if material.Name not in self.names:
                name = material.Name
                self.names[material.Name] = name
            else:
                name = self.names[material.Name]

            bottom = spacing
            top = spacing + material.Thickness

            text_pos = max((bottom + top) / 2, min_text_pos)
            min_text_pos = text_pos + 0.04
            ax.text(x=0.2, y=text_pos, s='{t} cm'.format(t=material.Thickness * 100),
                    verticalalignment='center', color=color, horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='white', boxstyle=('Round, pad=0.15')))
            ax.text(x=0.25, y=text_pos, s=name, verticalalignment='center', color=color,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='white', boxstyle=('Round, pad=0.15')))

            ax.plot([0, 1], [top, top], color='gray', linewidth=1)
            ax.fill_between(x=[0, 1], y1=[bottom, bottom], y2=[top, top], alpha=0.6, facecolor=color)
            spacing = top
        ax.axis('off')
        return ax
