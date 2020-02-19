from typing import Union, List, Tuple, Mapping

import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt

from firepy.model.geometry import Point, Vector, Line, Ray, Plane, Rectangle, Box, Face
from firepy.model.building import BuildingSurface, FenestrationSurface, NonZoneSurface, Zone, Building, Ref

Color = Union[str, Tuple[int, int, int, float]]


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

    def view(self):
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
        pass

    def fenestration_surface(self, fs: FenestrationSurface,
                             lines: bool = True,
                             face: bool = True,
                             line_color: Color = 'grey',
                             face_color: Color = 'darkcyan',
                             opacity=0.3, tag: str = None):
        pass

    def non_zone_surface(self, nzs: NonZoneSurface,
                         lines: bool = True,
                         face: bool = True,
                         line_color: Color = 'darkgreen',
                         face_color: Color = 'darkcyan'):
        pass

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
        pass

    def building(self, building: Building,
                 face: bool = True):

        surf_ext_vertices = []
        surf_int_vertices = []
        win_vertices = []
        points = []
        for zone in building.Zones:
            for surface in zone.BuildingSurfaces:
                # create a collection of faces by listing all vertices of them
                if surface.OutsideBoundaryCondition.lower() == 'outdoors':
                    surf_ext_vertices.append([vertex.coordinates() for vertex in surface.vertices])
                else:
                    surf_int_vertices.append([vertex.coordinates() for vertex in surface.vertices])

                # add window faces to the window collection
                for win in surface.Fenestration:
                    win_vertices.append([vertex.coordinates() for vertex in win.vertices])
                # create a separate list of the points for the scatter plot
                for vertex in surface.vertices:
                    points.append(vertex.coordinates())
        points = np.array(points)

        surf_ext_faces = Poly3DCollection(surf_ext_vertices, linewidths=0.5, edgecolors='k')
        if face:
            surf_ext_faces.set_facecolor((1, 0.55, 0, 0.2))
        else:
            surf_ext_faces.set_facecolor((0, 0, 1, 0))
        self.ax.add_collection(surf_ext_faces)

        surf_int_faces = Poly3DCollection(surf_int_vertices, linewidths=0.2, edgecolors='gray')
        if face:
            surf_int_faces.set_facecolor((1, 0.55, 0, 0.2))
        else:
            surf_int_faces.set_facecolor((0, 0, 1, 0))
        self.ax.add_collection(surf_int_faces)

        win_faces = Poly3DCollection(win_vertices, linewidths=0.5, edgecolors='b')
        if face:
            win_faces.set_facecolor((0, 1, 1, 0.2))
        else:
            win_faces.set_facecolor((0, 0, 1, 0))

        self.ax.add_collection(win_faces)

        self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0)

        # Create cubic bounding box to simulate equal aspect ratio

        # Get the largest size
        x_length = points[:, 0].max() - points[:, 0].min()
        y_width = points[:, 1].max() - points[:, 1].min()
        z_height = points[:, 2].max() - points[:, 2].min()

        # Create coordinates
        max_range = max(x_length, y_width, z_height)
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][
            0].flatten() + 0.5 * max_range
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][
            1].flatten() + -0.5 * max_range
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][
            2].flatten() + 0.5 * max_range

        # Plot box
        for xb, yb, zb in zip(Xb, Yb, Zb):
            self.ax.plot([xb], [yb], [zb], 'w')