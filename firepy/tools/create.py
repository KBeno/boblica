from typing import List, Union, Mapping
import logging

import pandas as pd

from firepy.model.building import OpaqueMaterial, WindowMaterial, ShadeMaterial, BlindMaterial, Shading, Construction
from firepy.model.building import BuildingSurface, FenestrationSurface, Zone, NonZoneSurface, Building
from firepy.model.building import Ref, ObjectLibrary
from firepy.model.geometry import Rectangle, Vector, Point, Box, move

logger = logging.getLogger(__name__)


class OpaqueMaterialCreator:
    def __init__(self):
        self._opaque_materials = {}

    @property
    def opaque_materials(self) -> Mapping[str, OpaqueMaterial]:
        return self._opaque_materials

    def from_db(self, material: pd.Series, properties: pd.Series, thickness: float,
                name: str = None, roughness: str = 'Rough', thermal_absorptance: float = 0.9,
                solar_absorptance: float = 0.7, visible_absorptance: float = 0.7):

        opaque_material = OpaqueMaterial(
            name=material['Name'] if name is None else name,
            db_id=material['Dataset-ID'],
            thickness=thickness,
            conductivity=properties['HovezetesiTenyezo'],
            density=properties['Suruseg'],
            specific_heat=properties['Fajho'],
            transport_scenario=material['Transport scenario'],
            disposal_scenario=material['Disposal scenario'],
            cutting_waste=material['cutting waste'],
            life_time=material['Life Time'],
            roughness=roughness,
            thermal_absorptance=thermal_absorptance,
            solar_absorptance=solar_absorptance,
            visible_absorptance=visible_absorptance
        )
        self._opaque_materials[opaque_material.IuId] = opaque_material
        return opaque_material


class WindowMaterialCreator:
    def __init__(self):
        self._window_materials = {}

    @property
    def window_materials(self) -> Mapping[str, WindowMaterial]:
        return self._window_materials

    def from_db(self, glazing: pd.Series, frame: pd.Series, name: str = None,
                u_value: float = 1.4, g_value: float = 0.75):

        window_material = WindowMaterial(
            name='{g}_{f}'.format(g=glazing['Name'], f=frame['Name']) if name is None else name,
            typ='SimpleGlazingSystem',
            glazing_id=glazing['Dataset-ID'],
            frame_id=frame['Dataset-ID'],
            u_value=u_value,
            g_value=g_value,
            transport_scenario_glazing=glazing['Transport scenario'],
            disposal_scenario_glazing=glazing['Disposal scenario'],
            cutting_waste_glazing=glazing['cutting waste'],
            life_time_glazing=glazing['Life Time'],
            transport_scenario_frame=frame['Transport scenario'],
            disposal_scenario_frame=frame['Disposal scenario'],
            cutting_waste_frame=frame['cutting waste'],
            life_time_frame=frame['Life Time']
        )
        self._window_materials[window_material.IuId] = window_material
        return window_material


class ShadeMaterialCreator:
    def __init__(self):
        self._shade_materials = {}

    @property
    def shade_materials(self) -> Mapping[str, ShadeMaterial]:
        return self._shade_materials

    def from_db(self, material: pd.Series, name: str = None, reflectance: float = 0.8,
                transmittance: float = 0.15, emissivity: float = 0.9, thickness: float = 0.005,
                conductivity: float = 0.2, density: float = 1400, dist_to_glass: float = 0.05):

        shading_material = ShadeMaterial(
            name=material['Name'] if name is None else name,
            db_id=material['Dataset-ID'],
            reflectance=reflectance,
            transmittance=transmittance,
            emissivity=emissivity,
            thickness=thickness,
            conductivity=conductivity,
            density=density,
            distance_to_glass=dist_to_glass
        )
        self._shade_materials[shading_material.IuId] = shading_material
        return shading_material


class BlindMaterialCreator:
    def __init__(self):
        self._blind_materials = {}
    # TODO

    @property
    def blind_materials(self) -> Mapping[str, ShadeMaterial]:
        return self._blind_materials


class ShadingCreator:
    def __init__(self):
        self._shadings = {}

    @property
    def shadings(self) -> Mapping[str, Shading]:
        return self._shadings

    def by_type(self, typ: str, placing: str, material: Union[ShadeMaterial, BlindMaterial], name: str = None):
        """
        Shading options: interior or exterior shades or blinds

        :param typ: "Shade" / "Blind"
        :param placing: "Interior / "Exterior"
        :param material: a ShadingMaterial object
        :param name: optional name, if not defined, the name of the material will be assigned

        :return:  Shading
        """
        shading_factors = {
            'shade': {
                'interior': 0.47,
                'exterior': 0.13,
            },
            'blind': {
                'interior': 0.55,
                'exterior': 0.12
            }
        }

        shading = Shading(
            name=material.Name if name is None else name,
            typ=placing.lower().capitalize() + typ.lower().capitalize(),
            properties={},
            material=material.get_ref(),
            shading_factor=shading_factors[typ.lower()][placing.lower()],
            is_scheduled=True,
        )
        self._shadings[shading.IuId] = shading
        return shading


class ConstructionCreator:
    def __init__(self):
        self._constructions = {}

    @property
    def constructions(self) -> Mapping[str, Construction]:
        return self._constructions

    def from_layers(self, materials: Union[List[OpaqueMaterial], List[WindowMaterial]], name: str = None):
        construction = Construction(
            name=name if name is not None else "NoNameConstruction",
            layers=[material.get_ref() for material in materials]
        )
        self._constructions[construction.IuId] = construction
        return construction


class FenestrationCreator:

    @staticmethod
    def from_rect(rect: Rectangle, construction: Ref, name: str = None, shading: Ref = None):
        return FenestrationSurface(
            name='NoNameWindow' if name is None else name,
            vertices=rect.to_points(),
            surface_type='window',
            shading=shading,
            construction=construction
        )

    @staticmethod
    def by_ratio(building_surface: BuildingSurface, construction: Ref, win_wall_ratio: float,
                 sill_height: float, window_height: float, break_up_number: int = 1,
                 name: str = None, shading: Ref = None, frame_name: str = None) -> List[FenestrationSurface]:
        if win_wall_ratio >= 0.9:
            win_wall_ratio = 0.9
        elif win_wall_ratio <= 0:
            return []

        # get lowest point
        z_values = [p.z for p in building_surface.vertices]
        lowest_point = building_surface.vertices[z_values.index(min(z_values))]

        # get sides starting from the lowest point
        l_sides = [side for side in building_surface.to_lines() if lowest_point in side.to_points()]
        if len(l_sides) < 2:
            raise Exception("Cannot find two sides connecting to the lowest point of the surface")

        # check if lowest point is the staring vertex of the sides and create vector from it
        reverse = [side.start != lowest_point for side in l_sides]
        l_vectors = [side.to_vector(rev) for side, rev in zip(l_sides[:2], reverse)]

        # use the side closer to horizontal as base for window, the other as the side
        if l_vectors[0].angle(Vector(0, 0, 1)) > l_vectors[1].angle(Vector(0, 0, 1)):
            horizontal_side = l_vectors[0]
            vertical_side = l_vectors[1]
        else:
            vertical_side = l_vectors[0]
            horizontal_side = l_vectors[1]

        if sill_height + window_height >= vertical_side.length():
            message = 'Height of windows changed from {o:.2f} '.format(o=window_height)
            window_height = vertical_side.length() - sill_height - 0.05
            message += 'to {n:.2f} because it would exceed the wall height '.format(n=window_height)
            message += 'on surface: {sn}'.format(sn=building_surface.Name)
            logger.debug(message)

        # calculate the area
        total_window_area = building_surface.area() * win_wall_ratio

        # get the size of the windows and distance between them
        window_total_width = total_window_area / window_height

        distance_between = (horizontal_side.length() - window_total_width) / (break_up_number + 1)
        if distance_between <= 0:
            message = 'Number of windows changed from {n} '.format(n=break_up_number)
            break_up_number = 1
            distance_between = 0.05
            message += 'to {n} they would not fit on the wall '.format(n=break_up_number)
            message += 'on surface: {sn}'.format(sn=building_surface.Name)
            logger.debug(message)
        window_width = window_total_width / break_up_number

        if window_width >= horizontal_side.length():
            max_size_reached = False
            window_width = horizontal_side.length() - 0.05 - 0.05
            # if only one window and it is wider than the surface -> make window higher
            message = 'Height of windows was increased from {o:.2f} '.format(o=window_height)
            window_height = total_window_area / window_width
            message += 'to {n:.2f} '.format(n=window_height)

            if sill_height + window_height >= vertical_side.length():
                # if still too high, make the sill lower
                message += 'and height of sill was decreased from {o:.2f} '.format(o=sill_height)
                sill_height = vertical_side.length() - 0.05 - window_height
                if sill_height < 0.05:
                    sill_height = 0.05
                    window_height = vertical_side.length() - 0.05 - 0.05
                    max_size_reached = True
                message += 'to {n:.2f} '.format(n=sill_height)
            message += 'to satisfy window ratio '
            if max_size_reached:
                message = 'Maximum window size of {w} × {h} is reached '.format(w=window_width, h=window_height)
            message += 'on surface: {sn}'.format(sn=building_surface.Name)
            logger.debug(message)

        h_unit = horizontal_side.unitize()
        v_unit = vertical_side.unitize()

        def make_window_geometry_points(edge_point: Point, width: float, height: float):
            return [
                edge_point,
                edge_point + h_unit * width,
                edge_point + h_unit * width + v_unit * height,
                edge_point + v_unit * height
            ]

        # get position of the windows
        window_edge_points = [lowest_point +
                              v_unit * sill_height +
                              h_unit * distance_between * (i+1) +
                              h_unit * window_width * i
                              for i in range(break_up_number)]

        fenestration = []

        for i, p in enumerate(window_edge_points):
            fenestration.append(FenestrationSurface(
                name=building_surface.Name + 'NoNameWindow' + str(i) if name is None else name + str(i),
                vertices=make_window_geometry_points(p, window_width, window_height),
                surface_type='window',
                shading=shading,
                construction=construction,
                frame_name=frame_name
            ))

        return fenestration

    @staticmethod
    def add_fenestration(building_surface: BuildingSurface, fenestration: List[FenestrationSurface]) -> BuildingSurface:
        for window in fenestration:
            if window is not None:
                building_surface.Fenestration.append(window)
        return building_surface

    @staticmethod
    def add_fenestration_by_orientation(zone: Zone, ratio: Union[Mapping[str, float], float],
                                        construction: Union[Mapping[str, Ref], Ref],
                                        sill_height: Union[Mapping[str, float], float] = 0.9,
                                        window_height: Union[Mapping[str, float], float] = 1.5,
                                        break_up_number: Union[Mapping[str, int], int] = 1,
                                        shading: Union[Mapping[str, Union[Ref, None]], Ref, None] = None,
                                        name: str = None
                                        ) -> Zone:
        # TODO use TypedDict later here (Python 3.8)
        """
        North direction equals to positive y axis
        Windows does not fit to the surface (would be too large), window height is reduced

        :param zone: Zone to add the fenestration to
        :param ratio: a dictionary with keys: 'north', 'south', 'east', 'west' to specify the fenestration ratio or a
        float to use for all directions
        :param construction: a dictionary with keys: 'north', 'south', 'east', 'west' to specify the construction, or
        a Construction to use for all directions
        :param sill_height: a dictionary with keys: 'north', 'south', 'east', 'west' to specify each sill height, or
        a float to use for all directions
        :param window_height: a dictionary with keys: 'north', 'south', 'east', 'west' to specify each window height, or
        a float to use for all directions
        :param break_up_number: a dictionary with keys: 'north', 'south', 'east', 'west' to specify number of windows
        on each side, or a number to use for all directions
        :param shading: a dictionary with keys: 'north', 'south', 'east', 'west' to specify the shading, or a Shading
        to use for all directions or None
        :param name: optional name to describe the windows
        :return: Zone with fenestration
        """

        def create_dict(param):
            if isinstance(param, dict):
                param_dict = {
                    'North': param['north'],
                    'NorthEast': param['north'],
                    'East': param['east'],
                    'SouthEast': param['east'],
                    'South': param['south'],
                    'SouthWest': param['south'],
                    'West': param['west'],
                    'NorthWest': param['west']
                }
            else:
                param_dict = {orient: param for orient in ['North', 'NorthEast', 'East', 'SouthEast',
                                                           'South', 'SouthWest', 'West', 'NorthWest']}
            return param_dict

        ratio_dict = create_dict(ratio)
        const_dict = create_dict(construction)
        sill_dict = create_dict(sill_height)
        height_dict = create_dict(window_height)
        breakup_dict = create_dict(break_up_number)
        shading_dict = create_dict(shading)

        for surface in zone.BuildingSurfaces:
            if surface.SurfaceType.lower() == 'wall':
                ori = surface.orientation()
                dict_args = dict(
                    building_surface=surface,
                    construction=const_dict[ori],
                    win_wall_ratio=ratio_dict[ori],
                    sill_height=sill_dict[ori],
                    window_height=height_dict[ori],
                    break_up_number=breakup_dict[ori],
                    name=name + ori if name is not None else 'NoNameWindow' + ori,
                    shading=shading_dict[ori]
                )
                fenestration = FenestrationCreator.by_ratio(**dict_args)
                surface.Fenestration.extend(fenestration)

        return zone


class BuildingSurfaceCreator:

    @staticmethod
    def from_rect(rect: Rectangle, construction: Construction, name: str = None, surf_type: str = 'guess',
                  max_roof_angle: int = 60, fenestration: List[FenestrationSurface] = None, outside_bc='outdoors'):
        if surf_type == 'guess':
            # guess surface type from angle to horizontal plane
            angle = rect.normal_vector().angle(Vector(0, 0, 1))
            if max_roof_angle < angle < 180 - max_roof_angle:
                surf_type = 'WALL'
            else:
                surf_type = 'ROOF'

        return BuildingSurface(
            name='NoName' + surf_type.lower().capitalize() if name is None else name,
            vertices=rect.to_points(),
            fenestration=fenestration if fenestration is not None else [],
            surface_type=surf_type,
            construction=construction.get_ref(),
            outside_boundary_condition=outside_bc
        )


class NonZoneSurfaceCreator:

    @staticmethod
    def from_rect(rect: Rectangle, construction: Construction, name: str = None,
                  surface_type: str = 'wall') -> NonZoneSurface:
        return NonZoneSurface(
            name='NoNameNonZoneSurface' if name is None else name,
            vertices=rect.to_points(),
            surface_type=surface_type,
            construction=construction.get_ref()
        )


class ZoneCreator:

    @staticmethod
    def from_box(box: Box, floor_construction: Construction, wall_construction: Construction,
                 ceiling_construction: Construction, name: str = None,
                 split_vertical: int = 1) -> Union[Zone, List[Zone]]:
        if split_vertical > 1:
            new_height = box.height_vector() * (1 / split_vertical)

            return [
                ZoneCreator.from_box(
                    box=Box(
                        base=move(box.base, new_height * i),
                        external_point=move(box.base.side.start, new_height * (i + 1))
                    ),
                    floor_construction=floor_construction,
                    wall_construction=wall_construction,
                    ceiling_construction=ceiling_construction,
                    name=name + '_{no}'.format(no=i) if name is not None else 'NoNameZone_{no}'.format(no=i),
                    split_vertical=1
                ) for i in range(split_vertical)
            ]

        else:
            surfaces = box.to_rects()  # [bottom, sides..., top]
            floor = BuildingSurfaceCreator.from_rect(
                rect=surfaces[0],
                construction=floor_construction,
                name="NoNameZoneFloor" if name is None else name + "Floor",
                surf_type='FLOOR'
            )
            walls = [BuildingSurfaceCreator.from_rect(
                rect=surfaces[i],
                construction=wall_construction,
                name="NoNameZoneWall" + str(i) if name is None else name + "Wall" + str(i),
                surf_type='WALL'
            ) for i in range(1, 5)]
            ceiling = BuildingSurfaceCreator.from_rect(
                rect=surfaces[5],
                construction=ceiling_construction,
                name="NoNameZoneCeiling" if name is None else name + "Floor",
                surf_type='CEILING'
            )

            return ZoneCreator.evaluate_surface_direction(
                zone=Zone(
                    name='NoNameZone' if name is None else name,
                    building_surfaces=[floor] + walls + [ceiling],
                    internal_masses=[],
                )
            )

    @staticmethod
    def evaluate_surface_direction(zone: Zone) -> Zone:
        """
        Evaluates if surface normals are pointing outside of the zone or not. Surfaces with normal vector pointing
        inside will be flipped. (Vertices will be reversed)
        Only convex zones are allowed.
        """
        def get_external_point(building_surface: BuildingSurface) -> Point:
            # get a point of another surface from the zone, that is none of the points from the actual surface
            for surf in zone.BuildingSurfaces:
                for point in surf.vertices:
                    if point not in building_surface.vertices:
                        return point

        for surface in zone.BuildingSurfaces:
            ext_vector = get_external_point(surface) - surface.vertices[0]
            if ext_vector * surface.normal_vector() > 0:
                # normal pointing inside of zone, reverse the vertices
                surface.vertices.reverse()
            elif ext_vector * surface.normal_vector() == 0:
                raise Exception("External point from other surface is in plane with the evaluated surface")

        return zone

    @staticmethod
    def evaluate_adjacency(zones: List[Zone]) -> List[Zone]:
        # TODO ceiling -> roof if not adjacent
        # TODO floor -> floor-to-ground if not adjacent
        for i, zone in enumerate(zones):
            for other_zone in zones[i+1:]:
                for surface in zone.BuildingSurfaces:
                    # equality check based on geometry only
                    if surface in other_zone.BuildingSurfaces:
                        # adjacent
                        other_index = other_zone.BuildingSurfaces.index(surface)
                        other_surface = other_zone.BuildingSurfaces[other_index]
                        surface.OutsideBoundaryCondition = other_surface.IuId
                        other_surface.OutsideBoundaryCondition = surface.IuId
        return zones


class BuildingCreator:

    def __init__(self, library: ObjectLibrary):
        self.library = library

    def make(self, zones: List[Zone], non_zone_surfaces: Union[List[NonZoneSurface], None] = None, name: str = None,
             building_function: str = 'residential'):
        return Building(
            name="NoNameBuilding" if name is None else name,
            zones=zones,
            non_zone_surfaces=non_zone_surfaces if non_zone_surfaces is not None else [],
            library=self.library,
            building_function=building_function
        )

    def from_box(self):
        # TODO
        pass
