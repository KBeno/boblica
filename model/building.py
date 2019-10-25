from typing import List, Dict
import math
import uuid
from .geometry import Vector, Point, Face

# TODO create base class for building objects to simplify code
# TODO create describe() method for base class OR describe(obj) function in tools module


class Ref:
    def __init__(self, ref_id: str, obj_type: str):
        self.RefId = ref_id
        self.ObjType = obj_type


# TODO we can leave this out and add this information at the IDF import level
class GlobalGeometryRules:
    def __init__(self, starting_vertex_position: str, vertex_entry_direction: str, coordinate_system: str,
                 north_axis: float):
        self.StartingVertexPosition = starting_vertex_position
        self.VertexEntryDirection = vertex_entry_direction
        self.CoordinateSystem = coordinate_system
        self.NorthAxis = north_axis


class OpaqueMaterial:
    def __init__(self, name: str, dbid: float, thickness: float, conductivity: float, density: float,
                 specific_heat: float, roughness: str = None, thermal_absorptance: float = None,
                 solar_absorptance: float = None, visible_absorptance: float = None):
        self.Name = name
        self.DbId = dbid
        self.IuId = str(uuid.uuid1())  # Internal Unique Identifier
        self.Roughness = roughness
        self.Thickness = thickness
        self.Conductivity = conductivity
        self.Density = density
        self.SpecificHeat = specific_heat
        self.ThermalAbsorptance = thermal_absorptance
        self.SolarAbsorptance = solar_absorptance
        self.VisibleAbsorptance = visible_absorptance

    def __str__(self):
        return self.Name + " (OpaqueMaterial)"

    # TODO move to base class or tools module
    def describe(self):
        description = 'Name:                {info}\n'.format(info=self.Name) + \
                      'Id:                  {info}\n'.format(info=self.DbId) + \
                      'Roughness:           {info}\n'.format(info=self.Roughness) + \
                      'Thickness:           {info}\n'.format(info=self.Thickness) + \
                      'Conductivity:        {info}\n'.format(info=self.Conductivity) + \
                      'Density:             {info}\n'.format(info=self.Density) + \
                      'Specific Heat:       {info}\n'.format(info=self.SpecificHeat) + \
                      'Thermal Absorptance: {info}\n'.format(info=self.ThermalAbsorptance) + \
                      'Solar Absorptance:   {info}\n'.format(info=self.SolarAbsorptance) + \
                      'Visible Absorptance: {info}\n'.format(info=self.VisibleAbsorptance)
        return description

    def get_ref(self):
        return Ref(self.IuId, self.__class__.__name__)


class WindowMaterial:
    def __init__(self, name: str, typ: str, glazing_id: float, frame_id: float, u_value: float, g_value: float):
        self.Name = name
        self.IuId = str(uuid.uuid1())
        self.Type = typ  # this is inherited from EnergyPlus and might not be useful
        self.GlazingId = glazing_id
        self.FrameId = frame_id
        self.UValue = u_value
        self.gValue = g_value

    def __str__(self):
        return self.Name + " (WindowMaterial)"

    def get_ref(self):
        return Ref(self.IuId, self.__class__.__name__)


class ShadingMaterial:
    def __init__(self, name: str, db_id: float, reflectance: float, transmittance: float, emissivity: float,
                 thickness: float, conductivity: float, density: float):
        self.Name = name
        self.DbId = db_id
        self.IuId = str(uuid.uuid1())
        self.Reflectance = reflectance
        self.Transmittance = transmittance
        self.Emissivity = emissivity
        self.Thickness = thickness
        self.Conductivity = conductivity
        self.Density = density  # not in honeybee

    def __str__(self):
        return self.Name + " (ShadingMaterial)"

    def get_ref(self):
        return Ref(self.IuId, self.__class__.__name__)


class Shading:
    def __init__(self, name: str, typ: str, properties: dict, material: Ref, shading_factor: float,
                 is_scheduled: bool, distance_to_glass: float = None, slat_width: float = None,
                 slat_separation: float = None, slat_angle: float = None):
        self.Name = name
        self.IuId = str(uuid.uuid1())
        self.Type = typ
        # Types are: ExteriorShade, ExteriorBlind, InteriorShade, InteriorBlind
        self.Properties = properties  # a strange dictionary used in honeybee
        self.Material = material
        self.DistanceToGlass = distance_to_glass
        self.SlatWidth = slat_width  # only in case of blinds
        self.SlatSeparation = slat_separation  # only in case of blinds
        self.SlatAngle = slat_angle  # only in case of blinds
        self.ShadingFactor = shading_factor  # 0 < float < 1
        self.IsScheduled = is_scheduled
        # self.LCA = LCA()

    def __str__(self):
        return '{name} (Shading)'.format(name=self.Name)

    def area_per_window_m2(self):
        if self.Type is not None:
            if 'BLIND' in self.Type.upper():  # venetian blinds - lamellas arnyekolo
                shading_material_area = 1 / self.SlatSeparation * self.SlatWidth
            elif 'SHADE' in self.Type.upper():  # shades - redony
                shading_material_area = 1
            else:
                raise Exception('Shade type has to be either *Blind or *Shade instead of {}'.format(self.Type))
            return shading_material_area
        else:
            return 0

    def get_ref(self):
        return Ref(self.IuId, self.__class__.__name__)


class Construction:
    def __init__(self, name: str, layers: List[Ref]):
        self.Name = name
        self.IuId = str(uuid.uuid1())
        self.Layers = layers  # first item is the outer-most layer

    def __str__(self):
        return self.Name + " (Construction)"

    def thickness(self, materials: dict):
        t = 0
        for layer in self.Layers:
            t += materials[layer.RefId].Thickness
        return t

    def get_ref(self):
        return Ref(self.IuId, self.__class__.__name__)


class Surface(Face):
    def __init__(self, name: str, vertices: List[Point], geometry_rules: GlobalGeometryRules = None):
        super().__init__(vertices)
        self.Name = name
        self.IuId = str(uuid.uuid1())
        self.GeometryRules = geometry_rules

    def normal(self):
        # normal vector pointing outside of the zone
        # requires GlobalGeometryRules from read_idf to be defined before running this method

        if self.GeometryRules.VertexEntryDirection == "CounterClockWise":
            return self.normal_vector()
        elif self.GeometryRules.VertexEntryDirection == "ClockWise":
            return self.normal_vector() * -1

    def orientation(self):

        north_axis = self.GeometryRules.NorthAxis

        north_vector = Vector(-math.sin(math.radians(north_axis)), math.cos(math.radians(north_axis)), 0.0)
        east_vector = Vector(north_vector.y, -north_vector.x, 0.0)

        normal = self.normal()
        normal_projected = Vector(normal.x, normal.y, 0)

        if normal_projected.length() == 0:
            raise Exception("horizontal surface doesn't have orientation")

        if north_vector.angle(normal_projected) <= 22.5:
            return "North"
        elif north_vector.angle(normal_projected) < 67.5:
            if east_vector.angle(normal_projected) < 90:
                return "NorthEast"
            else:
                return "NorthWest"
        elif north_vector.angle(normal_projected) <= 112.5:
            if east_vector.angle(normal_projected) < 90:
                return "East"
            else:
                return "West"
        elif north_vector.angle(normal_projected) < 157.5:
            if east_vector.angle(normal_projected) < 90:
                return "SouthEast"
            else:
                return "SouthWest"
        else:
            return "South"


class FenestrationSurface(Surface):
    def __init__(self, name: str, vertices: List[Point], surface_type: str, shading: Ref,
                 construction: Ref, shading_control_name: str = None, multiplier: int = None,
                 geometry_rules: GlobalGeometryRules = None):
        super().__init__(name, vertices, geometry_rules)
        self.Shading = shading
        self.SurfaceType = surface_type
        self.Construction = construction
        self.ShadingControlName = shading_control_name  # EnergyPlus stuff, might not needed
        self.Multiplier = multiplier  # EnergyPlus stuff, might not needed

    def __str__(self):
        return self.Name + " (FenestrationSurface)"

    def shading_factor(self, shadings: dict):
        if self.Shading is not None:
            return shadings[self.Shading.RefId].ShadingFactor, shadings[self.Shading.RefId].IsScheduled
        else:
            return 1, False

    def glazing_area(self, mode="Ratio", frame_width=0.1, glazing_ratio=0.8):
        if mode == "FrameWidth":
            # works properly only for rectangular surfaces

            side_vector1 = self.vertices[1] - self.vertices[0]
            side_vector2 = self.vertices[-1] - self.vertices[0]

            return (side_vector1.length() - 2 * frame_width) * (side_vector2.length() - 2 * frame_width)
        elif mode == "Ratio":
            return self.area() * glazing_ratio
        else:
            raise Exception("Mode options for glazing_area method: FrameWidth or Ratio.")

    def frame_area(self, mode="Ratio", frame_width=0.1, glazing_ratio=0.8):
        if mode == "FrameWidth":
            return self.area() - self.glazing_area("FrameWidth", frame_width)
        elif mode == "Ratio":
            return self.area() * (1 - glazing_ratio)
        else:
            raise Exception("Mode options for frame_area method: FrameWidth or Ratio.")


class BuildingSurface(Surface):
    def __init__(self, name: str, vertices: List[Point], fenestration: List[FenestrationSurface], surface_type: str,
                 construction: Ref, outside_boundary_condition: str,
                 geometry_rules: GlobalGeometryRules = None):
        super().__init__(name, vertices, geometry_rules)
        self.Fenestration = fenestration
        self.SurfaceType = surface_type
        self.Construction = construction
        self.OutsideBoundaryCondition = outside_boundary_condition

    def __str__(self):
        return self.Name + " (BuildingSurface)"

    def area_net(self):
        # area of the Surface (Fenestration areas subtracted)
        area = self.area()
        for window in self.Fenestration:
            area -= window.area()
        return area


class NonZoneSurface(Surface):
    def __init__(self, name: str, vertices: List[Point], surface_type: str, construction: Ref,
                 geometry_rules: GlobalGeometryRules = None):
        super().__init__(name, vertices, geometry_rules)
        self.SurfaceType = surface_type
        self.Construction = construction


class InternalMass:
    def __init__(self, name: str, construction: Ref, area: float):
        self.Name = name
        self.IuId = str(uuid.uuid1())
        self.Construction = construction
        self.Area = area


class Zone:
    def __init__(self, name: str, building_surfaces: List[BuildingSurface], internal_masses: List[InternalMass],
                 direction_of_relative_north: float = None, origin=None):
        self.Name = name
        self.IuId = str(uuid.uuid1())
        self.BuildingSurfaces = building_surfaces
        self.InternalMasses = internal_masses
        self.DirectionOfRelativeNorth = direction_of_relative_north  # EneryPlus stuff, might not needed
        self.Origin = origin  # EneryPlus stuff, might not needed

    def __str__(self):
        return str(self.Name) if self.Name is not None else "NoName" + " (Zone)"

    def volume(self):

        volume = 0
        for surface in self.BuildingSurfaces:
            area = surface.area(True)
            height = surface.vertices[0] * surface.normal()
            volume += area * height / 3
        return volume

    def heated_area(self):

        heated_area = 0
        for surface in self.BuildingSurfaces:
            if surface.SurfaceType == "FLOOR" or surface.SurfaceType == "ExposedFloor" \
                    or surface.SurfaceType == "SlabOnGrade":
                heated_area += surface.area()
        return heated_area


class Building:
    def __init__(self, name: str, zones: List[Zone], non_zone_surfaces: List[NonZoneSurface],
                 constructions: Dict[str, Construction], shadings: Dict[str, Shading],
                 opaque_materials: Dict[str, OpaqueMaterial], shading_materials: Dict[str, ShadingMaterial],
                 window_materials: Dict[str, WindowMaterial],
                 global_geometry_rules: GlobalGeometryRules = None,
                 building_function: str = None):
        self.Name = name
        self.IuId = str(uuid.uuid1())
        self.Zones = zones
        self.NonZoneSurfaces = non_zone_surfaces
        self.Constructions = constructions
        self.Shadings = shadings
        self.OpaqueMaterials = opaque_materials
        self.ShadingMaterials = shading_materials
        self.WindowMaterials = window_materials
        # TODO include hvac
        # self.HVAC = hvac
        self.GlobalGeometryRules = global_geometry_rules
        self.BuildingFunction = building_function

    def heated_area(self):
        heated_area = 0
        for zone in self.Zones:
            heated_area += zone.heated_area()
        return heated_area

    def volume(self):
        volume = 0
        for zone in self.Zones:
            volume += zone.volume()
        return volume

    def evaluate_geometry(self):
        # total length of connected edges of exposed surfaces
        connected_edge_length = 0
        # total window perimeter
        fenestration_perimeter = 0
        # total external wall area
        ext_wall_area = 0
        # total flat roof area
        flatroof_area = 0
        # total floor to ground area
        floor_to_ground_area = 0
        # total exposed floor area
        exposed_floor_area = 0
        # total window area
        fenestration_area = 0
        for zone in self.Zones:
            for surface in zone.BuildingSurfaces:
                if surface.OutsideBoundaryCondition.lower() in ['outdoors', 'ground']:
                    connected_edge_length += surface.perimeter()
                    if surface.SurfaceType.lower() == 'wall':
                        ext_wall_area += surface.area()
                        for window in surface.Fenestration:
                            fenestration_perimeter += window.perimeter()
                            fenestration_area += window.area()
                    elif surface.SurfaceType.lower() in ['ceiling', 'roof']:
                        flatroof_area += surface.area()
                    elif surface.SurfaceType.lower() == 'slabongrade':
                        floor_to_ground_area += surface.area()
                    elif surface.SurfaceType.lower() == 'exposedfloor':
                        exposed_floor_area += surface.area()
        return {
            'ConnectedEdgeLength': connected_edge_length / 2,
            'FenestrationPerimeter': fenestration_perimeter,
            'FenestrationArea': fenestration_area,
            'ExternalWallArea': ext_wall_area,
            'FlatRoofArea': flatroof_area,
            'FloorToGroundArea': floor_to_ground_area,
            'ExposedFloorArea': exposed_floor_area
        }
