from typing import List, Union, Mapping, MutableMapping
import math
import uuid
from .geometry import Vector, Point, Face
from .hvac import HVAC

# TODO create base class for building objects to simplify code
# TODO create describe() method for base class OR describe(obj) function in tools module

"""
Principles:
    - North direction equals to positive y axis (expressed as 0 degrees in GlobalGeometryRules
"""


class Ref:
    def __init__(self, ref_id: str, name: str, obj_type: str):
        self.RefId = ref_id
        self.RefName = name
        self.ObjType = obj_type

    def __str__(self):
        return '{n} ({t}Ref)'.format(n=self.RefName, t=self.ObjType)

# TODO we can leave this out and add this information at the IDF import level
class GlobalGeometryRules:
    def __init__(self, starting_vertex_position: str = 'LowerLeftCorner',
                 vertex_entry_direction: str = 'CounterClockWise',
                 coordinate_system: str = 'Relative', north_axis: float = 0):
        self.StartingVertexPosition = starting_vertex_position
        self.VertexEntryDirection = vertex_entry_direction
        self.CoordinateSystem = coordinate_system
        self.NorthAxis = north_axis


class OpaqueMaterial:
    def __init__(self, name: str, thickness: float, conductivity: float, density: float,
                 specific_heat: float, roughness: str, thermal_absorptance: float, solar_absorptance: float,
                 visible_absorptance: float, db_id: float = None,
                 transport_scenario: float = None, disposal_scenario: float = None,
                 cutting_waste: float = None, life_time: int = None):
        self.Name = name
        self.DbId = db_id
        self.IuId = str(uuid.uuid1())  # Internal Unique Identifier
        self.Roughness = roughness
        self.Thickness = thickness  # [m]
        self.Conductivity = conductivity
        self.Density = density  # [kg/m3]
        self.SpecificHeat = specific_heat  # [J/kgK]
        self.ThermalAbsorptance = thermal_absorptance
        self.SolarAbsorptance = solar_absorptance
        self.VisibleAbsorptance = visible_absorptance
        # Life Cycle properties # TODO leave this out of model
        self.TransportScenario = transport_scenario
        self.DisposalScenario = disposal_scenario
        self.CuttingWaste = cutting_waste
        self.LifeTime = life_time

    def __str__(self):
        return self.Name + " (OpaqueMaterial)"

    # TODO move to base class or tools module
    def describe(self):
        description = 'Name:                {info}\n'.format(info=self.Name) + \
                      'Database Id:         {info}\n'.format(info=self.DbId) + \
                      'Internal Id:         {info}\n'.format(info=self.IuId) + \
                      'Roughness:           {info}\n'.format(info=self.Roughness) + \
                      'Thickness:           {info}\n'.format(info=self.Thickness) + \
                      'Conductivity:        {info}\n'.format(info=self.Conductivity) + \
                      'Density:             {info}\n'.format(info=self.Density) + \
                      'Specific Heat:       {info}\n'.format(info=self.SpecificHeat) + \
                      'Thermal Absorptance: {info}\n'.format(info=self.ThermalAbsorptance) + \
                      'Solar Absorptance:   {info}\n'.format(info=self.SolarAbsorptance) + \
                      'Visible Absorptance: {info}\n'.format(info=self.VisibleAbsorptance) + \
                      'Transport Scenario:  {info}\n'.format(info=self.TransportScenario) + \
                      'Disposal Scenario:   {info}\n'.format(info=self.DisposalScenario) + \
                      'Cutting Waste:       {info}\n'.format(info=self.CuttingWaste) + \
                      'Life Time:           {info}\n'.format(info=self.LifeTime)
        return description

    def get_ref(self):
        return Ref(self.IuId, self.Name, self.__class__.__name__)


class WindowMaterial:
    def __init__(self, name: str, typ: str, u_value: float, g_value: float,
                 surface_weight: float = None, db_id: float = None,
                 glazing_id: float = None, frame_id: float = None,
                 transport_scenario_glazing: float = None,
                 disposal_scenario_glazing: float = None,
                 cutting_waste_glazing: float = None,
                 life_time_glazing: int = None,
                 transport_scenario_frame: float = None,
                 disposal_scenario_frame: float = None,
                 cutting_waste_frame: float = None,
                 life_time_frame: int = None,
                 transport_scenario_window: float = None,
                 disposal_scenario_window: float = None,
                 cutting_waste_window: float = None,
                 life_time_window: float = None):
        self.Name = name
        self.IuId = str(uuid.uuid1())
        self.Type = typ  # this is inherited from EnergyPlus and might not be useful
        self.DbId = db_id  # if db data include frame and glazing
        self.GlazingId = glazing_id  # if there is separate data for glazing
        self.FrameId = frame_id
        self.UValue = u_value
        self.gValue = g_value
        self.SurfaceWeight = surface_weight  # kg/m2

        # Life Cycle properties # TODO leave this out of model
        self.TransportScenarioWindow = transport_scenario_window
        self.DisposalScenarioWindow = disposal_scenario_window
        self.CuttingWastewindow = cutting_waste_window
        self.LifeTimeWindow = life_time_window

        # if separate frame and glazing
        self.TransportScenarioGlazing = transport_scenario_glazing
        self.TransportScenarioFrame = transport_scenario_frame
        self.DisposalScenarioGlazing = disposal_scenario_glazing
        self.DisposalScenarioFrame = disposal_scenario_frame
        self.CuttingWasteGlazing = cutting_waste_glazing
        self.CuttingWasteFrame = cutting_waste_frame
        self.LifeTimeGlazing = life_time_glazing
        self.LifeTimeFrame = life_time_frame

    def __str__(self):
        return self.Name + " (WindowMaterial)"

    def get_ref(self):
        return Ref(self.IuId, self.Name, self.__class__.__name__)


# TODO
# class GlazingMaterial:
# class FrameMaterial:
class ShadeMaterial:
    def __init__(self, name: str, reflectance: float, transmittance: float, emissivity: float,
                 thickness: float, conductivity: float, distance_to_glass: float, density: float = None,
                 db_id: float = None):
        self.Name = name
        self.DbId = db_id
        self.IuId = str(uuid.uuid1())
        self.Reflectance = reflectance
        self.Transmittance = transmittance
        self.Emissivity = emissivity
        self.Thickness = thickness
        self.Conductivity = conductivity
        self.Density = density  # not in honeybee
        self.DistanceToGlass = distance_to_glass

    def area_per_window_m2(self):
        return 1

    def __str__(self):
        return self.Name + " (ShadeMaterial)"

    def get_ref(self) -> Ref:
        return Ref(self.IuId, self.Name, self.__class__.__name__)


class BlindMaterial:
    """
    Venetian blinds - lamellas arnyekolo
    """
    def __init__(self, name: str, reflectance: float, transmittance: float, emissivity: float,
                 thickness: float, conductivity: float, distance_to_glass: float,
                 slat_width: float, slat_separation: float, slat_angle: float, density: float = None,
                 db_id: float = None):
        self.Name = name
        self.DbId = db_id
        self.IuId = str(uuid.uuid1())
        self.Reflectance = reflectance
        self.Transmittance = transmittance
        self.Emissivity = emissivity
        self.Thickness = thickness
        self.Conductivity = conductivity
        self.Density = density  # not in honeybee
        self.DistanceToGlass = distance_to_glass
        self.SlatWidth = slat_width
        self.SlatSeparation = slat_separation
        self.SlatAngle = slat_angle  # in degrees relative to vertical

    def area_per_window_m2(self):
        return 1 / self.SlatSeparation * self.SlatWidth

    def __str__(self):
        return self.Name + " (BlindMaterial)"

    def get_ref(self) -> Ref:
        return Ref(self.IuId, self.Name, self.__class__.__name__)

# class ShadingMaterial: # Deprecated
#     def __init__(self, name: str, reflectance: float, transmittance: float, emissivity: float,
#                  thickness: float, conductivity: float, density: float = None,
#                  db_id: float = None,
#                  transport_scenario: float = None, disposal_scenario: float = None,
#                  cutting_waste: float = None, life_time: int = None):
#         self.Name = name
#         self.DbId = db_id
#         self.IuId = str(uuid.uuid1())
#         self.Reflectance = reflectance
#         self.Transmittance = transmittance
#         self.Emissivity = emissivity
#         self.Thickness = thickness
#         self.Conductivity = conductivity
#         self.Density = density  # not in honeybee
#         # Life Cycle properties
#         self.TransportScenario = transport_scenario
#         self.DisposalScenario = disposal_scenario
#         self.CuttingWaste = cutting_waste
#         self.LifeTime = life_time
#
#     def __str__(self):
#         return self.Name + " (ShadingMaterial)"
#
#     def get_ref(self) -> Ref:
#         return Ref(self.IuId, self.Name, self.__class__.__name__)


class Shading:
    def __init__(self, name: str, typ: str, properties: dict, is_scheduled: bool,
                 material: Ref = None, shading_factor: float = 1, construction: Ref = None):
                 # distance_to_glass: float = None,
                 # slat_width: float = None, slat_separation: float = None, slat_angle: float = None):
        """
        Either material or construction is given, if both construction will be omitted
        :param name:
        :param typ:
        :param properties:
        :param is_scheduled:
        :param material:
        :param shading_factor:
        :param construction:
        """
        self.Name = name
        self.IuId = str(uuid.uuid1())
        self.Type = typ
        # Types are: ExteriorShade, ExteriorBlind, InteriorShade, InteriorBlind
        self.Properties = properties  # a strange dictionary used in honeybee # TODO remove
        self.Material = material  # ShadeMaterial or BlindMaterial
        self.Construction = construction  # glazing construction with shading
        # self.DistanceToGlass = distance_to_glass
        # self.SlatWidth = slat_width  # only in case of blinds
        # self.SlatSeparation = slat_separation  # only in case of blinds
        # self.SlatAngle = slat_angle  # only in case of blinds -- in degrees relative to vertical
        self.ShadingFactor = shading_factor  # 0 < float < 1 # TODO move this to simple energy calculation (TNM)
        self.IsScheduled = is_scheduled
        # self.LCA = LCA()

    def __str__(self):
        return '{name} (Shading)'.format(name=self.Name)

    # def area_per_window_m2(self):
    #     if self.Type is not None:
    #         if 'BLIND' in self.Type.upper():  # venetian blinds - lamellas arnyekolo
    #             shading_material_area = 1 / self.SlatSeparation * self.SlatWidth
    #         elif 'SHADE' in self.Type.upper():  # shades - redony
    #             shading_material_area = 1
    #         else:
    #             raise Exception('Shade type has to be either *Blind or *Shade instead of {}'.format(self.Type))
    #         return shading_material_area
    #     else:
    #         return 0

    def get_ref(self):
        return Ref(self.IuId, self.Name, self.__class__.__name__)


class Construction:
    def __init__(self, name: str, layers: List[Ref]):
        self.Name = name
        self.IuId = str(uuid.uuid1())
        self.Layers = layers  # first item is the outer-most layer

    def __str__(self):
        return self.Name + " (Construction)"

    def thickness(self, library: "ObjectLibrary"):
        t = 0
        for layer in self.Layers:
            material = library.get(layer)
            t += material.Thickness
        return t

    def get_ref(self):
        return Ref(self.IuId, self.Name, self.__class__.__name__)


class Surface(Face):
    def __init__(self, name: str, vertices: List[Point], geometry_rules: GlobalGeometryRules = GlobalGeometryRules()):
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

    def get_ref(self):
        return Ref(self.IuId, self.Name, self.__class__.__name__)


class FenestrationSurface(Surface):
    def __init__(self, name: str, vertices: List[Point], surface_type: str, shading: Union[Ref, None],
                 construction: Ref, frame_name: str = None, shading_control_name: str = None, multiplier: int = None,
                 geometry_rules: GlobalGeometryRules = GlobalGeometryRules()):
        super().__init__(name, vertices, geometry_rules)
        self.Shading = shading
        self.SurfaceType = surface_type
        self.Construction = construction
        self.FrameName = frame_name
        self.ShadingControlName = shading_control_name  # EnergyPlus stuff, might not needed
        self.Multiplier = multiplier  # EnergyPlus stuff, might not needed

    def __str__(self):
        return self.Name + " (FenestrationSurface)"

    def shading_factor(self, library: "ObjectLibrary"):
        if self.Shading is not None:
            shading = library.get(self.Shading)
            return shading.ShadingFactor, shading.IsScheduled
        else:
            return 1, False

    def glazing_area(self, mode="Ratio", frame_width=0.1, glazing_ratio=0.7):
        # TODO use frame material
        if mode == "FrameWidth":
            # works properly only for rectangular surfaces

            side_vector1 = self.vertices[1] - self.vertices[0]
            side_vector2 = self.vertices[-1] - self.vertices[0]

            return (side_vector1.length() - 2 * frame_width) * (side_vector2.length() - 2 * frame_width)
        elif mode == "Ratio":
            return self.area() * glazing_ratio
        else:
            raise Exception("Mode options for glazing_area method: FrameWidth or Ratio.")

    def frame_area(self, mode="Ratio", frame_width=0.1, glazing_ratio=0.7):
        # TODO use frame material
        if mode == "FrameWidth":
            return self.area() - self.glazing_area("FrameWidth", frame_width)
        elif mode == "Ratio":
            return self.area() * (1 - glazing_ratio)
        else:
            raise Exception("Mode options for frame_area method: FrameWidth or Ratio.")


class BuildingSurface(Surface):
    def __init__(self, name: str, vertices: List[Point], fenestration: List[FenestrationSurface], surface_type: str,
                 construction: Ref, outside_boundary_condition: str,
                 geometry_rules: GlobalGeometryRules = GlobalGeometryRules()):
        super().__init__(name, vertices, geometry_rules)
        self.Fenestration = fenestration
        self.SurfaceType = surface_type  # TODO Enumeration
        self.Construction = construction
        self.OutsideBoundaryCondition = outside_boundary_condition  # TODO Enumeration
        # (Ground/Adiabatic/Outdoors/Surface)

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
                 geometry_rules: GlobalGeometryRules = GlobalGeometryRules()):
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
                 direction_of_relative_north: float = None, origin: Point = None):
        self.Name = name
        self.IuId = str(uuid.uuid1())
        self.BuildingSurfaces = building_surfaces
        self.InternalMasses = internal_masses
        self.DirectionOfRelativeNorth = direction_of_relative_north  # EneryPlus stuff, might not needed
        self.Origin = origin  # EneryPlus stuff, might not needed

    def __str__(self):
        return str(self.Name) if self.Name is not None else "NoName" + " (Zone)"

    def volume(self):
        # Sum of signed volume of each pyramid formed by the sides with a common
        # external point (in this case the origin)
        volume = 0
        for surface in self.BuildingSurfaces:
            area = surface.area(signed=True)
            x, y, z = surface.vertices[0].coordinates()
            height = Vector(x, y, z) * surface.normal()
            volume += area * height / 3
        return volume

    def heated_area(self):

        heated_area = 0
        for surface in self.BuildingSurfaces:
            if surface.SurfaceType.lower() in ["floor", "exposedfloor", "slabongrade"]:
                heated_area += surface.area()
        return heated_area


class ObjectLibrary:
    """
    Object to hold a collection of building elements that are used in several places in the building
    e.g. materials, constructions
    objects can be retrieved by passing a Ref object
    """
    def __init__(self, constructions: MutableMapping[str, Construction] = None,
                 shadings: MutableMapping[str, Shading] = None,
                 opaque_materials: MutableMapping[str, OpaqueMaterial] = None,
                 shade_materials: MutableMapping[str, ShadeMaterial] = None,
                 blind_materials: MutableMapping[str, BlindMaterial] = None,
                 window_materials: MutableMapping[str, WindowMaterial] = None,
                 default_key: str = 'IuId'):
        """

        :param constructions:
        :param shadings:
        :param opaque_materials:
        :param shade_materials:
        :param blind_materials:
        :param window_materials:
        :param default_key: what property to use from a reference to search in the library
        """
        self.constructions = {} if constructions is None else constructions
        self.shadings = {} if shadings is None else shadings
        self.opaque_materials = {} if opaque_materials is None else opaque_materials
        self.shade_materials = {} if shade_materials is None else shade_materials
        self.blind_materials = {} if blind_materials is None else blind_materials
        self.window_materials = {} if window_materials is None else window_materials
        self.default_key = default_key

    @property
    def default_key(self) -> str:
        return self._obj_key

    @default_key.setter
    def default_key(self, key):
        if key in ['IuId', 'RefId']:
            # IuId is the property of the objects, RefId is the property holding the reference to the objects IuId
            self._ref_key = 'RefId'
            self._obj_key = 'IuId'
        elif key in ['Name', 'RefName']:
            # Name is the property of the objects, RefName is the property holding the reference to the objects Name
            self._ref_key = 'RefName'
            self._obj_key = 'Name'
        else:
            raise Exception('Key should be either Name ("Name", "RefName") or Id("IuId", "RefId")')

    def change_key(self, to: str):
        self.default_key = to

        constructions = {}
        for const in self.constructions.values():
            key = getattr(const, self._obj_key)
            constructions[key] = const

        shadings = {}
        for shade in self.shadings.values():
            key = getattr(shade, self._obj_key)
            shadings[key] = shade

        opaque_materials = {}
        for mat in self.opaque_materials.values():
            key = getattr(mat, self._obj_key)
            opaque_materials[key] = mat

        window_materials = {}
        for mat in self.window_materials.values():
            key = getattr(mat, self._obj_key)
            window_materials[key] = mat

        shade_materials = {}
        for mat in self.shade_materials.values():
            key = getattr(mat, self._obj_key)
            shade_materials[key] = mat

        blind_materials = {}
        for mat in self.blind_materials.values():
            key = getattr(mat, self._obj_key)
            blind_materials[key] = mat

        self.constructions = constructions
        self.shadings = shadings
        self.opaque_materials = opaque_materials
        self.window_materials = window_materials
        self.shade_materials = shade_materials
        self.blind_materials = blind_materials

    def get(self, ref: Ref):

        key = getattr(ref, self._ref_key)

        if ref.ObjType == 'OpaqueMaterial':
            return self.opaque_materials[key]

        elif ref.ObjType == 'WindowMaterial':
            return self.window_materials[key]

        elif ref.ObjType == 'ShadeMaterial':
            return self.shade_materials[key]

        elif ref.ObjType == 'BlindMaterial':
            return self.blind_materials[key]

        elif ref.ObjType == 'Shading':
            return self.shadings[key]

        elif ref.ObjType == 'Construction':
            return self.constructions[key]

    def find(self, obj_type: str, ref_name: str = None):
        if self._obj_key != 'Name':
            raise Exception("RefId is used as default key, please change default key to 'Name' first"
                            "with Library.change_key(to='Name')")

        if obj_type == 'OpaqueMaterial':
            if ref_name is None:
                return [obj for obj in self.opaque_materials.values()]
            return self.opaque_materials[ref_name]

        elif obj_type == 'WindowMaterial':
            if ref_name is None:
                return [obj for obj in self.window_materials.values()]
            return self.window_materials[ref_name]

        elif obj_type == 'ShadeMaterial':
            if ref_name is None:
                return [obj for obj in self.shade_materials.values()]
            return self.shade_materials[ref_name]

        elif obj_type == 'BlindMaterial':
            if ref_name is None:
                return [obj for obj in self.blind_materials.values()]
            return self.blind_materials[ref_name]

        elif obj_type == 'Shading':
            if ref_name is None:
                return [obj for obj in self.shadings.values()]
            return self.shadings[ref_name]

        elif obj_type == 'Construction':
            if ref_name is None:
                return [obj for obj in self.constructions.values()]
            return self.constructions[ref_name]

    def __contains__(self, item):

        key = getattr(item, self._obj_key)

        if isinstance(item, OpaqueMaterial):
            if key in self.opaque_materials:
                return True
            else:
                return False

        elif isinstance(item, WindowMaterial):
            if key in self.window_materials:
                return True
            else:
                return False

        elif isinstance(item, ShadeMaterial):
            if key in self.shade_materials:
                return True
            else:
                return False

        elif isinstance(item, BlindMaterial):
            if key in self.blind_materials:
                return True
            else:
                return False

        elif isinstance(item, Shading):
            if key in self.shadings:
                return True
            else:
                return False

        elif isinstance(item, Construction):
            if key in self.constructions:
                return True
            else:
                return False

    def add(self, obj: Union[OpaqueMaterial,
                             WindowMaterial,
                             ShadeMaterial,
                             BlindMaterial,
                             Shading,
                             Construction]) -> bool:

        if obj in self:
            return False

        # else:
        key = getattr(obj, self._obj_key)

        if isinstance(obj, OpaqueMaterial):
            self.opaque_materials[key] = obj
            return True

        elif isinstance(obj, WindowMaterial):
            self.window_materials[key] = obj
            return True

        elif isinstance(obj, ShadeMaterial):
            self.shade_materials[key] = obj
            return True

        elif isinstance(obj, BlindMaterial):
            self.blind_materials[key] = obj
            return True

        elif isinstance(obj, Shading):
            self.shadings[key] = obj
            return True

        elif isinstance(obj, Construction):
            self.constructions[key] = obj
            return True


class Building:
    def __init__(self, name: str, zones: List[Zone], non_zone_surfaces: List[NonZoneSurface],
                 library: ObjectLibrary,
                 # constructions: Mapping[str, Construction], shadings: Mapping[str, Shading],
                 # opaque_materials: Mapping[str, OpaqueMaterial], shading_materials: Mapping[str, ShadingMaterial],
                 # window_materials: Mapping[str, WindowMaterial],
                 building_function: str,
                 global_geometry_rules: GlobalGeometryRules = GlobalGeometryRules(),
                 hvac: HVAC = None):
        self.Name = name
        self.IuId = str(uuid.uuid1())
        self.Zones = zones
        self.NonZoneSurfaces = non_zone_surfaces
        self.Library = library
        # self.Constructions = constructions
        # self.Shadings = shadings
        # self.OpaqueMaterials = opaque_materials
        # self.ShadingMaterials = shading_materials
        # self.WindowMaterials = window_materials
        # TODO include hvac
        self.HVAC = hvac
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
