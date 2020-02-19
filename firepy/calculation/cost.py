from typing import Mapping, Union, MutableMapping, List
from pathlib import Path
import logging

import pandas as pd

import firepy.model.building
import firepy.model.hvac
from firepy.model.building import ObjectLibrary
# from firepy.tools.database import SqlDB  # ,OLCA

logger = logging.getLogger(__name__)

class CostResult:

    def __init__(self, ref_unit: str):
        self.ReferenceUnit = ref_unit
        self._costs = pd.Series(index=['Production', 'Installation', 'Replacement', 'Operation'])

    @property
    def costs(self) -> pd.Series:
        """
        Get cost results as Pandas Series

        :return: pandas Series
            - index: Production, Installation, Replacement, Operation
        """
        return self._costs

    @costs.setter
    def costs(self, new: pd.Series):
        self._costs = new


class CostCalculation:

    def __init__(self, reference_service_period: int = 50,
                 life_cycle_data: Union[str, pd.DataFrame] = None,
                 cost_data: Union[str, pd.DataFrame] = None,
                 db = None,
                 matching_col: str = 'DbId', matching_property: str = 'DbId',
                 considered_objects: List[str] = None):
        # these are basically cache objects
        self._cost_results = {}  # Dict of calculated cost of objects {IuId: ImpactRusult} Mapping[str, Impact]
        # self._inventories = {}  # for all objects calculate the amount of referenced objects in total
        # (opaque_mat, window_mat, shading_mat, [construction and shading])

        # the reference lifetime in years
        self.rsp = reference_service_period

        # how the model objects will be matched with the LifeCycle Data (column name in the life_cycle_data
        self.match_col = matching_col
        # what property of the model objects (e.g. materials) to use for the matching ('DbId' or 'Name')
        self.match_prop = matching_property

        self.considered = considered_objects

        # Life Cycle Data can be given:
        #   - directly by a DataFrame or a csv address
        #   - from a database if DbId-s are supplied within the model (database_connection)
        self.life_cycle_data = life_cycle_data  # DataFrame

        # Cost Data can be given:
        #   - directly by a DataFrame or a csv address
        #   - from a database if DbId-s are supplied within the model (database_connection)
        self.cost_data = cost_data  # DataFrame

        self._db = db  # TODO SqlDB instance to connect to the database if no ready results are supplied

        # we have to feed the class with either:
        #   - LifeCycleData AND CostData
        #   - DB

    @property
    def cost_results(self) -> MutableMapping[str, CostResult]:
        return self._cost_results

    # @property
    # def Inventories(self) -> Mapping[str, Inventory]:
    #     return self._inventories

    @property
    def life_cycle_data(self) -> pd.DataFrame:
        """
        pandas DataFrame with all the life cycle information
        Columns:
          - Name, DbId, Unit, openLCAname, openLCAid, TransportId, WasteTreatmentId,
            WasteTreatmentTransportId, LifeTime, CuttingWaste
        Rows:
          - Materials
        Index: DbId

        :return: DataFrame
        """

        return self._life_cycle_data

    @life_cycle_data.setter
    def life_cycle_data(self, source: Union[str, Path, pd.DataFrame]):
        if isinstance(source, str):
            # Read from path
            self._life_cycle_data = pd.read_csv(source, index_col=self.match_col)
        elif isinstance(source, Path):
            self._life_cycle_data = pd.read_csv(str(source), index_col=self.match_col)
        elif isinstance(source, pd.DataFrame):
            source.set_index(self.match_col)
            self._life_cycle_data = source

    @property
    def cost_data(self) -> pd.DataFrame:
        """
        pandas DataFrame with all the costs
        Columns: (MultiIndex)
          - Metadata[Name, DbId], Costs[Production, Installation], Units[Production, Installation]
        Rows:
          - Material / Energy / Waste Treatment
        Index: DbId
        """
        return self._cost_data

    @cost_data.setter
    def cost_data(self, source: Union[str, Path, pd.DataFrame]):
        if isinstance(source, str):
            # Read from path
            self._cost_data = pd.read_csv(source, index_col=0, header=[0, 1])
        elif isinstance(source, Path):
            self._cost_data = pd.read_csv(str(source), index_col=0, header=[0, 1])
        elif isinstance(source, pd.DataFrame):
            self._cost_data = source

    def clear_cache(self):
        self._cost_results = {}

    def calculate_cost(self, obj: Union[firepy.model.building.OpaqueMaterial,
                                        firepy.model.building.WindowMaterial,
                                        firepy.model.building.ShadeMaterial,
                                        firepy.model.building.BlindMaterial,
                                        firepy.model.building.Shading,
                                        firepy.model.building.Construction,
                                        firepy.model.building.BuildingSurface,
                                        firepy.model.building.FenestrationSurface,
                                        firepy.model.hvac.Heating,
                                        firepy.model.hvac.Cooling,
                                        firepy.model.hvac.Lighting,
                                        firepy.model.hvac.HVAC,
                                        firepy.model.building.Building],
                       library: ObjectLibrary = None, **kwargs) -> CostResult:
        """
        Calculate the life cycle costs of Firepy objects

        :param obj: any Firepy object to calculate the costs of
        :param library: object library that can return the objects for the references
        :return: CostResult with all the calculated costs
        """

        # check if this is already calculated:
        if obj.IuId in self.cost_results:
            # if yes, return the result
            return self.cost_results[obj.IuId]

        else:
            if isinstance(obj, (firepy.model.building.OpaqueMaterial,
                                firepy.model.building.WindowMaterial,
                                firepy.model.building.ShadeMaterial,
                                firepy.model.building.BlindMaterial)):

                if self.considered is not None and getattr(obj, self.match_prop) not in self.considered:
                    return self.__null_result(obj)

                else:
                    logger.debug('Calculating cost of {t}: {n}'.format(t=obj.__class__.__name__, n=obj.Name))

                    if isinstance(obj, firepy.model.building.OpaqueMaterial):
                        return self.__opaque_material(obj, **kwargs)  # **life_time_overwrites

                    elif isinstance(obj, firepy.model.building.WindowMaterial):
                        return self.__window_material(obj)

                    elif isinstance(obj, firepy.model.building.ShadeMaterial):
                        return self.__shade_or_blind_material(obj)

                    elif isinstance(obj, firepy.model.building.BlindMaterial):
                        return self.__shade_or_blind_material(obj)

            else:
                logger.debug('Calculating cost of {t}: {n}'.format(t=obj.__class__.__name__, n=obj.Name))

                if isinstance(obj, firepy.model.building.Shading):
                    return self.__shading(obj, library)

                elif isinstance(obj, firepy.model.building.Construction):
                    return self.__construction(obj, library, **kwargs)  # typ: 'opaque' / 'window' / 'shading'

                elif isinstance(obj, firepy.model.building.BuildingSurface):
                    return self.__building_surface(obj, library)

                elif isinstance(obj, firepy.model.building.FenestrationSurface):
                    return self.__fenestration_surface(obj, library)

                elif isinstance(obj, firepy.model.building.NonZoneSurface):
                    return self.__non_zone_surface(obj, library)

                elif isinstance(obj, firepy.model.building.InternalMass):
                    return self.__internal_mass(obj, library)

                elif isinstance(obj, firepy.model.building.Zone):
                    return self.__zone(obj, library)

                elif isinstance(obj, firepy.model.hvac.Heating):
                    return self.__heating(obj, **kwargs)  # heating_demand

                elif isinstance(obj, firepy.model.hvac.Cooling):
                    return self.__cooling(obj, **kwargs)  # cooling_demand

                elif isinstance(obj, firepy.model.hvac.Lighting):
                    return self.__lighting(obj, **kwargs)  # lighting_energy

                elif isinstance(obj, firepy.model.hvac.HVAC):
                    return self.__hvac(obj, **kwargs)  # demands

                elif isinstance(obj, firepy.model.building.Building):
                    return self.__building(obj, **kwargs)  # demands)

    def __null_result(self, obj) -> CostResult:
        # initiate empty result
        cost_result = CostResult(ref_unit='')
        # Add the result to the collection of results
        self.cost_results[obj.IuId] = cost_result

        return cost_result

    def __production(self, material: str, weight: float = None,
                     volume: float = None, area: float = 1, fraction: float = 1) -> float:  # Name or DbId

        """
        Function to calculate the production costs of materials (OpaqueMaterial, WindowMaterial, ShadingMaterial)
        :param material: Name or DbId of the material
        :param fraction: in fraction of weight or area or volume to multiply the amount with
        :param weight: in kg (needed only if the cost data is in kg reference)
        :param volume: in m3 (needed only if the cost data is in m3 reference)
        :param area: in m2 if cost data is in m2 reference (this is the default)
        :return: production cost
        """

        cost_id = self.life_cycle_data.loc[material, 'CostId']

        cost_data = self.cost_data.loc[cost_id, :]  # pd.Series with MultiIndex

        if cost_data['Units', 'Production'] == 'kg':
            if weight is None:
                raise UnitOfMeasurementError('Please provide weight value for material: {mat}'.format(mat=material))
            cost = cost_data['Costs', 'Production'] * weight * fraction  # float

        elif cost_data['Units', 'Production'] == 'm2':
            cost = cost_data['Costs', 'Production'] * area * fraction  # float

        elif cost_data['Units', 'Production'] == 'm3':
            if volume is None:
                raise UnitOfMeasurementError('Please provide volume value for material: {mat}'.format(mat=material))
            cost = cost_data['Costs', 'Production'] * volume * fraction  # float

        else:

            message = 'Unit of material in model does not match the unit of material production in cost data:\n'
            message += '{mat} - {mat_u} <-> {i_u} - {i}'.format(
                mat=material, mat_u='kg, m2 or m3',
                i=cost_data['Metadata', 'Name'], i_u=cost_data['Units', 'Production']
            )
            raise UnitOfMeasurementError(message)

        return cost

    def __installation(self, material: str, weight: float = None,
                       volume: float = None, area: float = 1, fraction: float = 1) -> float:  # Name or DbId):
        """
        Function to calculate the installation costs of materials (OpaqueMaterial, WindowMaterial, ShadingMaterial)
        :param material: Name or DbId of the material
        :param fraction: in fraction of weight or area or volume to multiply the amount with
        :param weight: in kg (needed only if the cost data is in kg reference)
        :param volume: in m3 (needed only if the cost data is in m3 reference)
        :param area: in m2 if cost data is in m2 reference (this is the default)
        :return: installation cost
        """

        cost_id = self.life_cycle_data.loc[material, 'CostId']

        cost_data = self.cost_data.loc[cost_id, :]  # pd.Series with MultiIndex

        if cost_data['Units', 'Installation'] == 'kg':
            if weight is None:
                raise UnitOfMeasurementError('Please provide weight value for material: {mat}'.format(mat=material))
            cost = cost_data['Costs', 'Installation'] * weight * fraction  # float

        elif cost_data['Units', 'Installation'] == 'm2':
            cost = cost_data['Costs', 'Installation'] * area * fraction  # float

        elif cost_data['Units', 'Installation'] == 'm3':
            if volume is None:
                raise UnitOfMeasurementError('Please provide volume value for material: {mat}'.format(mat=material))
            cost = cost_data['Costs', 'Installation'] * volume * fraction  # float

        else:

            message = 'Unit of material in model does not match the unit of material installation in cost data:\n'
            message += '{mat} - {mat_u} <-> {i_u} - {i}'.format(
                mat=material, mat_u='kg, m2 or m3',
                i=cost_data['Metadata', 'Name'], i_u=cost_data['Units', 'Installation']
            )
            raise UnitOfMeasurementError(message)

        return cost

    def __replacement(self, material: str, life_time: int, weight: float = None,
                      volume: float = None, area: float = 1, fraction: float = 1) -> float:
        """
        Function to calculate the replacement cost of materials (OpaqueMaterial, WindowMaterial, ShadingMaterial
        :param material: Name or DbId of the material
        :param fraction: the amount to be replaced in fraction of weight, volume or area
        :param life_time: in years
        :param weight: in kg (needed only if the cost data is in kg reference)
        :param volume: in m3 (needed only if the cost data is in m3 reference)
        :param area: in m2 if cost data is in m2 reference (this is the default)
        :return: replacement cost

        """

        # count of replacements
        replacement_count = (self.rsp - 1) // life_time
        # -1 because we want to make sure that if the rsp equals to the lifetime, no replacement is calculated

        replacement = self.__production(material=material, weight=weight, volume=volume, area=area, fraction=fraction)

        replacement += self.__installation(material=material, weight=weight, volume=volume, area=area,
                                           fraction=fraction)

        replacement *= replacement_count

        return replacement

    def __operation(self, energy_source: str, energy_demand: float) -> float:
        """
        Function to calculate the cost of used energy
        :param energy_source: Name or DbId of the energy source
        :param energy_demand: Calculated energy demand in kWh
        :return:
        """
        cost_id = self.life_cycle_data.loc[energy_source, 'CostId']
        cost_data = self.cost_data.loc[cost_id, :]  # pd.Series with MultiIndex

        if cost_data['Units', 'Production'] == 'MJ':
            costs = cost_data['Costs', 'Production'] * energy_demand * 3.6  # float

        elif cost_data['Units', 'Production'] == 'kWh':
            costs = cost_data['Costs', 'Production'] * energy_demand

        else:
            message = 'Unit of energy demand does not match unit of energy production in cost data:\n'
            message += '{en} - {en_u} <-> {i_u} - {i}'.format(
                en=energy_source, en_u='kWh or MJ',
                i=cost_data['Metadata', 'Name'], i_u=cost_data['Metadata', 'Unit']
            )
            raise UnitOfMeasurementError(message)

        return costs

    def __opaque_material(self, material: firepy.model.building.OpaqueMaterial,
                          life_time_overwrites: dict = None) -> CostResult:
        """
        Results refer to 1 m2 of material
        :param material:
        :param life_time_overwrites: if lifetimes are evaluated on construction level this should contain a
            dict with the new lifetimes {match_prop: lifetime}
        :return: cost result in cost / m2 basis

        """

        # initiate the new result
        cost_result = CostResult(ref_unit='m2')

        mat = getattr(material, self.match_prop)  # Name or DbId
        weight = material.Thickness * material.Density  # in kg (/m2)
        volume = material.Thickness  # in m3 (/m2)
        cutting_waste = self.life_cycle_data.loc[mat, 'CuttingWaste']

        # Production
        production = self.__production(
            material=mat,
            weight=weight,
            volume=volume,
            area=1,
            fraction=1 + cutting_waste
        )

        cost_result.costs.loc['Production'] = production

        # Installation
        installation = self.__installation(
            material=mat,
            weight=weight,
            volume=volume,
            area=1,
            fraction=1 + cutting_waste
        )

        cost_result.costs.loc['Installation'] = installation

        # Replacement
        if life_time_overwrites is not None and mat in life_time_overwrites:
            life_time = life_time_overwrites[mat]
        else:
            life_time = self.life_cycle_data.loc[mat, 'LifeTime']

        replacement = self.__replacement(
            material=mat,
            life_time=life_time,
            weight=weight,
            volume=volume,
            area=1,
            fraction=1 + cutting_waste
        )
        # here we could specify if only a part of it is replaced by fraction=replace_fraction * (1 + cutting_waste)

        cost_result.costs.loc['Replacement'] = replacement

        # Add the result to the collection of results
        self.cost_results[material.IuId] = cost_result

        return cost_result

    def __window_material(self, window_material: firepy.model.building.WindowMaterial) -> CostResult:
        """
        This should contain both frame and glazing, also cost data should contain the cost of frame and glazing too
        Results refer to 1 m2 of material
        :param window_material:
        :return:
        """

        # initiate the new result
        cost_result = CostResult(ref_unit='m2')

        mat = getattr(window_material, self.match_prop)  # Name or DbId
        weight = self.life_cycle_data.loc[mat, 'SurfaceWeight']  # kg/m2
        cutting_waste = self.life_cycle_data.loc[mat, 'CuttingWaste']

        # Production
        production = self.__production(
            material=mat,
            weight=weight,  # this is also very unlikely to define cost on weight basis
            volume=None,  # we cannot have the cost in m3
            area=1,
            fraction=1 + cutting_waste
        )

        cost_result.costs.loc['Production'] = production

        # Installation
        installation = self.__installation(
            material=mat,
            weight=weight,  # this is also very unlikely to define cost on weight basis
            volume=None,  # we cannot have the cost in m3
            area=1,
            fraction=1 + cutting_waste
        )

        cost_result.costs.loc['Installation'] = installation

        # Replacement
        life_time = self.life_cycle_data.loc[mat, 'LifeTime']

        replacement = self.__replacement(
            material=mat,
            life_time=life_time,
            weight=weight,
            volume=None,
            area=1,
            fraction=1 + cutting_waste
        )
        # here we could specify if only a part of it is replaced by fraction=replace_fraction * (1 + cutting_waste)

        cost_result.costs.loc['Replacement'] = replacement

        # Add the result to the collection of results
        self.cost_results[window_material.IuId] = cost_result

        return cost_result

    def __shade_or_blind_material(self, material: Union[firepy.model.building.ShadeMaterial,
                                                        firepy.model.building.BlindMaterial]) -> CostResult:
        """
        Results refer to 1 m2 of material
        :param material:
        :return:
        """

        # initiate the new result
        cost_result = CostResult(ref_unit='m2')

        mat = getattr(material, self.match_prop)  # Name or DbId
        if material.Density is not None:
            density = material.Density
        # elif TODO from DB
        elif self.life_cycle_data.loc[mat, 'Density'] is not None:
            density = self.life_cycle_data.loc[mat, 'Density']
        else:
            raise Exception('No density data found for shading material: {m}'.format(m=material.Name))

        weight = material.Thickness * density  # in kg (/m2)
        cutting_waste = self.life_cycle_data.loc[mat, 'CuttingWaste']
        volume = material.Thickness  # in m3 (/m2)

        # Production
        production = self.__production(
            material=mat,
            weight=weight,
            volume=volume,
            area=1,
            fraction=1 + cutting_waste
        )

        cost_result.costs.loc['Production'] = production

        # Installation
        installation = self.__installation(
            material=mat,
            weight=weight,
            volume=volume,
            area=1,
            fraction=1 + cutting_waste
        )

        cost_result.costs.loc['Installation'] = installation

        # Replacement
        life_time = self.life_cycle_data.loc[mat, 'LifeTime']

        replacement = self.__replacement(
            material=mat,
            life_time=life_time,
            weight=weight,
            volume=volume,
            area=1,
            fraction=1 + cutting_waste
        )
        # here we could specify if only a part of it is replaced by fraction=replace_fraction * (1 + cutting_waste)

        cost_result.costs.loc['Replacement'] = replacement

        # Add the result to the collection of results
        self.cost_results[material.IuId] = cost_result

        return cost_result

    def __shading(self, shading: firepy.model.building.Shading, library: ObjectLibrary) -> CostResult:
        """
        Results refer to 1 m2 of window (vertical) area
        :param shading:
        :param library: ObjectLibrary collection that holds all shading objects
        :return:
        """

        # Initiate cost result
        cost_result = CostResult(ref_unit='m2')  # of window area

        if shading.Material is not None:
            # Get material object from library
            material = library.get(shading.Material)

            # calculate the cost of the shading material
            material_cost = self.calculate_cost(material)

            # calculate to window m2
            m2_cost = material_cost.costs.mul(material.area_per_window_m2())

        elif shading.Construction is not None:
            # Get construction object from library
            construction = library.get(shading.Construction)

            # calculate the cost of the construction
            m2_cost = self.calculate_cost(construction, library, typ='shading')
            # cost is calculated for window m2 in construction
        else:
            raise Exception('Neither shading construction, nor shading material is defined: {n}'.format(n=shading.Name))

        # add the cost to the shading
        cost_result.costs = cost_result.costs.add(m2_cost.costs, fill_value=0)

        self.cost_results[shading.IuId] = cost_result

        return cost_result

    def evaluate_construction_lifetimes(self, construction: firepy.model.building.Construction,
                                        library: ObjectLibrary) -> Mapping[str, float]:

        # TODO how to know which is the core layer?
        # for now we just leave the innermost 2 layers as they are (assuming that the core is the second from inside)
        # in future check the material if it is a WindowMaterial or GlazingMaterial

        life_times = {}
        layers = []

        # collect initial data
        # layer order outside to inside
        for material_ref in construction.Layers:

            # Get material object from library
            material = library.get(material_ref)

            mat = getattr(material, self.match_prop)  # Name or DbId
            life_time = self.life_cycle_data.loc[mat, 'LifeTime']
            life_times[mat] = life_time
            layers.append(mat)

        if len(construction.Layers) <= 2:
            # nothing to evaluate
            return life_times

        # reverse order an collect name (or dbid)
        layers_rev = [mat for mat in layers[::-1]]

        for inner, outer in zip(layers_rev[1:], layers_rev[2:]):
            # changes made from the second innermost layer
            if life_times[inner] < life_times[outer]:
                life_times[inner] = life_times[outer]

        return life_times

    def __construction(self, construction: firepy.model.building.Construction, library: ObjectLibrary,
                       typ: str = 'opaque') -> CostResult:

        """
        Results refer to 1 m2 surface made of this construction
        :param construction:
        :param library: ObjectLibrary collection that holds all shading objects
        :param typ: 'opaque' / 'window' / 'shading'
        :return:
        """

        # Initiate cost result
        cost_result = CostResult(ref_unit='m2')
        if typ == 'opaque':
            # check lifetimes based on layer order
            life_times = self.evaluate_construction_lifetimes(construction, library)

            for layer in construction.Layers:
                # Get material object from library
                material = library.get(layer)

                # calculate the cost of the material
                material_cost = self.calculate_cost(material, life_time_overwrites=life_times)

                # add the cost to the construction cost
                cost_result.costs = cost_result.costs.add(material_cost.costs, fill_value=0)

        elif typ == 'window':
            for layer in construction.Layers:
                # Get material object from library
                material = library.get(layer)

                # calculate the cost of the material
                material_cost = self.calculate_cost(material)

                # add the cost to the construction cost
                cost_result.costs = cost_result.costs.add(material_cost.costs, fill_value=0)

        elif typ == 'shading':
            for layer in construction.Layers:
                # Get material object from library
                material = library.get(layer)

                # add shading only, not the window
                if isinstance(material, (firepy.model.building.BlindMaterial, firepy.model.building.ShadeMaterial)):
                    # calculate the cost of the material
                    material_cost = self.calculate_cost(material)

                    # calculate cost based on window m2 and add the cost to the construction cost
                    cost_per_m2 = material_cost.costs.mul(material.area_per_window_m2())
                    cost_result.costs = cost_result.costs.add(cost_per_m2, fill_value=0)

        else:
            raise Exception('Invalid construction cost calculation type: {t}'.format(t=typ))

        self.cost_results[construction.IuId] = cost_result

        return cost_result

    def __building_surface(self, building_surface: firepy.model.building.BuildingSurface,
                           library: ObjectLibrary) -> CostResult:
        """
        Cost refers to the total surface including cost of windows
        :param building_surface:
        :param library: ObjectLibrary collection that holds all shading objects
        :return:
        """

        # Initiate cost result
        cost_result = CostResult(ref_unit='total')

        # Add cost of all windows to the total
        for window in building_surface.Fenestration:
            window_cost = self.calculate_cost(window, library)
            cost_result.costs = cost_result.costs.add(window_cost.costs, fill_value=0)

        # Get construction of the surface
        construction = library.get(building_surface.Construction)
        construction_cost = self.calculate_cost(construction, library, typ='opaque')

        # Add cost of opaque surface to the total (excluding windows)
        opaque_cost = construction_cost.costs.mul(building_surface.area_net() * 1.15)
        # TODO multiplication with 1.15 to include materials at junctions (inside reference surface)

        cost_result.costs = cost_result.costs.add(opaque_cost, fill_value=0)

        self.cost_results[building_surface.IuId] = cost_result

        return cost_result

    def __fenestration_surface(self, fenestration_surface: firepy.model.building.FenestrationSurface,
                               library: ObjectLibrary) -> CostResult:
        """
        Cost refers to the total surface including cost of shading
        :param fenestration_surface:
        :param library:
        :return:
        """
        # TODO cost refers to window construction which should include frame and glazing
        # In the future separate glazing and frame

        # Initiate cost result
        cost_result = CostResult(ref_unit='total')

        # Get shading of the window
        shading = library.get(fenestration_surface.Shading)

        shading_cost = self.calculate_cost(shading, library)

        # Add cost of shading to the total
        shading_total_cost = shading_cost.costs.mul(fenestration_surface.area())

        cost_result.costs = cost_result.costs.add(shading_total_cost, fill_value=0)

        # Get construction of the surface
        construction = library.get(fenestration_surface.Construction)
        construction_cost = self.calculate_cost(construction, library, typ='window')

        # Add cost of window surface to the total
        # TODO from old calculation if frame and glazing defined separately:
        # window_lca = (glazing_lca * window.glazing_area() + frame_lca * window.frame_area()) / window.area()
        window_cost = construction_cost.costs.mul(fenestration_surface.area())

        cost_result.costs = cost_result.costs.add(window_cost, fill_value=0)

        self.cost_results[fenestration_surface.IuId] = cost_result

        return cost_result

    def __non_zone_surface(self, non_zone_surface: firepy.model.building.NonZoneSurface,
                           library: ObjectLibrary) -> CostResult:
        """
        Cost refers to the total surface
        :param non_zone_surface:
        :param library:
        :return:
        """

        # Initiate cost result
        cost_result = CostResult(ref_unit='total')

        # Get construction of the surface

        construction = library.get(non_zone_surface.Construction)
        construction_cost = self.calculate_cost(construction, library, typ='opaque')

        # Calculate cost of the total area
        surface_cost = construction_cost.costs.mul(non_zone_surface.area())

        cost_result.costs = cost_result.costs.add(surface_cost, fill_value=0)

        self.cost_results[non_zone_surface.IuId] = cost_result

        return cost_result

    def __internal_mass(self, internal_mass: firepy.model.building.InternalMass, library: ObjectLibrary) -> CostResult:
        """
        Cost refers to the total
        :param internal_mass:
        :param library:
        :return:
        """

        # Initiate cost result
        cost_result = CostResult(ref_unit='total')

        # Get construction of the surface
        construction = library.get(internal_mass.Construction)
        construction_cost = self.calculate_cost(construction, library, typ='opaque')

        # Calculate cost of the total area
        mass_cost = construction_cost.costs.mul(internal_mass.Area)
        cost_result.costs = cost_result.costs.add(mass_cost, fill_value=0)

        self.cost_results[internal_mass.IuId] = cost_result

        return cost_result

    def __zone(self, zone: firepy.model.building.Zone, library: ObjectLibrary) -> CostResult:
        """
        Cost refers to total of zone
        :param zone:
        :param library:
        :return:
        """

        # Initiate cost result

        cost_result = CostResult(ref_unit='m2')
        # add cost of all surfaces

        for surface in zone.BuildingSurfaces:
            # calculate the cost of the surface
            surface_cost = self.calculate_cost(surface, library)

            # add the cost to the zone cost
            cost_result.costs = cost_result.costs.add(surface_cost.costs, fill_value=0)

        # add cost of all internal masses
        for mass in zone.InternalMasses:
            # calculate the cost of the internal mass
            mass_cost = self.calculate_cost(mass, library)

            # add the cost to the zone cost
            cost_result.costs = cost_result.costs.add(mass_cost.costs, fill_value=0)

        self.cost_results[zone.IuId] = cost_result

        return cost_result

    def __building(self, building: firepy.model.building.Building, demands: pd.DataFrame) -> CostResult:
        """
        Cost refers to total of building
        :param building:
        :return:
        """

        # Initiate cost result
        cost_result = CostResult(ref_unit='m2')

        library = building.Library

        # add cost of all zones
        for zone in building.Zones:
            # calculate the cost of the zone
            zone_cost = self.calculate_cost(zone, library)

            # add the cost to the building cost
            cost_result.costs = cost_result.costs.add(zone_cost.costs, fill_value=0)

        # add cost of all non-zone surfaces
        for surface in building.NonZoneSurfaces:
            # calculate the cost of the surface
            surface_cost = self.calculate_cost(surface, library)

            # add the cost to the building cost
            cost_result.costs = cost_result.costs.add(surface_cost.costs, fill_value=0)

        # TODO add cost of HVAC systems and use phase
        # add cost of operation
        operation_cost = self.calculate_cost(building.HVAC, demands=demands)

        # add the hvac cost to the building cost
        cost_result.costs = cost_result.costs.add(operation_cost.costs, fill_value=0)

        self.cost_results[building.IuId] = cost_result

        return cost_result

    def __heating(self, heating: firepy.model.hvac.Heating, heating_demand: float) -> cost_results:
        """
        Impact refers to one year
        :param heating:
        :param heating_demand: Net value of heating demand in kWh/year
        :return:
        """
        # Initiate impact result
        cost_result = CostResult(ref_unit='year')
        energy_source = heating.energy_source  # Name or DbId
        yearly_demand = heating_demand / heating.efficiency  # gross demand

        # Operation
        operation = self.__operation(
            energy_source=energy_source,
            energy_demand=yearly_demand
        )

        cost_result.costs.loc['Operation'] = operation

        # Add the result to the collection of results
        self.cost_results[heating.IuId] = cost_result

        return cost_result

    def __cooling(self, cooling: firepy.model.hvac.Cooling, cooling_demand: float) -> cost_results:
        """
        Impact refers to one year
        :param cooling:
        :param cooling_demand: Net value of heating demand in kWh/year
        :return:
        """
        # Initiate impact result
        cost_result = CostResult(ref_unit='year')
        energy_source = cooling.energy_source  # Name or DbId
        yearly_demand = cooling_demand / cooling.efficiency  # gross demand

        # Operation
        operation = self.__operation(
            energy_source=energy_source,
            energy_demand=yearly_demand
        )

        cost_result.costs.loc['Operation'] = operation

        # Add the result to the collection of results
        self.cost_results[cooling.IuId] = cost_result

        return cost_result

    def __lighting(self, lighting: firepy.model.hvac.Lighting, lighting_energy: float) -> cost_results:
        """
        Impact refers to one year
        :param lighting:
        :param lighting_energy: Total lighting (electric) energy in kWh/year
        :return:
        """
        # Initiate impact result
        cost_result = CostResult(ref_unit='year')
        energy_source = lighting.energy_source  # Name or DbId
        yearly_demand = lighting_energy * lighting.inefficiency  # gross demand

        # Operation
        operation = self.__operation(
            energy_source=energy_source,
            energy_demand=yearly_demand
        )

        cost_result.costs.loc['Operation'] = operation

        # Add the result to the collection of results
        self.cost_results[lighting.IuId] = cost_result

        return cost_result

    def __hvac(self, hvac: firepy.model.hvac.HVAC, demands: pd.DataFrame) -> cost_results:
        """
        Impact refers to total reference period
        :param hvac:
        :param demands: pandas DataFrame with three columns named 'heating', 'cooling' and 'lights'
            containing the yearly impact in the sum of the columns in kWh
            e.g. output from energy calculation
        :return:
        """
        # Initiate impact result
        cost_result = CostResult(ref_unit='total')

        # Yearly impact of heating and cooling
        heating_demand = abs(demands.loc[:, 'heating'].sum())
        cooling_demand = abs(demands.loc[:, 'cooling'].sum())  # in case it would be negative
        lighting_energy = abs(demands.loc[:, 'lights'].sum())

        heating_cost = self.calculate_cost(hvac.Heating, heating_demand=heating_demand)
        cooling_cost = self.calculate_cost(hvac.Cooling, cooling_demand=cooling_demand)
        lighting_cost = self.calculate_cost(hvac.Lighting, lighting_energy=lighting_energy)

        # Add impact to the total
        cost_result.costs = cost_result.costs.add(heating_cost.costs.mul(self.rsp), fill_value=0)
        cost_result.costs = cost_result.costs.add(cooling_cost.costs.mul(self.rsp), fill_value=0)
        cost_result.costs = cost_result.costs.add(lighting_cost.costs.mul(self.rsp), fill_value=0)

        # Add the result to the collection of results
        self.cost_results[hvac.IuId] = cost_result

        return cost_result


class UnitOfMeasurementError(Exception):
    pass
