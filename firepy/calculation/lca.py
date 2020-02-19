from typing import Mapping, Union, MutableMapping, List
from pathlib import Path
import logging

import pandas as pd

import firepy.model.building
import firepy.model.hvac
from firepy.model.building import ObjectLibrary
# from firepy.tools.database import SqlDB, OLCA

logger = logging.getLogger(__name__)

IMPACT_CATEGORIES = {
    'GWP 100a': {
        'Name': 'climate change (GWP 100a)',
        'Impact Assessment Method': 'CML 2001',
        'Impact Unit': 'kg-CO2-eq'
    },
    'AP': {
        'Name': 'acidification potential (average European)',
        'Impact Assessment Method': 'CML 2001',
        'Impact Unit': 'kg-SO2-eq'
    },
    'CED biomass': {
        'Name': 'biomass (renewable energy resources, biomass)',
        'Impact Assessment Method': 'cumulative energy demand',
        'Impact Unit': 'MJ-eq'
    },
    'CED fossil': {
        'Name': 'fossil (non-renewable energy resources, fossil)',
        'Impact Assessment Method': 'cumulative energy demand',
        'Impact Unit': 'MJ-eq'
    },
    'CED renewable': {
        'Name': 'renewable energy resources (biomass, geothermal, solar, water, wind)',
        'Impact Assessment Method': 'cumulative energy demand',
        'Impact Unit': 'MJ-eq'
    },
    'CED non-renewable': {
        'Name': 'non-renewable energy resources (fossil, nuclear, primary forest)',
        'Impact Assessment Method': 'cumulative energy demand',
        'Impact Unit': 'MJ-eq'
    }
}

LIFE_CYCLE_STAGES = {
    'A1-3': {
        'Name': 'Product Stage',
        'Modules': {
            'A1': 'Raw material supply',
            'A2': 'Transport',
            'A3': 'Manufacturing',
        }
    },
    'A4-5': {
        'Name': 'Construction Process Stage',
        'Modules': {
            'A4': 'Transport',
            'A5': 'Construction-installation process',
        }
    },
    'B1-7': {
        'Name': 'Use Stage',
        'Modules': {
            'B1': 'Use',
            'B2': 'Maintenance',
            'B3': 'Repair',
            'B4': 'Replacement',
            'B5': 'Refurbishment',
            'B6': 'Operational energy use',
            'B7': 'Operational water use',
        }
    },
    'C1-4': {
        'Name': 'End Of Life Stage',
        'Modules': {
            'C1': 'De-construction demolition',
            'C2': 'Transport',
            'C3': 'Waste processing',
            'C4': 'Disposal',
        }
    },
}


class Impact:
    # TODO unused
    def __init__(self, impact_category: str, value: float, stage: str):
        self.ImpactAssessmentMethod = IMPACT_CATEGORIES[impact_category]['Impact Assessment Method']
        self.ImpactCategory = impact_category  # e.g. GWP 100a
        self.ImpactCategoryName = IMPACT_CATEGORIES[impact_category]['Name']  # e.g. CML2001
        self.ImpactUnit = IMPACT_CATEGORIES[impact_category]['Impact Unit']  # e.g. kg CO2-eq.
        self.Value = value
        self.LifeCycleStage = stage  # e.g. A1 / Production

    # def to_series(self) -> pd.Series:
    #     return pd.Series(data=self.__dict__, name=self.ImpactCategory)


class ImpactResult:
    """
    A collection of all calculated impacts of an object (e.g. Construction) as Impact instances
    """

    def __init__(self, basis_unit):
        self.BasisUnit = basis_unit  # the unit of object that the impact refers to (e.g. m2 for a Construction)

        stages = ['A1-3', 'A4-5', 'B1-7', 'C1-4'] + \
                 ['A{}'.format(i + 1) for i in range(5)] + \
                 ['B{}'.format(i + 1) for i in range(7)] + \
                 ['C{}'.format(i + 1) for i in range(4)]

        df = pd.DataFrame(columns=stages)
 
        self._impacts = df

    @property
    def impacts(self) -> pd.DataFrame:
        """
        Get impact results as Pandas DataFrame

        :return: pandas DataFrame
            - columns: life cycle stages
            - index: impact category
        """
        return self._impacts

    @impacts.setter
    def impacts(self, new: pd.DataFrame):
        self._impacts = new


class InventoryItem:
    # TODO unused

    def __init__(self, typ: str, amount: float, unit: str, name: str, db_id: str):
        self.Type = typ  # Material, Energy, Transport, WasteTreatment
        self.TotalAmount = amount
        self.Unit = unit
        self.Name = name
        self.DbId = db_id


class Inventory:
    # TODO unused

    def __init__(self, ref_unit: str):
        self.ReferenceUnit = ref_unit
        self._items = pd.DataFrame(columns=[])

    @property
    def items(self) -> pd.DataFrame:
        """
        Get impact results as Pandas DataFrame

        :return: pandas DataFrame
            columns: Name, DbId, Type, Unit, TotalAmount (Production, Installation, Maintenance, Disposal, Transport?)
            rows: all items (Materials, Energy, Transport, Waste treatment)
            index: DbId (olca id?)
        """
        return self._items

    def as_frame(self) -> pd.DataFrame:
        """
        Get impact results as Pandas DataFrame

        :return: pandas DataFrame
            columns: Name, DbId, Type, Unit, TotalAmount
            rows: all items (Materials, Energy, Transport, Waste treatment)
        """
        # TODO
        pass


class LCACalculation:

    def __init__(self, reference_service_period: int = 50,
                 life_cycle_data: Union[str, pd.DataFrame] = None,
                 impact_data: Union[str, pd.DataFrame] = None,
                 db = None, olca = None,
                 matching_col: str = 'DbId', matching_property: str = 'DbId',
                 considered_objects: List[str] = None):
        """

        :param reference_service_period: the reference service period in years
        :param life_cycle_data:
        :param impact_data:
        :param db:
        :param olca:
        :param matching_col:
        :param matching_property:
        :param considered_objects: list of DbIds or Names - calculate only the objects that are in the list
            if None, all objects are calculated
        """
        # these are basically cache objects
        self._impact_results = {}  # Dict of calculated impact of objects {IuId: ImpactResult} Mapping[str, Impact]
        # self._inventories = {}  # for all objects calculate the amount of referenced objects in total
        # (opaque_mat, window_mat, shading_mat, [construction and shading])

        # the reference service period in years
        self.rsp = reference_service_period

        # # what impacts will be calculated
        # this is unused, calculations are made for all available impact categories from impact_data
        # self._impact_categories = None

        # how the model objects will be matched with the LifeCycle Data (column name in the life_cycle_data
        self.match_col = matching_col
        # what property of the model objects (e.g. materials) to use for the matching ('DbId' or 'Name')
        self.match_prop = matching_property

        self.considered = considered_objects

        # Life Cycle Data can be given:
        #   - directly by a DataFrame or a csv address
        #   - from a database if DbId-s are supplied within the model (database_connection)
        self.LifeCycleData = life_cycle_data  # DataFrame

        # Impact Data (LCIA) can be given:
        #   - directly by a DataFrame or a csv address
        #   - from a database if DbId-s are supplied within the model (database_connection)
        #   - by calculating it with OpenLCA ipc server (olca_server_connection)
        self.ImpactData = impact_data  # DataFrame

        self.impact_categories = self.__impact_categories()

        self._db = db  # TODO SqlDB instance to connect to the database if no ready results are supplied
        self._olca = olca  # TODO OLCA instance to calculate the impacts directly from ecoinvent

        # we have to feed the class with either:
        #   - LifeCycleData AND ImpactData
        #   - DB
        #   - LifeCycleData AND olca

    @property
    def impact_results(self) -> MutableMapping[str, ImpactResult]:
        return self._impact_results

    # @property
    # def Inventories(self) -> Mapping[str, Inventory]:
    #     return self._inventories

    # @property
    # def impact_categories(self) -> List[str]:
    #     return self._impact_categories

    @property
    def LifeCycleData(self) -> pd.DataFrame:
        """
        pandas DataFrame with all the life cycle information
        Columns:
          - Name, DbId, Unit, openLCAname, openLCAid, ProductionId, TransportId, WasteTreatmentId,
            WasteTreatmentTransportId, LifeTime, CuttingWaste, [SurfaceWeight {window}], [Density {shading material}]
        Rows:
          - Materials
        Index: DbId

        :return: DataFrame
        """

        return self._life_cycle_data

    @LifeCycleData.setter
    def LifeCycleData(self, source: Union[str, Path, pd.DataFrame]):
        if isinstance(source, str):
            # Read from path
            self._life_cycle_data = pd.read_csv(source, index_col=self.match_col)
        elif isinstance(source, Path):
            self._life_cycle_data = pd.read_csv(str(source), index_col=self.match_col)
        elif isinstance(source, pd.DataFrame):
            source.set_index(self.match_col)
            self._life_cycle_data = source

    @property
    def ImpactData(self) -> pd.DataFrame:
        """
        pandas DataFrame with all the environmental impacts
        Columns: (Multiindex)
          - Metadata[Name, DbId, Unit], Impact categories[GWP, CED, ...]
        Rows:
          - Material / Energy / Waste Treatment / Transport
        Index: DbId
        """
        return self._impact_data

    @ImpactData.setter
    def ImpactData(self, source: Union[str, Path, pd.DataFrame]):
        if isinstance(source, str):
            # Read from path
            self._impact_data = pd.read_csv(source, index_col=0, header=[0, 1])
        elif isinstance(source, Path):
            self._impact_data = pd.read_csv(str(source), index_col=0, header=[0, 1])
        elif isinstance(source, pd.DataFrame):
            # TODO sql db will need to serve the same methods as used for a DataFrame
            self._impact_data = source

    def __impact_categories(self):
        return self.ImpactData['Impact categories'].columns.to_list()

    def clear_cache(self):
        self._impact_results = {}

    def calculate_impact(self, obj: Union[firepy.model.building.OpaqueMaterial,
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
                         library: ObjectLibrary = None, **kwargs) -> ImpactResult:
        """
        Calculate the environmental impact of Firepy objects

        :param obj: Any Firepy object to calculate the inventory of
        :param library: object library that can return the objects for the references
        :return: ImpactResult with all the calculated impacts
        """

        # check if this is already calculated:
        if obj.IuId in self.impact_results:
            # if yes, return the result
            return self.impact_results[obj.IuId]

        else:
            if isinstance(obj, (firepy.model.building.OpaqueMaterial,
                                firepy.model.building.WindowMaterial,
                                firepy.model.building.ShadeMaterial,
                                firepy.model.building.BlindMaterial)):

                if self.considered is not None and getattr(obj, self.match_prop) not in self.considered:
                    return self.__null_result(obj)

                else:
                    logger.debug('Calculating impact of {t}: {n}'.format(t=obj.__class__.__name__, n=obj.Name))

                    if isinstance(obj, firepy.model.building.OpaqueMaterial):
                        return self.__opaque_material(obj, **kwargs)  # **life_time_overwrites

                    elif isinstance(obj, firepy.model.building.WindowMaterial):
                        return self.__window_material(obj)

                    elif isinstance(obj, firepy.model.building.ShadeMaterial):
                        return self.__shade_or_blind_material(obj)

                    elif isinstance(obj, firepy.model.building.BlindMaterial):
                        return self.__shade_or_blind_material(obj)

            else:
                logger.debug('Calculating impact of {t}: {n}'.format(t=obj.__class__.__name__, n=obj.Name))

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
                    return self.__building(obj, **kwargs)  # demands

    def __null_result(self, obj) -> ImpactResult:
        logger.debug('Returning NullResult for {t}: {n}'.format(t=obj.__class__.__name__, n=obj.Name))
        # initiate empty result
        impact_result = ImpactResult(basis_unit='')

        # Add the result to the collection of results
        self.impact_results[obj.IuId] = impact_result

        return impact_result

    def __production(self, material: str, weight: float = None, fraction: float = 1) -> pd.Series:  # Name or DbId
        """
        Function to calculate the production impact of materials (OpaqueMaterial, WindowMaterial, ShadingMaterial
        :param material: Name or DbId of the material
        :param fraction: in fraction of weight or area to multiply the amount with
        :param weight: in kg (needed only if the impact data is not in m2 reference)
        :return: impact for all impact categories
        """
        production_id = self.LifeCycleData.loc[material, 'ProductionId']
        impact_data = self.ImpactData.loc[production_id, :]  # pd.Series with MultiIndex

        if impact_data['Metadata', 'Unit'] == 'kg':
            if weight is None:
                raise UnitOfMeasurementError('Please provide weight value for material: {mat}'.format(mat=material))
            impacts = impact_data['Impact categories'] * weight * fraction  # pd.Series SingleIndex

        elif impact_data['Metadata', 'Unit'] == 'm2':
            impacts = impact_data['Impact categories'] * fraction  # pd.Series SingleIndex

        else:
            message = 'Unit of material in model does not match the unit of material in impact data:\n'
            message += '{mat} - {mat_u} <-> {i_u} - {i}'.format(
                mat=material, mat_u='kg or m2',
                i=impact_data['Metadata', 'Name'], i_u=impact_data['Metadata', 'Unit']
            )
            raise UnitOfMeasurementError(message)
        return impacts

    def __transport(self, transport: str, weight: float) -> pd.Series:
        """
        Function to calculate the transportation impact of materials or waste
        :param transport: DbId or Name of material/waste
        :param weight: in kg
        :return: transport impact for all impact categories
        """
        transport_data = self.ImpactData.loc[transport, :]

        if transport_data['Metadata', 'Unit'] == 'kg':
            transport = transport_data['Impact categories'] * weight  # pd.Series SingleIndex

        else:
            message = 'Unit of material/waste in model does not match the unit of transportation in impact data:\n'
            message += 'Material/Waste {mat_u} <-> {i_u} - {i}'.format(
                mat_u='kg',
                i=transport_data['Metadata', 'Name'], i_u=transport_data['Metadata', 'Unit']
            )
            raise UnitOfMeasurementError(message)
        return transport

    def __replacement(self, material: str, life_time: int, weight: float = None, fraction: float = 1) -> pd.Series:
        """
        Function to calculate the replacement impact of materials (OpaqueMaterial, WindowMaterial, ShadingMaterial
        :param material: Name or DbId of the material
        :param fraction: the amount to be replaced in fraction of weight or area
        :param life_time: in years
        :param weight: in kg (needed only if the impact data is not in m2 reference)
        :return:
        """

        # count of replacements
        replacement_count = (self.rsp - 1) // life_time
        # -1 because we want to make sure that if the rsp equals to the lifetime, no replacement is calculated

        transport_id = self.LifeCycleData.loc[material, 'TransportId']

        replacement = self.__production(material=material, weight=weight, fraction=fraction)
        replacement += self.__transport(transport=transport_id, weight=weight)
        replacement += self.__waste_treatment(material=material, weight=weight, fraction=fraction)  # incl cutting_waste

        replacement *= replacement_count
        return replacement

    def __waste_treatment(self, material: str, weight: float, fraction: float = 1):
        waste_scenario = self.LifeCycleData.loc[material, 'WasteTreatmentId']
        waste_data = self.ImpactData.loc[waste_scenario, :]

        if waste_data['Metadata', 'Unit'] == 'kg':
            if weight is None:
                raise UnitOfMeasurementError('Please provide weight value for material: {mat}'.format(mat=material))
            waste_treatment = waste_data['Impact categories'] * weight * fraction

        elif waste_data['Metadata', 'Unit'] == 'm2':
            waste_treatment = waste_data['Impact categories'] * fraction  # pd.Series SingleIndex

        else:
            message = 'Unit of material in model does not match the unit of waste_treatment in impact data:\n'
            message += '{mat} - {mat_u} <-> {i_u} - {i}'.format(
                mat=material, mat_u='kg',
                i=waste_data['Metadata', 'Name'], i_u=waste_data['Metadata', 'Unit']
            )
            raise UnitOfMeasurementError(message)

        waste_transport_id = self.LifeCycleData.loc[material, 'WasteTreatmentTransportId']
        waste_transport = self.__transport(waste_transport_id, weight=weight * fraction)

        waste_treatment += waste_transport

        return waste_treatment

    def __operation(self, energy_source: str, energy_demand: float) -> pd.Series:
        """
        Function to calculate the impact of used energy
        :param energy_source: Name or DbId of the energy source
        :param energy_demand: Calculated energy demand in kWh
        :return:
        """
        production_id = self.LifeCycleData.loc[energy_source, 'ProductionId']
        impact_data = self.ImpactData.loc[production_id, :]  # pd.Series with MultiIndex

        if impact_data['Metadata', 'Unit'] == 'MJ':
            impacts = impact_data['Impact categories'] * energy_demand * 3.6  # pd.Series SingleIndex

        elif impact_data['Metadata', 'Unit'] == 'kWh':
            impacts = impact_data['Impact categories'] * energy_demand

        else:
            message = 'Unit of energy demand does not match unit of energy production in impact data:\n'
            message += '{en} - {en_u} <-> {i_u} - {i}'.format(
                en=energy_source, en_u='kWh or MJ',
                i=impact_data['Metadata', 'Name'], i_u=impact_data['Metadata', 'Unit']
            )
            raise UnitOfMeasurementError(message)

        return impacts

    def __opaque_material(self, material: firepy.model.building.OpaqueMaterial,
                          life_time_overwrites: dict = None) -> ImpactResult:
        """
        Results refer to 1 m2 of material
        :param material:
        :param life_time_overwrites: if lifetimes are evaluated on construction level this should contain a
            dict with the new lifetimes {match_prop: lifetime}
        :return: impact result in impact / m2 basis
        """

        # initiate the new result
        impact_result = ImpactResult(basis_unit='m2')

        mat = getattr(material, self.match_prop)  # Name or DbId
        weight = material.Thickness * material.Density  # in kg (/m2)
        cutting_waste = self.LifeCycleData.loc[mat, 'CuttingWaste']

        # Production
        production = self.__production(
            material=mat,
            weight=weight,
            fraction=1 + cutting_waste
        )
        production_waste = self.__waste_treatment(
            material=mat,
            weight=weight,
            fraction=cutting_waste
        )
        impact_result.impacts.loc[:, 'A1-3'] = production
        impact_result.impacts.loc[:, 'A5'] = production_waste

        # Transport
        transport_id = self.LifeCycleData.loc[mat, 'TransportId']
        transport = self.__transport(
            transport=transport_id,
            weight=weight * (1 + cutting_waste)
        )
        impact_result.impacts.loc[:, 'A4'] = transport

        # Replacement
        if life_time_overwrites is not None and mat in life_time_overwrites:
            life_time = life_time_overwrites[mat]
        else:
            life_time = self.LifeCycleData.loc[mat, 'LifeTime']
        replacement = self.__replacement(
            material=mat,
            life_time=life_time,
            weight=weight,
            fraction=1 + cutting_waste
        )
        # here we could specify if only a part of it is replaced by fraction=replace_fraction * (1 + cutting_waste)
        impact_result.impacts.loc[:, 'B4'] = replacement

        # End of Life
        waste_treatment = self.__waste_treatment(
            material=mat,
            weight=weight,
            fraction=1  # cutting waste treated as production waste, replaced material in replacement
        )
        impact_result.impacts.loc[:, 'C1-4'] = waste_treatment

        # Add the result to the collection of results
        self.impact_results[material.IuId] = impact_result

        return impact_result

    def __window_material(self, window_material: firepy.model.building.WindowMaterial) -> ImpactResult:
        """
        This should contain both frame and glazing, also impact data should contain the impact of frame and glazing too
        Results refer to 1 m2 of material
        :param window_material:
        :return:
        """

        # initiate the new result
        impact_result = ImpactResult(basis_unit='m2')

        mat = getattr(window_material, self.match_prop)  # Name or DbId

        cutting_waste = self.LifeCycleData.loc[mat, 'CuttingWaste']
        weight = self.LifeCycleData.loc[mat, 'SurfaceWeight']  # kg/m2

        # Production
        production = self.__production(
            material=mat,
            weight=weight,
            fraction=1 + cutting_waste
        )
        production_waste = self.__waste_treatment(
            material=mat,
            weight=weight,
            fraction=cutting_waste
        )
        impact_result.impacts.loc[:, 'A1-3'] = production
        impact_result.impacts.loc[:, 'A5'] = production_waste

        # Transport
        transport_id = self.LifeCycleData.loc[mat, 'TransportId']
        transport = self.__transport(
            transport=transport_id,
            weight=weight * (1 + cutting_waste)
        )
        impact_result.impacts.loc[:, 'A4'] = transport

        # Replacement
        life_time = self.LifeCycleData.loc[mat, 'LifeTime']
        replacement = self.__replacement(
            material=mat,
            life_time=life_time,
            weight=weight,
            fraction=1 + cutting_waste
        )
        # here we could specify if only a part of it is replaced by fraction=replace_fraction * (1 + cutting_waste)
        impact_result.impacts.loc[:, 'B4'] = replacement

        # End of Life
        waste_treatment = self.__waste_treatment(
            material=mat,
            weight=weight,
            fraction=1  # cutting waste treated as production waste, replaced material in replacement
        )
        impact_result.impacts.loc[:, 'C1-4'] = waste_treatment

        # Add the result to the collection of results
        self.impact_results[window_material.IuId] = impact_result

        return impact_result

    def __shade_or_blind_material(self, material: Union[firepy.model.building.ShadeMaterial,
                                                        firepy.model.building.BlindMaterial]) -> ImpactResult:
        """
        Results refer to 1 m2 of material
        :param material:
        :return:
        """
        # initiate the new result
        impact_result = ImpactResult(basis_unit='m2')

        mat = getattr(material, self.match_prop)  # Name or DbId

        if material.Density is not None:
            density = material.Density
        # elif TODO from DB
        elif self.LifeCycleData.loc[mat, 'Density'] is not None:
            density = self.LifeCycleData.loc[mat, 'Density']
        else:
            raise Exception('No density data found for shading material: {m}'.format(m=material.Name))

        weight = material.Thickness * density  # in kg (/m2)
        cutting_waste = self.LifeCycleData.loc[mat, 'CuttingWaste']

        # Production
        production = self.__production(
            material=mat,
            weight=weight,
            fraction=1 + cutting_waste
        )
        production_waste = self.__waste_treatment(
            material=mat,
            weight=weight,
            fraction=cutting_waste
        )
        impact_result.impacts.loc[:, 'A1-3'] = production
        impact_result.impacts.loc[:, 'A5'] = production_waste

        # Transport
        transport_id = self.LifeCycleData.loc[mat, 'TransportId']
        transport = self.__transport(
            transport=transport_id,
            weight=weight * (1 + cutting_waste)
        )
        impact_result.impacts.loc[:, 'A4'] = transport

        # Replacement
        life_time = self.LifeCycleData.loc[mat, 'LifeTime']
        replacement = self.__replacement(
            material=mat,
            life_time=life_time,
            weight=weight,
            fraction=1 + cutting_waste
        )
        # here we could specify if only a part of it is replaced by fraction=replace_fraction * (1 + cutting_waste)
        impact_result.impacts.loc[:, 'B4'] = replacement

        # End of Life
        waste_treatment = self.__waste_treatment(
            material=mat,
            weight=weight,
            fraction=1  # cutting waste treated as production waste, replaced material in replacement
        )
        impact_result.impacts.loc[:, 'C1-4'] = waste_treatment

        # Add the result to the collection of results
        self.impact_results[material.IuId] = impact_result

        return impact_result

    def __shading(self, shading: firepy.model.building.Shading, library: ObjectLibrary) -> ImpactResult:
        """
        Results refer to 1 m2 of window (vertical) area
        :param shading:
        :param library: ObjectLibrary collection that holds all shading objects
        :return:
        """
        # Initiate impact result
        impact_result = ImpactResult(basis_unit='m2')  # of window area

        if shading.Material is not None:
            # Get material object from library
            material = library.get(shading.Material)

            # calculate the impact of the shading material
            material_impact = self.calculate_impact(material)

            # calculate to window m2
            m2_impact = material_impact.impacts.mul(material.area_per_window_m2())

        elif shading.Construction is not None:
            # Get construction object from library
            construction = library.get(shading.Construction)

            # calculate the impact of the construction
            m2_impact = self.calculate_impact(construction, library, typ='shading')
            # impact is calculated for window m2 in construction
        else:
            raise Exception('Neither shading construction, nor shading material is defined: {n}'.format(n=shading.Name))

        # add the impact to the shading
        impact_result.impacts = impact_result.impacts.add(m2_impact.impacts, fill_value=0)

        self.impact_results[shading.IuId] = impact_result

        return impact_result

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
            life_time = self.LifeCycleData.loc[mat, 'LifeTime']
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
                       typ: str = 'opaque') -> ImpactResult:
        """
        Results refer to 1 m2 surface made of this construction
        :param construction:
        :param library: ObjectLibrary collection that holds all shading objects
        :param typ: 'opaque' / 'window' / 'shading'
        :return:
        """
        # Initiate impact result
        impact_result = ImpactResult(basis_unit='m2')

        if typ == 'opaque':
            # check lifetimes based on layer order
            life_times = self.evaluate_construction_lifetimes(construction, library)

            for layer in construction.Layers:

                # Get material object from library
                material = library.get(layer)

                # calculate the impact of the material
                material_impact = self.calculate_impact(material, life_time_overwrites=life_times)

                # add the impact to the construction impact
                impact_result.impacts = impact_result.impacts.add(material_impact.impacts, fill_value=0)

        elif typ == 'window':
            for layer in construction.Layers:

                # Get material object from library
                material = library.get(layer)

                # calculate the impact of the material
                material_impact = self.calculate_impact(material)

                # add the impact to the construction impact
                impact_result.impacts = impact_result.impacts.add(material_impact.impacts, fill_value=0)

        elif typ == 'shading':
            for layer in construction.Layers:
                # Get material object from library
                material = library.get(layer)

                # add shading only, not the window
                if isinstance(material, (firepy.model.building.BlindMaterial, firepy.model.building.ShadeMaterial)):
                    # calculate the impact of the material
                    material_impact = self.calculate_impact(material)

                    # calculate impact based on window m2 and add the impact to the construction impact
                    impact_per_m2 = material_impact.impacts.mul(material.area_per_window_m2())
                    impact_result.impacts = impact_result.impacts.add(impact_per_m2, fill_value=0)

        else:
            raise Exception('Invalid construction impact calculation type: {t}'.format(t=typ))

        self.impact_results[construction.IuId] = impact_result

        return impact_result

    def __building_surface(self, building_surface: firepy.model.building.BuildingSurface,
                           library: ObjectLibrary) -> ImpactResult:
        """
        Impact refers to the total surface including impact of windows
        :param building_surface:
        :param library: ObjectLibrary collection that holds all shading objects
        :return:
        """
        # Initiate impact result
        impact_result = ImpactResult(basis_unit='total')

        # Add impact of all windows to the total
        for window in building_surface.Fenestration:
            window_impact = self.calculate_impact(window, library)
            impact_result.impacts = impact_result.impacts.add(window_impact.impacts, fill_value=0)

        # Get construction of the surface
        construction = library.get(building_surface.Construction)
        construction_impact = self.calculate_impact(construction, library, typ='opaque')

        # Add impact of opaque surface to the total (excluding windows)
        opaque_impact = construction_impact.impacts.mul(building_surface.area_net() * 1.15)
        # TODO multiplication with 1.15 to include materials at junctions (inside reference surface)
        impact_result.impacts = impact_result.impacts.add(opaque_impact, fill_value=0)

        self.impact_results[building_surface.IuId] = impact_result

        return impact_result

        # old kwarg surf_type usage:

        # if surf_type.lower() == building_surface.SurfaceType.lower() or building_surface == 'all':
        #     if sub == 'Fenestration':
        #         return window_lca
        #     elif sub == 'Opaque':
        #         return opaque_lca
        #     else:
        #         return opaque_lca + window_lca

    def __fenestration_surface(self, fenestration_surface: firepy.model.building.FenestrationSurface,
                               library: ObjectLibrary) -> ImpactResult:
        """
        Impact refers to the total surface including impact of shading
        :param fenestration_surface:
        :param library:
        :return:
        """
        # TODO impact refers to window construction which should include frame and glazing
        # In the future separate glazing and frame

        # Initiate impact result
        impact_result = ImpactResult(basis_unit='total')

        # Get shading of the window
        shading = library.get(fenestration_surface.Shading)
        shading_impact = self.calculate_impact(shading, library)

        # Add impact of shading to the total
        shading_total_impact = shading_impact.impacts.mul(fenestration_surface.area())
        impact_result.impacts = impact_result.impacts.add(shading_total_impact, fill_value=0)

        # Get construction of the surface
        construction = library.get(fenestration_surface.Construction)
        construction_impact = self.calculate_impact(construction, library, typ='window')

        # Add impact of window surface to the total
        # TODO from old calculation if frame and glazing defined separately:
        # window_lca = (glazing_lca * window.glazing_area() + frame_lca * window.frame_area()) / window.area()
        window_impact = construction_impact.impacts.mul(fenestration_surface.area())
        impact_result.impacts = impact_result.impacts.add(window_impact, fill_value=0)

        self.impact_results[fenestration_surface.IuId] = impact_result

        return impact_result

    def __non_zone_surface(self, non_zone_surface: firepy.model.building.NonZoneSurface,
                           library: ObjectLibrary) -> ImpactResult:
        """
        Impact refers to the total surface
        :param non_zone_surface:
        :param library:
        :return:
        """
        # Initiate impact result
        impact_result = ImpactResult(basis_unit='total')

        # Get construction of the surface
        construction = library.get(non_zone_surface.Construction)
        construction_impact = self.calculate_impact(construction, library, typ='opaque')

        # Calculate impact of the total area
        surface_impact = construction_impact.impacts.mul(non_zone_surface.area())
        impact_result.impacts = impact_result.impacts.add(surface_impact, fill_value=0)

        self.impact_results[non_zone_surface.IuId] = impact_result

        return impact_result

    def __internal_mass(self, internal_mass: firepy.model.building.InternalMass,
                        library: ObjectLibrary) -> ImpactResult:
        """
        Impact refers to the total
        :param internal_mass:
        :param library:
        :return:
        """
        # Initiate impact result
        impact_result = ImpactResult(basis_unit='total')

        # Get construction of the surface
        construction = library.get(internal_mass.Construction)
        construction_impact = self.calculate_impact(construction, library, typ='opaque')

        # Calculate impact of the total area
        mass_impact = construction_impact.impacts.mul(internal_mass.Area)
        impact_result.impacts = impact_result.impacts.add(mass_impact, fill_value=0)

        self.impact_results[internal_mass.IuId] = impact_result

        return impact_result

    def __zone(self, zone: firepy.model.building.Zone, library: ObjectLibrary) -> ImpactResult:
        """
        Impact refers to total of zone
        :param zone:
        :param library:
        :return:
        """
        # Initiate impact result
        impact_result = ImpactResult(basis_unit='total')

        # add impact of all surfaces
        for surface in zone.BuildingSurfaces:

            # calculate the impact of the surface
            surface_impact = self.calculate_impact(surface, library)

            # add the impact to the zone impact
            impact_result.impacts = impact_result.impacts.add(surface_impact.impacts, fill_value=0)

        # add impact of all internal masses
        for mass in zone.InternalMasses:
            # calculate the impact of the internal mass
            mass_impact = self.calculate_impact(mass, library)

            # add the impact to the zone impact
            impact_result.impacts = impact_result.impacts.add(mass_impact.impacts, fill_value=0)

        self.impact_results[zone.IuId] = impact_result

        return impact_result

    def __building(self, building: firepy.model.building.Building, demands: pd.DataFrame) -> ImpactResult:
        """
        Impact refers to total of building
        :param building:
        :param demands: pandas DataFrame with two columns named 'heating' and 'cooling' containing the
            yearly impact in the sum of the columns in kWh
            e.g. output from energy calculation
        :return:
        """
        # Initiate impact result
        impact_result = ImpactResult(basis_unit='total')

        library = building.Library

        # add impact of all zones
        for zone in building.Zones:

            # calculate the impact of the zone
            zone_impact = self.calculate_impact(zone, library)

            # add the impact to the building impact
            impact_result.impacts = impact_result.impacts.add(zone_impact.impacts, fill_value=0)

        # add impact of all non-zone surfaces
        for surface in building.NonZoneSurfaces:

            # calculate the impact of the surface
            surface_impact = self.calculate_impact(surface, library)

            # add the impact to the building impact
            impact_result.impacts = impact_result.impacts.add(surface_impact.impacts, fill_value=0)

        # add impact of operation
        operation_impact = self.calculate_impact(building.HVAC, demands=demands)

        # add the hvac impact to the building impact
        impact_result.impacts = impact_result.impacts.add(operation_impact.impacts, fill_value=0)

        self.impact_results[building.IuId] = impact_result

        return impact_result

    def __heating(self, heating: firepy.model.hvac.Heating, heating_demand: float) -> impact_results:
        """
        Impact refers to one year
        :param heating:
        :param heating_demand: Net value of heating demand in kWh/year
        :return:
        """
        # Initiate impact result
        impact_result = ImpactResult(basis_unit='year')
        energy_source = heating.energy_source  # Name or DbId
        yearly_demand = heating_demand / heating.efficiency  # gross demand

        # Operation
        operation = self.__operation(
            energy_source=energy_source,
            energy_demand=yearly_demand
        )

        impact_result.impacts.loc[:, 'B6'] = operation

        # Add the result to the collection of results
        self.impact_results[heating.IuId] = impact_result

        return impact_result

    def __cooling(self, cooling: firepy.model.hvac.Cooling, cooling_demand: float) -> impact_results:
        """
        Impact refers to one year
        :param cooling:
        :param cooling_demand: Net value of cooling demand in kWh/year
        :return:
        """
        # Initiate impact result
        impact_result = ImpactResult(basis_unit='year')
        energy_source = cooling.energy_source  # Name or DbId
        yearly_demand = cooling_demand / cooling.efficiency  # gross demand

        # Operation
        operation = self.__operation(
            energy_source=energy_source,
            energy_demand=yearly_demand
        )

        impact_result.impacts.loc[:, 'B6'] = operation

        # Add the result to the collection of results
        self.impact_results[cooling.IuId] = impact_result

        return impact_result

    def __lighting(self, lighting: firepy.model.hvac.Lighting, lighting_energy: float) -> impact_results:
        """
        Impact refers to one year
        :param lighting:
        :param lighting_energy: Total lighting (electric) energy in kWh/year
        :return:
        """
        # Initiate impact result
        impact_result = ImpactResult(basis_unit='year')
        energy_source = lighting.energy_source  # Name or DbId
        yearly_demand = lighting_energy * lighting.inefficiency  # gross demand

        # Operation
        operation = self.__operation(
            energy_source=energy_source,
            energy_demand=yearly_demand
        )

        impact_result.impacts.loc[:, 'B6'] = operation

        # Add the result to the collection of results
        self.impact_results[lighting.IuId] = impact_result

        return impact_result

    def __hvac(self, hvac: firepy.model.hvac.HVAC, demands: pd.DataFrame) -> impact_results:
        """
        Impact refers to total reference period
        :param hvac:
        :param demands: pandas DataFrame with three columns named 'heating', 'cooling' and 'lights'
            containing the yearly impact in the sum of the columns in kWh
            e.g. output from energy calculation
        :return:
        """
        # Initiate impact result
        impact_result = ImpactResult(basis_unit='total')

        # Yearly impact of heating and cooling
        heating_demand = abs(demands.loc[:, 'heating'].sum())
        cooling_demand = abs(demands.loc[:, 'cooling'].sum())
        lighting_energy = abs(demands.loc[:, 'lights'].sum())

        heating_impact = self.calculate_impact(hvac.Heating, heating_demand=heating_demand)
        cooling_impact = self.calculate_impact(hvac.Cooling, cooling_demand=cooling_demand)
        lighting_impact = self.calculate_impact(hvac.Lighting, lighting_energy=lighting_energy)

        # Add impact to the total
        impact_result.impacts = impact_result.impacts.add(heating_impact.impacts.mul(self.rsp), fill_value=0)
        impact_result.impacts = impact_result.impacts.add(cooling_impact.impacts.mul(self.rsp), fill_value=0)
        impact_result.impacts = impact_result.impacts.add(lighting_impact.impacts.mul(self.rsp), fill_value=0)

        # Add the result to the collection of results
        self.impact_results[hvac.IuId] = impact_result

        return impact_result


class UnitOfMeasurementError(Exception):
    pass
