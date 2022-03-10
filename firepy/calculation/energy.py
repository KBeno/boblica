from json import JSONDecodeError
from typing import List, Union, Tuple
import requests
import json
import logging
import math
from pathlib import Path

from eppy.modeleditor import IDF
import esoreader
import pandas as pd
import numpy as np
from pandas import Series

from firepy.model import HVAC, Heating, Cooling, NaturalVentilation
from firepy.model.building import Construction, OpaqueMaterial, WindowMaterial, ObjectLibrary, BuildingSurface, \
    Building, Zone, Ref

logger = logging.getLogger(__name__)


class RemoteConnection:

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        if not host.startswith('http'):
            self.host = 'http://' + self.host
        self.url = '{host}:{port}'.format(host=self.host, port=self.port)

    def setup(self, name: str = None, epw: str = None, idd: str = None, variables: dict = None):
        """

        :param name: name of the calculation setup
        :param epw: full epw string
        :param idd: full idd string
        :param variables: dict of variables
        :return:
        """
        url = self.url + '/setup'

        if epw is not None:
            if name is None:
                raise Exception('Please provide a name for the setup')
            logger.debug('Setting up EPW on server')
            requests.post(url=url, params={'name': name, 'type': 'epw'}, data=epw)

        if idd is not None:
            logger.debug('Setting up IDD on server')
            requests.post(url=url, params={'type': 'idd'}, data=idd)

        if variables is not None:
            logger.debug('Setting up variables dict')
            requests.post(url=url, params={'type': 'vars'}, json=variables)

    def check(self, name) -> bool:
        url = self.url + '/check'

        response = requests.get(url=url, params={'name': name})

        if response.text == "OK":
            logger.debug('Server check response: {}'.format(response.text))
            return True
        else:
            logger.debug('Server check response: {}'.format(response.text))
            return False

    def run(self, name: str, idf: IDF, sim_id: str = None) -> str:

        url = self.url + '/run'
        logger.debug('Running simulation at: {}'.format(url))
        data = idf.idfstr()
        params = {'name': name}
        if sim_id is not None:
            params['id'] = sim_id
        response = requests.post(url=url, params=params, data=data)
        return response.text

    def results(self, variables: List[str], name: str, sim_id: str, typ: str, period: str):
        url = self.url + '/results'
        logger.debug('Requesting results from: {}'.format(url))
        payload = {'variables': variables, 'name': name, 'id': sim_id, 'type': typ, 'period': period}
        response = requests.get(url=url, params=payload)
        logger.debug('Response from server: {}'.format(response.text))
        return response

    def results_detailed(self, variable: str, name: str, sim_id: str, typ: str, period: str):
        url = self.url + '/results/detailed'
        logger.debug('Requesting detailed results from: {}'.format(url))
        payload = {'variable': variable, 'name': name, 'id': sim_id, 'type': typ, 'period': period}
        response = requests.get(url=url, params=payload)
        return response.json()

    def clean_up(self, name: str) -> str:
        url = self.url + '/cleanup'
        logger.debug('Cleaning up server')
        response = requests.get(url=url, params={'name': name})
        return response.text

    def drop_result(self, name: str, sim_id: str) -> str:
        url = self.url + '/cleanup/result'
        logger.debug('Deleting result on server for id: {id}'.format(id=sim_id))
        response = requests.get(url=url, params={'name': name, 'id': sim_id})
        return response.text


class EnergyPlusSimulation:

    # TODO separate remote and local class

    var_dict = {
        'zone': {
            'heating': 'Zone Ideal Loads Supply Air Total Heating Energy',
            'cooling': 'Zone Ideal Loads Supply Air Total Cooling Energy',
            'infiltration': 'Zone Infiltration Total Heat Loss Energy',
            'solar gains': 'Zone Windows Total Transmitted Solar Radiation Energy',
            'glazing loss': 'Zone Windows Total Heat Loss Energy',
            'opaque loss': 'Zone Opaque Surface Outside Face Conduction Loss Energy',
            'ventilation': 'Zone Ventilation Sensible Heat Loss Energy',
            'lights': 'Zone Lights Electric Energy',
            'equipment': 'Zone Electric Equipment Electric Energy',
            'other': 'Zone Other Equipment Total Heating Energy',
            'people': 'Zone People Total Heating Energy'
        },
        'surface': {
            'opaque loss': 'Surface Average Face Conduction Heat Transfer Energy',
            'glazing loss': 'Surface Window Heat Loss Energy',
            'glazing gain': 'Surface Window Heat Gain Energy',
            'conduction rate': 'Surface Average Face Conduction Heat Transfer Rate per Area'
        },
        'balance': {
            'internal gain': 'Zone Air Heat Balance Internal Convective Heat Gain Rate',
            'convective': 'Zone Air Heat Balance Surface Convection Rate',
            'interzone air': 'Zone Air Heat Balance Interzone Air Transfer Rate',
            'outdoor air': 'Zone Air Heat Balance Outdoor Air Transfer Rate',
            'system air': 'Zone Air Heat Balance System Air Transfer Rate',
            'system convective': 'Zone Air Heat Balance System Convective Heat Gain Rate',
            'air storage': 'Zone Air Heat Balance Air Energy Storage Rate',
            'deviation': 'Zone Air Heat Balance Deviation Rate'
        }
    }

    units = {
        'heating': 'J',
        'cooling': '-J',
        'infiltration': '-J',
        'solar gains': 'J',
        'glazing loss': '-J',
        'opaque loss': '-J',
        'ventilation': '-J',
        'lights': 'J',
        'equipment': 'J',
        'people': 'J',
        'other': 'J',

        'glazing gain': 'J',
        'conduction rate': 'W/m2',

        'internal gain': 'W',
        'convective': 'W',
        'interzone air': 'W',
        'outdoor ait': 'W',
        'system air': 'W',
        'system convective': 'W',
        'air storage': 'W',
        'deviation': 'W',
    }

    def __init__(self, idf: IDF = None, epw_path: str = None, epw: str = None, output_freq: str = 'monthly',
                 typ: str = 'local', output_directory: str = None, remote_server: RemoteConnection = None):
        """
        A class to run EnergyPlus simulations either locally or remotely on a server

        :param idf: eppy IDF instance to hold model information
        :param epw_path: path to the weather file
        :param epw: full epw string from the weather file
        :param output_freq: output will be saved at this frequency and any lower frequency
                            (e.g. monthly, annual, runperiod)
        :param typ: 'local' or 'remote'; either output_directory need to be set (for local), or server (for remote)
        :param output_directory: a directory path to save EnergyPlus output to
        :param remote_server: a RemoteConnection instance that can connect to the EnergyPlus server
        """

        self.idf = idf
        if self.idf is not None:
            self.idf.epw = epw_path

        self.typ = typ
        if epw is None:
            if epw_path is not None:
                with open(epw_path, 'r') as epw_file:  # 'rb' for binary open?
                    self.epw = epw_file.read()
            else:
                self.epw = None
        else:
            self.epw = epw

        if typ == 'local':
            self.output_directory = output_directory
        if typ == 'remote':
            self.server = remote_server

        self.output_frequency = output_freq

    @property
    def output_frequency(self):
        return self._output_frequency

    @output_frequency.setter
    def output_frequency(self, output_freq: str):
        self._output_frequency = []
        freq_list = ['runperiod', 'annual', 'monthly', 'daily', 'hourly', 'timestep']
        if output_freq not in freq_list:
            raise Exception('Parameter "output_freq" can be one of: {i}'.format(i=', '.join(freq_list)))
        freq_index = freq_list.index(output_freq)
        self._output_frequency = freq_list[:freq_index+1]

    def run(self, **kwargs) -> str:
        if self.idf is None:
            raise Exception('No idf set, unable to run simulation')
        if self.typ == 'local':
            self.run_local()
            return 'Local run complete'
        elif self.typ == 'remote':
            server_response = self.run_remote(**kwargs)
            return server_response

    def run_local(self):
        self.idf.run(output_directory=self.output_directory)

    def run_remote(self, name: str, force_setup: bool = False, sim_id: str = None) -> str:
        # check first
        if not self.server.check(name) or force_setup:
            self.setup_server(name=name)
        # than tun
        if sim_id is not None:
            server_response = self.server.run(name=name, idf=self.idf, sim_id=sim_id)
        else:
            server_response = self.server.run(name=name, idf=self.idf)
        return server_response

    def setup_server(self, name: str, epw: str = None):
        variables = {'var_dict': EnergyPlusSimulation.var_dict, 'units': EnergyPlusSimulation.units}
        if self.epw is None and epw is None:
            raise Exception('No epw is set, please provide epw before setting up the server')
        if epw is not None:
            self.epw = epw
        self.server.setup(name=name, epw=self.epw, variables=variables)
        # optionally we could set the idd

    def clear_outputs(self):
        # clear all output variables
        while len(self.idf.idfobjects['Output:Variable']) > 0:
            self.idf.popidfobject('Output:Variable', 0)

    def set_outputs(self, *args, typ: str = None):
        """
        options:
        ZONE ENERGY (from Honeybee)
            Zone Ideal Loads Supply Air {type} {energy} Cooling Energy
                for type in [Total, Sensible, Latent]
                for energy in [Heating, Cooling]
            Cooling Coil Electric Energy
            Chiller Electric Energy
            Boiler Heating Energy
            Heating Coil Total Heating Energy
            Heating Coil Gas Energy
            Heating Coil Electric Energy
            Humidifier Electric Energy
            Fan Electric Energy
            Zone Ventilation Fan Electric Energy
            Zone Lights Electric Energy
            Zone Electric Equipment Electric Energy
            Earth Tube Fan Electric Energy
            Pump Electric Energy
            Zone VRF Air Terminal Cooling Electric Energy
            Zone VRF Air Terminal Heating Electric Energy
            VRF Heat Pump Cooling Electric Energy
            VRF Heat Pump Heating Electric Energy

        ZONE GAINS AND LOSSES (from Honeybee)
            Zone Windows Total Transmitted Solar Radiation Energy
            Zone Ventilation Sensible Heat Loss Energy
            Zone Ventilation Sensible Heat Gain Energy

            Zone People {type} Heating Energy
            Zone Ideal Loads Zone {type} Heating Energy
            Zone Ideal Loads Zone {type} Cooling Energy
            Zone Infiltration {type} Heat Loss Energy
            Zone Infiltration {type} Heat Gain Energy
                for type in [Total, Sensible, Latent]

        ZONE COMFORT (from Honeybee)
            Zone Operative Temperature
            Zone Mean Air Temperature
            Zone Mean Radiant Temperature
            Zone Air Relative Humidity

        COMFORT MAP (from Honeybee)
            Zone Ventilation Standard Density Volume Flow Rate
            Zone Infiltration Standard Density Volume Flow Rate
            Zone Mechanical Ventilation Standard Density Volume Flow Rate
            Zone Air Heat Balance Internal Convective Heat Gain Rate
            Zone Air Heat Balance Surface Convection Rate
            Zone Air Heat Balance System Air Transfer Rate
            Surface Window System Solar Transmittance

        HVAC METRICS (from Honeybee)
            System Node Standard Density Volume Flow Rate
            System Node Temperature
            System Node Relative Humidity
            Zone Cooling Setpoint Not Met Time
            Zone Heating Setpoint Not Met Time

        SURFACE TEMPERATURE (from Honeybee)
            Surface Outside Face Temperature
            Surface Inside Face Temperature

        SURFACE ENERGY (from Honeybee)
            Surface Average Face Conduction Heat Transfer Energy
            Surface Window Heat Loss Energy
            Surface Window Heat Gain Energy

        GLAZING SOLAR (from Honeybee)
            Surface Window Transmitted Beam Solar Radiation Energy
            Surface Window Transmitted Diffuse Solar Radiation Energy
            Surface Window Transmitted Solar Radiation Energy

        :param args: 'heating' / 'cooling' / etc.
        :param typ: 'zone' / 'surface'
        :return: None
        """
        if 'all' in args:
            if typ is not None:
                for var in EnergyPlusSimulation.var_dict[typ.lower()].values():
                    self.add_variable(var)
            else:
                for typ in EnergyPlusSimulation.var_dict.keys():
                    for var in EnergyPlusSimulation.var_dict[typ].values():
                        self.add_variable(var)
        else:
            if typ is not None:
                for var in args:
                    self.add_variable(EnergyPlusSimulation.var_dict[typ.lower()][var])
            else:
                raise Exception('Please specify output type: {t}'.format(
                    t=' or '.join(EnergyPlusSimulation.var_dict.keys())))

    def add_variable(self, var_name: str):
        variable_names = [ov.Variable_Name for ov in self.idf.idfobjects['Output:Variable']]
        if var_name not in variable_names:
            for output_freq in self.output_frequency:
                self.idf.newidfobject(
                    key='Output:Variable',
                    Key_Value='*',
                    Variable_Name=var_name,
                    Reporting_Frequency=output_freq
                )
        else:
            output_frequencies = [output_var.Reporting_Frequency
                                  for output_var in self.idf.idfobjects['Output:Variable']
                                  if output_var.Variable_Name == var_name]
            for output_freq in self.output_frequency:
                if output_freq not in output_frequencies:
                    self.idf.newidfobject(
                        key='Output:Variable',
                        Key_Value='*',
                        Variable_Name=var_name,
                        Reporting_Frequency=output_freq
                    )

    def results(self, variables: Union[str, List[str]], name: str = None,  sim_id: str = None,
                typ: str = 'zone', period: str = 'monthly'):
        if self.typ == 'local':
            return self.results_local(variables=variables, typ=typ, period=period)
        elif self.typ == 'remote':
            if name is None:
                raise Exception('Please provide "name" to access remote results')
            if sim_id is None:
                raise Exception('Please provide simulation id to access remote results')
            return self.results_remote(variables=variables, name=name, sim_id=sim_id, typ=typ, period=period)

    def results_local(self, variables: Union[str, List[str]], typ: str = 'zone', period: str = 'monthly'):
        if variables == 'all':
            variables = EnergyPlusSimulation.var_dict[typ.lower()].keys()

        elif isinstance(variables, str):
            variables = [variables]

        eso_path = self.output_directory + r'\eplusout.eso'
        eso = esoreader.read_from_path(eso_path)
        res_dfs = []
        for var in variables:
            var_name = EnergyPlusSimulation.var_dict[typ][var]
            df = eso.to_frame(var_name, frequency=period)
            df = df.sum(axis='columns')
            df.name = var
            if EnergyPlusSimulation.units[var] == 'J':  # Convert to kWh
                df /= (3.6*1e6)
            elif EnergyPlusSimulation.units[var] == '-J':
                df /= -(3.6 * 1e6)
            res_dfs.append(df)

        return pd.concat(res_dfs, axis='columns')

    def results_remote(self, variables: Union[str, List[str]], name: str,  sim_id: str,
                       typ: str = 'zone', period: str = 'monthly') -> pd.DataFrame:
        if variables == 'all':
            variables = EnergyPlusSimulation.var_dict[typ.lower()].keys()

        elif isinstance(variables, str):
            variables = [variables]

        response = self.server.results(variables, name, sim_id, typ, period)

        try:
            return pd.read_json(response.json(), orient='split')
        except JSONDecodeError:
            return response.text

    def results_detailed(self, variable: str, name: str = None, sim_id: str = None,
                         typ: str = 'zone', period: str = 'monthly'):
        if self.typ == 'local':
            return self.results_detailed_local(variable, typ, period)
        elif self.typ == 'remote':
            if name is None:
                raise Exception('Please provide "name" to access remote results')
            if sim_id is None:
                raise Exception('Please provide "simulation id" to access remote results')
            return self.results_detailed_remote(variable=variable, name=name, sim_id=sim_id,
                                                typ=typ, period=period)

    def results_detailed_local(self, variable: str, typ: str, period: str):

        eso_path = self.output_directory + r'\eplusout.eso'
        eso = esoreader.read_from_path(eso_path)

        var_name = EnergyPlusSimulation.var_dict[typ][variable]
        df = eso.to_frame(var_name, frequency=period)
        if EnergyPlusSimulation.units[variable] == 'J':  # Convert to kWh
            df /= (3.6 * 1e6)
        elif EnergyPlusSimulation.units[variable] == '-J':
            df /= -(3.6 * 1e6)

        return df

    def results_detailed_remote(self, variable: str, name: str, sim_id: str,
                                typ: str, period: str):

        response_json = self.server.results_detailed(variable=variable, name=name, sim_id=sim_id,
                                                     typ=typ, period=period)
        return pd.read_json(response_json, orient='split')


class SteadyStateCalculation:

    month_lengths = [
        744,
        672,
        744,
        720,
        744,
        720,
        744,
        744,
        720,
        744,
        720,
        744
    ]  # in hours

    year_length = 8760

    def __init__(self, weather_data: Union[pd.DataFrame, Path, str] = None):
        self.weather_data = weather_data

    @property
    def weather_data(self) -> pd.DataFrame:
        return self._weather_data

    @weather_data.setter
    def weather_data(self, data):
        if isinstance(data, pd.DataFrame):
            self._weather_data = data
        elif isinstance(data, (Path, str)):
            if isinstance(data, str):
                data = Path(data)
            self._weather_data = pd.read_csv(str(data), header=[0,1], index_col=[0,1])
        elif data is None:
            self._weather_data = None
        else:
            raise Exception('Only Path, str or pandas DataFrame can be parsed to weather data')

    @staticmethod
    def generate_weather_data(epw: Path = None) -> pd.DataFrame:
        """
        Generate weather data as pandas DataFrame if epw is supplied, weather data will be calculated
        from the epw, if not, a blank DataFrame will be created to be filled by the user
        :param epw: the weather file in .epw format
        :return: pandas DataFrame
        """
        index_labels = [('Monthly', '{m:02n}'.format(m=m)) for m in range(1, 13)] + [('Yearly', 'Yearly')]

        column_labels = [('External Temperature', 'Mean')]
        orientation_list = ['North',  'NorthEast', 'East', 'SouthEast', 'South', 'SouthWest', 'West', 'NorthWest']
        column_labels += [('Total Solar Radiation Energy', orientation) for orientation in orientation_list]

        cols = pd.MultiIndex.from_tuples(column_labels)
        indx = pd.MultiIndex.from_tuples(index_labels)
        weather_data = pd.DataFrame(columns=cols, index=indx)
        if epw is None:
            return weather_data
        else:
            raise Exception('epw data generation is not implemented yet')

    def u_value(self, construction: Ref, library: ObjectLibrary, surface_type="WALL"):
        """
        works with opaque constructions and simple glazing system
        TODO inhomogenity in construction
        TODO effect of screws, fixing elements
        TODO air layers
        """

        surface_heat_resistance = {
            # (R_si, R_se)
            'ROOF': (0.1, 0.04),
            'CEILING': (0.1, 0.04),
            'EXPOSEDFLOOR': (0.17, 0.04),
            'FLOOR': (0.17, 0.04),
            'SLABONGRADE': (0.17, 0.04),
            'WALL': (0.13, 0.04)
            # to be continued...
        }
        try:
            rs_i, rs_e = surface_heat_resistance[surface_type.upper()]
        except KeyError:
            raise Exception('No heat transfer coefficient defined for surface of type: {st}'.format(st=surface_type))

        r_value = rs_i + rs_e

        u_value_win = 0

        construction_obj = library.get(construction)
        for mat in construction_obj.Layers:
            material = library.get(mat)
            if isinstance(material, OpaqueMaterial):
                r_value += material.Thickness / material.Conductivity
            elif isinstance(material, WindowMaterial):
                u_value_win = material.UValue
            else:
                message = "Layer in construction needs to be either OpaqueMaterial or WindowMaterial: "
                message += "{material} - {t}".format(material=material.RefName, t=material.ObjType)
                raise Exception(message)
        if u_value_win != 0:
            u_value = u_value_win
        else:
            u_value = 1 / r_value
        return u_value

    def u_value_floor_to_ground(self, surface: BuildingSurface, wall_thickness: float, library: ObjectLibrary,
                                soil_type='sand') -> float:
        """
        Calculate U value of a floor-to-ground based on ISO 13370 Standard
        :param surface:
        :param wall_thickness:
        :param library:
        :param soil_type:
        :return:
        """
        # TODO underground wall!

        if surface.SurfaceType.lower() not in ['slabongrade', 'floor']:
            raise Exception(
                "U value calculation of Floor to Ground not suitable for {st}!".format(st=surface.SurfaceType))

        soil_conductivity = {
            # W/mK
            'clay': 1.5,  # agyag
            'slit': 1.5,  # iszap
            'sand': 2.0,  # homok
            'gravel': 2.0,  # kavics
            'stone': 3.5
        }

        soil_heat_store_capacity = {
            # J/m3K
            'clay': 3e6,
            'slit': 3e6,
            'sand': 2e6,
            'gravel': 2e6,
            'stone': 2e6
        }

        def heat_resistance(construction):

            r_value = 0
            for mat in construction.Layers:
                material = library.get(mat)
                if isinstance(material, OpaqueMaterial):
                    r_value += material.Thickness / material.Conductivity
                else:
                    raise Exception('Cannot calculate R value for material: {}'.format(material))
            return r_value

        # TODO perimeter of Building, not one surface!!
        # TODO distinguish between heated and non-heated in perimeter!

        # characteristic size:
        B = surface.area() / (0.5 * surface.perimeter())

        # Resistance values
        R_si, R_se = 0.17, 0.04
        R_f = heat_resistance(library.get(surface.Construction))

        Lambda = soil_conductivity[soil_type]

        # equivalent thickness:
        w = wall_thickness
        d_t = w + Lambda * (R_f + R_si)

        if d_t >= B:  # equivalent_thickness >= characteristic_size
            u_value = Lambda / (0.457 * B + d_t)
        else:  # equivalent_thickness < characteristic_size
            u_value = 2 * Lambda / (math.pi * B + d_t) * math.log(math.pi * B / d_t + 1)

        return u_value

    def g_value(self, construction: Construction, library: ObjectLibrary):
        """
        works only with simple glazing system
        """
        g_value_win = 0
        for mat in construction.Layers:
            material = library.get(mat)
            if isinstance(material, WindowMaterial):
                g_value_win = material.gValue
            else:
                raise Exception('Cannot calculate g_value for material: {m}'.format(m=material))
        return g_value_win

    def heat_store_capacity(self, obj: Union[Construction, Zone, Building], library: ObjectLibrary):
        """
        Calculate heat store capacity of Construction / Zone / Building
        For constructions layers from inside are considered until they reach any of the following condition:
            - we reach the first insulation layer
            - we reach 10 cm into the construction
            - we reach the the 1/2 of the construction thickness
        :param obj: Construction, Zone or Building
        :param library: Object library that holds the data of the Materials
        :return: kappa value in [J/m^2*K] for Construction and in [J/K] for Zone
        """

        if isinstance(obj, Construction):
            d = 0  # [m] position in the construction from inside
            kappa = 0
            # layers from inside to outside
            for layer in obj.Layers[::-1]:
                material = library.get(layer)
                if isinstance(material, OpaqueMaterial):
                    if material.Conductivity < 0.1:  # insulation material
                        break
                    elif d + material.Thickness >= min(obj.thickness(library) / 2, 0.1):
                        # we reached the 1/2 of the construction thickness or 10 cm
                        t = min(obj.thickness(library) / 2, 0.1) - d
                        kappa += material.Density * t * material.SpecificHeat
                        d += t
                        break
                    else:
                        kappa += material.Density * material.Thickness * material.SpecificHeat
                        d += material.Thickness
                else:
                    raise Exception('Heat store capacity cannot be calculated for: {m}'.format(m=material))
            return kappa  # [J/m^2*K]

        elif isinstance(obj, Zone):
            capacity = 0
            for surface in obj.BuildingSurfaces:
                if surface.SurfaceType.upper() in ["WALL", "ROOF", "CEILING", "FLOOR", "SLABONGRADE"]:
                    construction = library.get(surface.Construction)
                    capacity += surface.area_net() * self.heat_store_capacity(construction, library)
            for internal in obj.InternalMasses:
                construction = library.get(internal.Construction)
                capacity += 2 * internal.Area * self.heat_store_capacity(construction, library)
                # we take this 2 times, because both sides of the internal structures are exposed to this zone
            return capacity  # [J/K]

        elif isinstance(obj, Building):
            capacity = 0
            for zone in obj.Zones:
                capacity += self.heat_store_capacity(zone, library)
            return capacity  # [J/K]

        else:
            raise Exception('Type of parameter "obj" needs to be one of: Construction, Zone, Building')

    def sum_AU_envelope(self, zone: Zone, library: ObjectLibrary):
        """
        Calculate Summa A*U for the envelope surfaces in [W/K]
        TODO simplified correction factor for heatbridges
        """

        sum__au = 0

        for surface in zone.BuildingSurfaces:

            if surface.OutsideBoundaryCondition.lower() == "outdoors":

                u_value = self.u_value(construction=surface.Construction, library=library,
                                       surface_type=surface.SurfaceType)
                sum__au += u_value * surface.area_net()
                for window in surface.Fenestration:
                    u_value = self.u_value(construction=window.Construction, library=library,
                                           surface_type=surface.SurfaceType)
                    sum__au += u_value * window.area()

        return sum__au  # [W/K]

    def sum_AU_ground(self, zone: Zone, library: ObjectLibrary):
        """
        Calculate Summa A*U for ground contact surfaces in [W/K]
        """

        # get average wall thickness of zone for floor U value calculation
        thickness_list = [library.get(surface.Construction).thickness(library) for surface in zone.BuildingSurfaces if
                          surface.SurfaceType.lower() == 'wall']
        wall_thickness = sum(thickness_list) / len(thickness_list)

        sum__au = 0

        for surface in zone.BuildingSurfaces:

            if surface.OutsideBoundaryCondition.lower() in ["ground", "othersideconditionsmodel"]:
                # OtherSideConditionsModel in case of Ground Domain

                u_value = self.u_value_floor_to_ground(surface, wall_thickness, library)  # define to soil type?
                sum__au += u_value * surface.area()

        return sum__au  # [W/K]

    def sum_lpsi_ground(self, zone: Zone):
        """
        Calculate the heat loss through floor-to-ground perimeter heat bridge in [W/K]
        TODO not implemented yet
        :param zone:
        :return:
        """
        return None

    def sum_lpsi_envelope(self, zone: Zone):
        """
        Calculate the heat loss through envelope heat bridges in [W/K]
        TODO not implemented yet
        :param zone:
        :return:
        """
        return None

    def heat_transmission_direct(self, zone: Zone, library: ObjectLibrary):
        """
        Calculate total direct heat transmission through envelope surfaces (H_tr_D) in [W/K]
        TODO only sum A*U is calculated, sum L*psi and point heat bridges are neglected
        :param zone:
        :param library
        :return:
        """
        h_tr_d = self.sum_AU_envelope(zone=zone, library=library)

        return h_tr_d  # [W/K]

    def heat_transmission_ground(self, zone: Zone, library: ObjectLibrary):
        """
        Calculate total heat transmission through ground contact surfaces (H_tr_T) in [W/K]
        TODO only sum A*U is calculated, sum L*psi and point heat bridges are neglected
        :param zone:
        :param library
        :return:
        """
        h_tr_t = self.sum_AU_ground(zone=zone, library=library)

        return h_tr_t  # [W/K]

    def heat_energy_transmission(self, zone: Zone, hvac: HVAC, library: ObjectLibrary, heating=True) -> pd.Series:
        """
        Calculate the heat transmission (gain or loss) of a zone (Q_tr) in [kWh]
        :param zone:
        :param hvac:
        :param library:
        :return: Heat transmission for each month of the year as pandas Series
        """
        h_tr_d = self.heat_transmission_direct(zone, library)
        h_tr_t = self.heat_transmission_ground(zone, library)
        if heating:
            theta_i = hvac.Heating.set_point  # heating setpoint temperature
        else:  # cooling
            theta_i = hvac.Cooling.set_point  # cooling setpoint temperature
        # monthly mean temperature (pd.Series)
        theta_e_monthly = self.weather_data.loc['Monthly', ('External Temperature', 'Mean')]
        # yearly mean temperature (float)
        theta_e_year = self.weather_data.loc[('Yearly', 'Yearly'), ('External Temperature', 'Mean')]
        # length of months (pd.Series)
        delta_t = pd.Series(data=SteadyStateCalculation.month_lengths, index=[str(i) for i in range(1, 13)])

        q_tr = ((h_tr_d)*(theta_i - theta_e_monthly) + h_tr_t * (theta_i - theta_e_year)) * delta_t / 1000

        return q_tr  # [kWh]

    def heat_natural_ventilation(self, zone: Zone, hvac: HVAC):

        # get natural ventilation ACH
        n_req = hvac.required_ach
        n_fil = hvac.infiltration_ach

        if n_req > n_fil:
            n_nat = max(n_req - n_fil, 0.2)
        else:
            n_nat = 0.2

        # any other constant natural ventilation
        n_nat += hvac.NaturalVentilation.ach

        h_nat_vent = 0.35 * (n_nat + n_fil) * zone.volume()  # [W/K]

        return h_nat_vent

    def heat_natural_ventilation_summer_night(self, zone: Zone, hvac: HVAC):
        # calculate natural ventilation extra ACH for summer nights
        b_night = 1.5  # TODO calculate from weather data
        duration = hvac.NaturalVentilation.summer_night_duration
        n_night = hvac.NaturalVentilation.summer_night_ach
        h_nat_vent_sum_night = 0.35 * b_night * duration / 24 * n_night * zone.volume()  # [W/K]

        return h_nat_vent_sum_night

    def heat_ventilation(self, zone: Zone, hvac: HVAC, heating=True) -> pd.Series:
        """
        H_szell in [W/K]
        :param zone:
        :param hvac:
        :param heating:
        :return:
        """
        h_nat_vent = self.heat_natural_ventilation(zone, hvac)
        h_nat_vent_summer = self.heat_natural_ventilation_summer_night(zone, hvac)

        # monthly mean temperature (pd.Series)
        theta_e_monthly = self.weather_data.loc['Monthly', ('External Temperature', 'Mean')]

        if heating:
            h_vent = pd.Series(data=h_nat_vent, index=[str(i) for i in range(1, 13)])
        else:  # Cooling
            # set night ventilation for cooling season months
            h_nat_vent_summer = pd.Series(data=h_nat_vent_summer, index=[str(i) for i in range(1, 13)])
            h_nat_vent_summer.loc[['1', '2', '3', '4', '10', '11', '12']] = 0  # Cooling season only: May-Sept
            h_vent = h_nat_vent + h_nat_vent_summer

        return h_vent  # [W/K]

    def heat_energy_ventilation(self, zone: Zone, hvac: HVAC, heating=True) -> pd.Series:
        """
        Q_szell in [kWh]
        :param zone:
        :param hvac:
        :param heating:
        :return:
        """
        h_vent = self.heat_ventilation(zone, hvac, heating=heating)

        if heating:
            theta_i = hvac.Heating.set_point  # heating setpoint temperature
        else:  # cooling
            theta_i = hvac.Cooling.set_point  # cooling setpoint temperature

        # monthly mean temperature (pd.Series)
        theta_e_monthly = self.weather_data.loc['Monthly', ('External Temperature', 'Mean')]

        # length of months (pd.Series)
        delta_t = pd.Series(data=SteadyStateCalculation.month_lengths, index=[str(i) for i in range(1, 13)])

        q_vent: pd.Series = h_vent * (theta_i - theta_e_monthly) * delta_t / 1000

        return q_vent  # [kWh]

    def heat_energy_solar(self, zone: Zone, library: ObjectLibrary, heating=True) -> pd.Series:

        q_sd_zone = pd.Series(data=0, index=[str(i) for i in range(1, 13)])

        for surface in zone.BuildingSurfaces:
            for window in surface.Fenestration:
                construction = library.get(window.Construction)

                # g value of glazing system
                f_g = 0.9  # solar incident correction factor
                g_n = self.g_value(construction, library)  # g factor for perpendicular radiation
                g_w = g_n * f_g

                # area of glazing
                a_w = window.glazing_area(mode='FrameWidth', frame_width=0.1)  # TODO include frame width in model

                # g value of shading
                if window.Shading is not None:
                    shading = library.get(window.Shading)
                    g_sh = shading.ShadingFactor
                    if not shading.IsScheduled:
                        g_sh = 1
                    if heating:
                        g_sh = 1
                else:
                    g_sh = 1

                # TODO f_s shading factor of external shading surfaces

                # total solar radiation energy (pd.Series)
                orientation = window.orientation()
                g_s = self.weather_data.loc['Monthly', ('Total Solar Radiation Energy', orientation)]

                q_sd = g_w * a_w * g_sh * g_s
                q_sd_zone += q_sd

        return q_sd_zone  # [kWh]

    def heat_energy_internal(self, zone: Zone, hvac: HVAC) -> pd.Series:

        a_n = zone.heated_area()
        q_b = hvac.internal_gain

        # length of months (pd.Series)
        delta_t = pd.Series(data=SteadyStateCalculation.month_lengths, index=[str(i) for i in range(1, 13)])

        q_b_zone = a_n * q_b * delta_t / 1000
        return q_b_zone  # kWh

    def gamma_tao_loss_gain(self, zone: Zone, hvac: HVAC, library: ObjectLibrary,
                            heating: bool) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Helper function to calculate gamma (loss/gain ratio) and tao (time factor)
        :param zone:
        :param hvac:
        :param library:
        :return: gamma, tao, loss, gain as pd.Series
        """
        # Total loss in kWh
        q_loss = self.heat_energy_transmission(zone, hvac, library, heating=heating)
        q_loss += self.heat_energy_ventilation(zone, hvac, heating=heating)

        # Total gain in kWh
        q_gain = self.heat_energy_solar(zone, library, heating=heating)
        q_gain += self.heat_energy_internal(zone, hvac)

        # loss/gain ratio
        gamma: pd.Series = q_gain / q_loss

        # zone heat store capacity in [kJ/K]
        c_m_eff = self.heat_store_capacity(zone, library) / 1000

        h_tr_d = self.heat_transmission_direct(zone, library)
        h_tr_t = self.heat_transmission_ground(zone, library)
        h_vent = self.heat_ventilation(zone, hvac, heating=heating)

        # time factor [h] as pd.Series
        tao: pd.Series = c_m_eff / 3.6 / (h_tr_d + h_tr_t + h_vent)

        return gamma, tao, q_loss, q_gain

    def heating_demand(self, zone: Zone, hvac: HVAC, library: ObjectLibrary) -> pd.Series:
        """
        Monthly net heating energy in [kWh]
        :param zone:
        :param hvac:
        :param library:
        :return: pd.Series with the monthly demand
        """
        gamma_h, tao_h, q_loss, q_gain = self.gamma_tao_loss_gain(zone, hvac, library, heating=True)

        # numeric factors for monthly calculation in case of heating
        a_h_0 = 1
        tao_h_0 = 15
        # TODO for seasonal calculation:
        # a_h_0 = 0.8
        # tao_h_0 = 30

        a_h = a_h_0 + tao_h / tao_h_0  # pd.Series

        df = pd.concat([a_h, gamma_h, q_loss, q_gain], keys=['a', 'gamma', 'loss', 'gain'],  axis='columns')

        # utilization factor
        def utilization_factor(gamma: float, a: float, gain: float):
            if gamma > 0 and gamma != 1:
                theta = (1 - gamma ** a) / (1 - gamma ** (a + 1))
            elif gamma == 1:
                theta = a / (a + 1)
            else:  # gamma <= 0
                if gain > 0:
                    theta = 1 / gamma
                else:  # q_gain <=0
                    theta = 1
            return theta

        # calculate for series:
        df['theta'] = df.apply(lambda row: utilization_factor(row['gamma'], row['a'], row['gain']), axis='columns')

        # net heating demand
        def net_demand(gamma: float, theta: float, loss: float, gain: float):
            if gamma <= 0 and gain > 0:
                demand = 0
            elif gamma > 2:
                demand = 0
            else:
                demand = loss - theta * gain

            if demand < 0:
                demand = 0

            return demand

        q_h_net = df.apply(lambda row: net_demand(row['gamma'], row['theta'], row['loss'], row['gain']), axis='columns')
        q_h_net.name = zone.Name

        return q_h_net

    def cooling_demand(self, zone: Zone, hvac: HVAC, library: ObjectLibrary):
        """
        Monthly net cooling energy in [kWh]
        :param zone:
        :param hvac:
        :param library:
        :return: pd.Series with the monthly demand
        """
        gamma_c, tao_c, q_loss, q_gain = self.gamma_tao_loss_gain(zone, hvac, library, heating=False)

        # numeric factors for monthly calculation in case of cooling
        a_c_0 = 1
        tao_c_0 = 15
        # TODO for seasonal calculation:
        # a_h_0 = 0.8
        # tao_h_0 = 30

        a_c = a_c_0 + tao_c / tao_c_0  # pd.Series

        df = pd.concat([a_c, gamma_c, q_loss, q_gain], keys=['a', 'gamma', 'loss', 'gain'], axis='columns')

        # utilization factor
        def utilization_factor(gamma: float, a: float):
            if gamma > 0 and gamma != 1:
                theta = (1 - gamma ** (-a)) / (1 - gamma ** (-(a + 1)))
            elif gamma == 1:
                theta = a / (a + 1)
            else:  # gamma <= 0
                theta = 1
            return theta

        # calculate for series:
        df['theta'] = df.apply(lambda row: utilization_factor(row['gamma'], row['a']), axis='columns')

        # net heating demand
        def net_demand(gamma: float, theta: float, loss: float, gain: float):
            if 1 / gamma > 2:
                demand = 0
            else:
                demand = gain - theta * loss

            if demand < 0:
                demand = 0

            return demand

        q_c_net = df.apply(lambda row: net_demand(row['gamma'], row['theta'], row['loss'], row['gain']), axis='columns')
        q_c_net.name = zone.Name

        return q_c_net

    def lighting_demand(self, zone: Zone, hvac: HVAC):
        """
        Lighting demand with simplified calculation
        :param zone:
        :param hvac:
        :return: Total lighting demand of zone in [kWh/year]
        """

        # power density in W/m2
        p = hvac.Lighting.power_density
        f_fe = 1  # non-dimmable lights
        f_szab = 1
        t_nappal = 3000  # [h]
        t_ejjel = 2000  # [h]

        envelope_area = 0
        glazing_area = 0
        for surface in zone.BuildingSurfaces:
            if surface.OutsideBoundaryCondition.lower() == "outdoors" and surface.SurfaceType.lower() == 'wall':
                envelope_area += surface.area()
                for window in surface.Fenestration:
                    glazing_area += window.glazing_area(mode='FrameWidth', frame_width=0.1)
                    # TODO include frame width in model

        if envelope_area == 0:
            glazing_ratio = 0
        else:
            glazing_ratio = glazing_area / envelope_area

        if glazing_ratio > 0.8:
            f_nappal = 0.56
        elif 0.8 > glazing_ratio > 0.4:
            f_nappal = 0.7
        else:
            f_nappal = 0.83

        w_vil = f_fe * p * f_szab * (t_nappal * f_nappal + t_ejjel) * zone.heated_area() / 1000

        return w_vil

    def calculate(self, building: Building) -> pd.DataFrame:
        """
        Calculate heating cooling and lighting demand of building for each month
        :param building:
        :return: demands as pandas DataFrame
        """

        heating = []
        cooling = []
        lights = []

        for zone in building.Zones:
            heating.append(self.heating_demand(zone=zone, hvac=building.HVAC, library=building.Library))
            cooling.append(self.cooling_demand(zone=zone, hvac=building.HVAC, library=building.Library))
            lights.append(self.lighting_demand(zone=zone, hvac=building.HVAC))

        heating_demand = pd.concat(heating, axis='columns').sum(axis='columns')
        cooling_demand = pd.concat(cooling, axis='columns').sum(axis='columns')
        lights_demand = sum(lights)  # float (yearly demand)

        result = pd.concat([heating_demand, cooling_demand], axis='columns',
                           keys=['heating', 'cooling'])
        result['lights'] = lights_demand / 12  # (monthly)
        return result