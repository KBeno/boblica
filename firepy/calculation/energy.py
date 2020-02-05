from typing import List, Union
from eppy.modeleditor import IDF
import requests
import json
import logging
import esoreader
import pandas as pd

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

    def run(self, name: str, idf: IDF) -> str:

        url = self.url + '/run'
        logger.debug('Running simulation at: {}'.format(url))
        data = idf.idfstr()
        response = requests.post(url=url, params={'name': name}, data=data)
        return response.text

    def results(self, variables: List[str], name: str, sim_id: str, typ: str, period: str):
        url = self.url + '/results'
        logger.debug('Requesting results from: {}'.format(url))
        payload = {'variables': variables, 'name': name, 'id': sim_id, 'type': typ, 'period': period}
        response = requests.get(url=url, params=payload)
        logger.debug('Response from server: {}'.format(response.text))
        return response.json()

    def results_detailed(self, variable: str, typ: str, period: str):
        # TODO not implemented on server side
        url = self.url + '/results/detailed'
        logger.debug('Requesting detailed results from: {}'.format(url))
        payload = {'variables': variable, 'type': typ, 'period': period}
        response = requests.get(url=url, params=payload)
        return response.json()

    def clean_up(self, name: str) -> str:
        url = self.url + '/cleanup'
        logger.debug('Cleaning up server')
        response = requests.get(url=url, params={'name': name})
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
            'people': 'Zone People Total Heating Energy'
        },
        'surface': {
            'opaque loss': 'Surface Average Face Conduction Heat Transfer Energy',
            'glazing loss': 'Surface Window Heat Loss Energy',
            'glazing gain': 'Surface Window Heat Gain Energy'
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

        'glazing gain': 'J'
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

        self.output_frequency = []
        freq_list = ['runperiod', 'annual', 'monthly', 'daily', 'hourly', 'timestep']
        if output_freq not in freq_list:
            raise Exception('Parameter "output_freq" can be one of: {i}'.format(i=', '.join(freq_list)))
        freq_index = freq_list.index(output_freq)
        self.output_frequency = freq_list[:freq_index+1]

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

    def run_remote(self, name: str, force_setup: bool = False) -> str:
        # check first
        if not self.server.check(name) or force_setup:
            self.setup_server(name=name)
        # than tun
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

        response_json = self.server.results(variables, name, sim_id, typ, period)

        return pd.read_json(response_json, orient='split')

    def results_detailed(self, variable: str, typ: str = 'zone', period: str = 'monthly'):
        if self.typ == 'local':
            return self.results_detailed_local(variable, typ, period)
        elif self.typ == 'remote':
            return self.results_detailed_remote(variable, typ, period)

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

    def results_detailed_remote(self, variable: str, typ: str, period: str):

        response_json = self.server.results_detailed(variable, typ, period)
        return pd.read_json(response_json, orient='split')
