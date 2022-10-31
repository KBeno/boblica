import logging
import shutil
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import MutableMapping, Mapping, Union, Callable, Tuple, List

from eppy.modeleditor import  IDF
import pandas as pd

from firepy.calculation.energy import EnergyPlusSimulation, SteadyStateCalculation
from firepy.tools.optimization import Parameter
from firepy.model.building import Building
from firepy.calculation.lca import LCACalculation, ImpactResult
from firepy.calculation.cost import CostCalculation, CostResult
from firepy.tools.serializer import IdfSerializer

logger = logging.getLogger(__name__)


class LocalClient:

    def __init__(self):
        self.name = None
        self.epw = None
        self.weather_data = None
        self.idf = None
        self.model = None
        self.parameters = None
        self.lca_calculation = None
        self.cost_calculation = None
        self.energy_calculation = None
        # extras for local mode
        self._results = None
        self.update_model = None
        self.evaluate = None

        self._idd_path = None
        self.idf_parser = None

        # args of the IdfSerializer.update_idf() method to be used in the update method:
        self.idf_update_options = {
            'update_collections': True,  # True / False
            'zone_method': None,  # 'recreate' / 'update' / None
            'non_zone_surf_method': None,  # 'recreate' / 'update' / None
            'fenestration_method': 'recreate',  # 'recreate' / 'update' / None
            'surface_method': None,  # 'recreate' / 'update' / None
            'internal_mass_method': None  # 'recreate' / 'update' / None
        }

        # args of the EnergyPlusSimulation.set_outputs() method by output type
        # as well as to the EnergyPlusSimulation.
        self.energy_calculation_options = {
            # what output to save during simulation
            'outputs': {
                'zone': [
                    'heating',  # this is the minimum required by the lca calculation
                    'cooling',  # this is the minimum required by the lca calculation
                    # 'infiltration',
                    # 'solar gains',
                    # 'glazing loss',
                    # 'opaque loss',
                    # 'ventilation',
                    'lights',  # this is the minimum required by the lca calculation
                    # 'equipment',
                    # 'people',
                ],
                'surface': [
                    # 'opaque loss',
                    # 'glazing loss',
                    # 'glazing gain',
                ]
            },
            # all lower resolutions will be saved
            'output_resolution': 'runperiod',  # 'runperiod' / 'annual' / 'monthly' / 'daily' / 'hourly' / 'timestep'
            'clear_existing_variables': True
        }

        self.simulation_output_folder = None
        self.simulation = None
        self.steady_state = None

    @property
    def idd_path(self):
        return self._idd_path

    @idd_path.setter
    def idd_path(self, path: str):
        self._idd_path = path
        idd_path = Path(path)
        self.idf_parser = IdfSerializer(idd_path=idd_path)

    @property
    def results(self) -> pd.DataFrame:
        return self._results

    @results.setter
    def results(self, df: pd.DataFrame):
        self._results = df

    @property
    def energy_plus_path(self) -> str:
        return self._energy_plus_path

    @energy_plus_path.setter
    def energy_plus_path(self, path: Union[str, Path]):
        if isinstance(path, str):
            full_path = Path(path).absolute()
        else:
            full_path = path.absolute()
        self._energy_plus_path = full_path

    def setup(self,
              name: str,
              epw: Path = None,
              weather_data: Path = None,
              idf: IDF = None,
              model: Building = None,
              parameters: MutableMapping[str, Parameter] = None,
              lca_calculation: LCACalculation = None,
              cost_calculation: CostCalculation = None,
              energy_calculation: str = None,
              update_model_function: Callable[[MutableMapping[str, Parameter], Building], Building] = None,
              evaluate_function: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.Series] = None,
              init_db: bool = True) -> str:
        """
        Setup the server with the following options. The options can also be set independently.

        :param name: Name of the calculation setup tp create or update
        :param epw: Path to epw file for weather data for simulation
        :param weather_data: Path to csv weather data for steady state energy calculation
        :param idf: eppy IDF model of the building
        :param model: converted firepy model of the building
        :param parameters: dict of firepy Parameters to use for parametric definition
        :param lca_calculation:
        :param cost_calculation:
        :param energy_calculation: 'simulation' or 'steady_state'
        :param init_db: set True (default) to create results database for the setup
        :return: success message
        """

        logger.info('Setting up calculation locally')
        self.name = name

        if epw is not None:
            logger.debug('Setting up EPW')
            self.epw = epw

        if weather_data is not None:
            logger.debug('Setting up weather data')
            weather = pd.read_csv(str(weather_data), header=[0, 1], index_col=[0, 1])
            self.weather_data = weather

        if idf is not None:
            logger.debug('Setting up IDF')
            self.idf = idf
            self.idf.epw = str(self.epw)
            self.idd_path = idf.iddname

        if model is not None:
            logger.debug('Setting up model')
            self.model = model

        if parameters is not None:
            logger.debug('Setting up parameters locally')
            self.parameters = parameters

        if lca_calculation is not None:
            logger.debug('Setting up LCA Calculation locally')
            self.lca_calculation = lca_calculation

        if cost_calculation is not None:
            logger.debug('Setting up Cost Calculation locally')
            self.cost_calculation = cost_calculation

        if init_db:
            logger.debug('Initiating result DataFrame')
            self._results = pd.DataFrame()

        if energy_calculation:
            if energy_calculation not in ['simulation', 'steady_state']:
                return 'Energy calculation type can be one of the following: simulation / steady_state'
            logger.debug('Setting up Energy Calculation type locally')
            self.energy_calculation = energy_calculation

            if energy_calculation == 'simulation':
                self.simulation_output_folder = Path(f"./simulation_output").absolute()
                self.simulation = EnergyPlusSimulation(typ='local',
                                                       output_directory=str(self.simulation_output_folder),
                                                       ep_exe_path=str(self.energy_plus_path))
            elif energy_calculation == 'steady_state':
                self.steady_state = SteadyStateCalculation()
                self.steady_state.weather_data = self.weather_data

        if update_model_function is not None:
            self.update_model = update_model_function

        if evaluate_function is not None:
            self.evaluate = evaluate_function

        return 'OK'

    def calculate(self, parameters: MutableMapping[str, Union[float, int, str]], name=None):
        """
        Calculate the impact based on the parameters
        Model is updated, calculations are made and results are written into a local result DataFrame
        This is the entry point for external optimization algorithms

        :param name: Name of the calculation setup
        :param parameters: Parameters as a dict
        :return: result of the evaluation function as a dict
        """
        # get the name of the calculation setup

        parameters, msg = self._update_params(parameters=parameters)

        try:
            # ----------------------- MODEL UPDATE --------------------------------
            model, idf = self._update_model(parameters=parameters)

            # ----------------------- CALCULATIONS --------------------------------
            tic = time.perf_counter()
            impact_result, cost_result, energy_result, sim_id = self._run(model=model, idf=idf,
                                                                          drop_sim_result=True)
            toc = time.perf_counter()

            # measure execution time
            exec_time = toc - tic

            # ----------------------- EVALUATION --------------------------------
            result = self.evaluate(impacts=impact_result.impacts, costs=cost_result.costs, energy=energy_result)

        except Exception as err:
            # if anything goes wrong return an invalid result value (e.g. infinity)
            logger.info('Calculation failed with error: {e}: {r}'.format(e=sys.exc_info()[0], r=err))
            logger.debug('Traceback: {tr}'.format(tr=traceback.format_exc()))

            result = self.evaluate()
            sim_id = 'failed'
            exec_time = float('inf')

        # -------------------- WRITE RESULTS TO DATABASE --------------------
        logger.info('Saving results to result DataFrame for: {id}'.format(id=sim_id))

        # collect updated parameters
        data = {p.name: p.value for p in parameters.values()}

        # Create pandas Series from parameters and results
        result_series = pd.Series(data=data, name=sim_id)
        result_series = result_series.append(result)
        result_series['calculation_id'] = sim_id
        result_series['calculation_time'] = exec_time
        result_series['timestamp'] = time.perf_counter()
        result_frame = result_series.to_frame().transpose()

        if self.results is None:
            self.results = result_frame
        else:
            self.results = self.results.append(result_frame, ignore_index=True)

        return result.to_dict()

    def _update_params(self,
                       parameters: MutableMapping[str, Union[float, int, str]],
                       calculation_id: str = None) -> Tuple[MutableMapping[str, Parameter], Union[str, None]]:
        """

        Parameters
        ----------
        parameters
        calculation_id

        Returns
        -------

        """

        msg = None

        if calculation_id is not None:
            try:
                db_values = self.results.loc[calculation_id, :]
            except KeyError:
                msg = 'No previous calculation found for id: {id}'.format(id=calculation_id)
                return self.parameters, msg
        else:
            db_values = {n: None for n in parameters.keys()}

        for par_name, param in self.parameters.items():
            if calculation_id is not None:
                # get value from previous calculations
                value = db_values[par_name]
            else:
                # get value from passed dictionary
                value = parameters[par_name]

            if value is None:
                msg = 'Missing value for parameter: {p}'.format(p=par_name)
                return self.parameters, msg

            # convert type of parameter
            try:
                if param.type == 'float':
                    value = float(value)
                elif param.type == 'str':
                    value = str(value)
                else:
                    msg = 'Parameter type of {p} needs to be one of ["str", "float"], not {pt}'.format(
                        pt=param.type, p=param.name
                    )
                    return self.parameters, msg
            except ValueError as e:
                msg = 'Parameter conversion failed: {e}'.format(e=e)
                return self.parameters, msg

            if param.limits != (None, None):
                minimum, maximum = param.limits
                if not minimum <= value <= maximum:
                    msg = 'Parameter value {v} of {p} exceeds its limits: {lim}'.format(
                        v=value, p=param.name, lim=param.limits
                    )
                    return self.parameters, msg

            if param.type == 'str' and param.options is not None:
                if value not in param.options:
                    msg = 'Parameter value {v} of {p} is invalid, options are: {o}'.format(
                        v=value, p=param.name, o=param.options
                    )
                    return self.parameters, msg

            # update parameter value
            param.value = value

        return self.parameters, msg

    def _update_model(self, parameters: MutableMapping[str, Parameter]) -> Tuple[Building, IDF]:

        logger.info('Updating model: {n}'.format(n=self.name))
        param_values = ['{n}: {v}'.format(n=name, v=p.value) for name, p in parameters.items()]
        logger.debug('Parameters: ' + '; '.join(param_values))

        # update the model
        self.model = self.update_model(parameters=parameters, model=self.model)

        # If idf is present, update idf too along with the model
        if self.idf is not None:
            logger.debug('Updating idf based on model: {n}'.format(n=self.name))

            self.idf_parser.idf = self.idf
            self.idf_parser.update_idf(model=self.model, **self.idf_update_options)

            return self.model, self.idf_parser.idf
        
        # Otherwise, return the model only, and None for idf
        else:
            return self.model, None

    def _run(self,
             model: Building,
             idf: IDF = None,
             simulation_id: str = None,
             simulation_options: MutableMapping = None,
             drop_sim_result: bool = False) -> Tuple[ImpactResult, CostResult, pd.DataFrame, str]:
        """
        Run calculations with the model. Either idf or simulation_id is needed. If simulation_id is given, no
        simulation will run, existing results will be read
        :param name: name of the calculation setup
        :param model: Building model tu run calculation on
        :param idf: IDF representing the same model to use in simulation
        :param simulation_id: if simulation has been made before, the id of the simulation
        :param simulation_options: optional dictionary to pass to customize the simulation
        :param drop_sim_result: weather to keep the simulation results on the server or not
        :return: impact result, cost result, energy result and simulation id
        """
        ENERGY_SIMULATION = self.simulation
        ENERGY_STEADY_STATE = self.steady_state
        name = self.name

        def run_simulation(options, sim_id=None):
            logger.info('Running simulation')

            frequency = options['output_resolution']
            if frequency is not None:
                ENERGY_SIMULATION.output_frequency = frequency

            ENERGY_SIMULATION.idf = idf

            if options['clear_existing_variables']:
                ENERGY_SIMULATION.clear_outputs()

            zone_outputs: List = options['outputs']['zone']
            logger.debug('Setting zone outputs: {}'.format(zone_outputs))
            if zone_outputs:  # not an empty list
                ENERGY_SIMULATION.set_outputs(*zone_outputs, typ='zone')
            else:  # this would never happen since we set the defaults above
                ENERGY_SIMULATION.set_outputs('heating', 'cooling', 'lights', typ='zone')

            surface_outputs: List = options['outputs']['surface']
            logger.debug('Setting surface outputs: {}'.format(surface_outputs))
            if surface_outputs:  # not an empty list
                ENERGY_SIMULATION.set_outputs(*surface_outputs, typ='surface')

            if sim_id is not None:
                sim_id = ENERGY_SIMULATION.run(name=name, sim_id=sim_id)
            else:
                sim_id = ENERGY_SIMULATION.run(name=name)

            logger.info('Simulation ready, id: {sid}'.format(sid=sim_id))
            return sim_id

        # energy calculation
        if self.energy_calculation == 'simulation':
            if simulation_options is None:
                energy_calculation_options = self.energy_calculation_options
            else:
                energy_calculation_options = simulation_options

            # Add defaults to the specification
            if 'outputs' not in energy_calculation_options:
                energy_calculation_options['outputs'] = {
                    'zone': ['heating', 'cooling', 'lights'],
                    'surface': []
                }
            else:
                if 'zone' not in energy_calculation_options['outputs']:
                    energy_calculation_options['outputs']['zone'] = ['heating', 'cooling', 'lights']
                else:
                    if 'heating' not in energy_calculation_options['outputs']['zone']:
                        energy_calculation_options['outputs']['zone'].append('heating')
                    if 'cooling' not in energy_calculation_options['outputs']['zone']:
                        energy_calculation_options['outputs']['zone'].append('cooling')
                    if 'lights' not in energy_calculation_options['outputs']['zone']:
                        energy_calculation_options['outputs']['zone'].append('lights')
                if 'surface' not in energy_calculation_options['outputs']:
                    energy_calculation_options['outputs']['surface'] = []

            if 'output_resolution' not in energy_calculation_options:
                energy_calculation_options['output_resolution'] = 'runperiod'

            if 'clear_existing_variables' not in energy_calculation_options:
                energy_calculation_options['clear_existing_variables'] = False

            if simulation_id is not None:
                logger.info('Getting previous results for simulation with id: {sid}'.format(sid=simulation_id))
                response = ENERGY_SIMULATION.results(variables=['heating', 'cooling', 'lights'],
                                                     name=name,
                                                     sim_id=simulation_id,
                                                     typ='zone', period='runperiod')

                if 'No result directory' in response:
                    logger.info('No results found for id: {sid}, rerunning simulation...'.format(sid=simulation_id))
                    simulation_id = run_simulation(options=energy_calculation_options, sim_id=simulation_id)

                    logger.info('Getting results for simulation with id: {sid}'.format(sid=simulation_id))
                    response = ENERGY_SIMULATION.results(variables=['heating', 'cooling', 'lights'],
                                                         name=name,
                                                         sim_id=simulation_id,
                                                         typ='zone', period='runperiod')

            else:
                simulation_id = run_simulation(options=energy_calculation_options)

                logger.info('Getting results for simulation with id: {sid}'.format(sid=simulation_id))
                time.sleep(3)
                response = ENERGY_SIMULATION.results(variables=['heating', 'cooling', 'lights'],
                                                     name=name,
                                                     sim_id=simulation_id,
                                                     typ='zone', period='runperiod')

            if isinstance(response, pd.DataFrame):
                energy_calc_results = response
                if drop_sim_result:
                    logger.debug('Disposing result of simulation: {sid}'.format(sid=simulation_id))
                    ENERGY_SIMULATION.drop_local_result(name=name, sim_id=simulation_id)
            else:
                error_message = 'EnergyPlus error: {t}'.format(t=response)
                logger.info(error_message)
                raise Exception(error_message)
            calculation_id = simulation_id

        elif self.energy_calculation == 'steady_state':
            calculation_id = str(uuid.uuid1())
            logger.info('Running steady state energy calculation with id: {id}'.format(id=calculation_id))

            energy_calc_results = ENERGY_STEADY_STATE.calculate(model)

        else:
            raise Exception('Energy calculation option "{ec}" not implemented.'.format(ec=self.energy_calculation))

        # TODO make impact and cost calculation optional
        # impact calculation
        logger.info('Calculating life cycle impact for: {id}'.format(id=calculation_id))

        self.lca_calculation.clear_cache()

        lca_result = self.lca_calculation.calculate_impact(model, demands=energy_calc_results)

        # cost calculation
        logger.info('Calculating life cycle costs for: {id}'.format(id=calculation_id))

        self.cost_calculation.clear_cache()

        cost_result = self.cost_calculation.calculate_cost(model, demands=energy_calc_results)

        return lca_result, cost_result, energy_calc_results, calculation_id

    def cleanup(self, target: str = None, calc_id: str = None) -> str:
        """
        Cleanup server from stored data if target is not specified both will be deleted
        Prompts for confirmation

        Parameters
        ----------
        target
            'results' / 'individuals' / 'simulations'
        calc_id
            the calculation id if 'individuals' option is selected

        Returns
        -------
            message

        """

        if target == 'results' or target is None:
            logger.warning('Result data will be cleared for setup: {n}'.format(n=self.name))
        if target == 'individuals':
            if calc_id is None:
                logger.warning('All result for setup {n} will be cleared'.format(n=self.name))
            else:
                logger.warning('Individual result for setup {n} with id: {cid} will be cleared'.format(n=self.name,
                                                                                                       cid=calc_id))
        if target == 'simulations' or target is None:
            logger.warning('Simulation results will be deleted for setup: {n}'.format(n=self.name))

        if input('Are you sure? (y/n): ') == 'y':
            if target == 'results' or target is None:
                self._results = None
            if target == 'individuals':
                if calc_id is None:
                    self._results = pd.DataFrame()
                else:
                    self._results = self.results[self.results["calculation_id"] != calc_id]
            if target == 'simulations' or target is None:
                shutil.rmtree(f'{self.simulation_output_folder}_{self.name}')
            return 'OK'
        else:
            logger.warning('Cleanup cancelled')

    def reinstate(self, name: str, calc_id: str) -> Mapping:
        """
        Same as calculate() but the results are not saved to the database and the parameters are
        retrieved from the result database based on the calculation id
        Use this to update the state of the server to further analyse the model

        Parameters
        ----------
        name
        calc_id

        Returns
        -------

        """

        # url = self.url + '/reinstate'
        # response = requests.get(url=url, params={'name': name, 'id': calc_id})
        # try:
        #     return response.json()
        # except JSONDecodeError:
        #     return response.text

    def instate(self, parameters: Mapping[str, Union[float, int, str]],
                options: Mapping = None) -> Mapping:
        """
        Update state of server for the desired parameters and evaluate.
        Calculation results will not be saved to database.

        Parameters
        ----------
        parameters
            parameters as a dict to run calculation for
        options
            simulation options in the following patter
            {
                'outputs': 'all' or {
                    'zone': [
                        'heating' / 'cooling' / 'lights' / 'infiltration' / 'solar gains' / 'glazing loss' /
                        'opaque loss' /'ventilation' / 'equipment' / 'people' ],
                    'surface': [
                        'opaque loss' / 'glazing loss' / 'glazing gain' ]
                },
                'output_resolution': 'runperiod' / 'runperiod' / 'annual' / 'monthly' / 'daily' / 'hourly' / 'timestep',
                'clear_existing_variables': True
            }
        Returns
        -------
            result, simulation id and calculation time in a dict

        """

        # url = self.url + '/instate'
        # payload = {'name': name}
        # payload.update(parameters)
        # if options is not None:
        #     response = requests.post(url=url, params=payload, data=json.dumps(options))
        # else:
        #     response = requests.get(url=url, params=payload)
        # try:
        #     return response.json()
        # except JSONDecodeError:
        #     return response.text

    def get_params(self) -> Mapping:
        return {name: par.value for name, par in self.parameters.items()}

    def get_full_params(self, name=None) -> List[Parameter]:
        return [p for p in self.parameters.values()]

    def get_energy(self, calc_id: str = None,
                   variables: List[str] = None,
                   typ: str = 'zone',
                   period: str = 'runperiod') -> pd.DataFrame:
        """
        Retrieves the energy calculation results from the server. If steady state calculation is used,
        the result is a DataFrame with columns: 'heating' 'cooling', 'lights. If simulation is used,
        the result is specified with the other input parameters

        Parameters
        ----------
        name
        calc_id
            id of a previously run simulation (omitted is steady state calculation)
        variables
            variables to get from the simulation (omitted is steady state calculation)
        typ
            'zone' (default) / 'surface' / 'balance' (omitted is steady state calculation)
        period
            'runperiod' (default) / 'annual' / 'monthly' / 'daily' / 'hourly' / 'timestep'
            (omitted is steady state calculation)
        Returns
        -------
        DataFrame
            with the requested data

        """

        # url = self.url + '/energy'
        # if variables is None:
        #     variables = ['heating', 'cooling', 'lights']
        # response = requests.get(url=url, params={'name': name,
        #                                          'id': calc_id,
        #                                          'variables': variables,
        #                                          'type': typ,
        #                                          'period': period})
        # try:
        #     df = pd.read_json(response.json(), orient='split')
        #     return df
        # except JSONDecodeError:
        #     return response.text

    def get_energy_detailed(self, calc_id: str,
                            variable: str,
                            typ: str,
                            period: str) -> pd.DataFrame:
        """
        Get the detailed energy results as a DataFrame
        Parameters
        ----------
        calc_id
        variable
        typ
        period

        Returns
        -------

        """
        # url = self.url + '/energy/detailed'
        #
        # response = requests.get(url=url, params={'name': name,
        #                                          'id': calc_id,
        #                                          'variable': variable,
        #                                          'type': typ,
        #                                          'period': period})
        # try:
        #     df = pd.read_json(response.json(), orient='split')
        #     return df
        # except JSONDecodeError:
        #     return response.text

    def get_idf(self) -> IDF:
        """
        Return the actual eppy IDF object

        Returns
        -------

        """

