import logging
import os
import configparser
import json
import sys
import time
import traceback
import uuid
from importlib import reload
from pathlib import Path
from typing import MutableMapping, Tuple, Union, List, Mapping
from datetime import datetime

import redis
import dill
import pandas as pd
import sqlalchemy
from flask import Flask, request, jsonify
from eppy.modeleditor import IDF

import firepy.setup.functions
from firepy.tools.optimization import Parameter
from firepy.tools.serializer import IdfSerializer
from firepy.calculation.energy import RemoteConnection, EnergyPlusSimulation, SteadyStateCalculation
from firepy.model.building import Building
from firepy.calculation.lca import ImpactResult, LCACalculation
from firepy.calculation.cost import CostResult, CostCalculation

app = Flask('firepy')

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

# get configuration
config = configparser.ConfigParser()
try:
    config_path = Path(os.environ['FIREPY_CONFIG'])
except KeyError:
    config_path = Path('../setup/config.ini')

app.logger.debug('Reading configuration from: {fp}'.format(fp=config_path))
config.read(str(config_path))

# initiate redis client from config to store and get objects
redis_host = config['Redis'].get('host')
redis_port = config['Redis'].getint('port')
R = redis.Redis(host=redis_host, port=redis_port)
# Redis keys: [
#   calculation_name:epw, full epw string
#   calculation_name:idf, full idf string
#   calculation_name:model,
#   calculation_name:parameters,
#   calculation_name:lca_calculation,
#   calculation_name:cost_calculation,
#   calculation_name:energy_calculation,
#   calculation_name:weather_data]
#   calculation_name:last_calculation (timestamp of last calculation)
#   calculation_name:running (integer number of currently running calculations)

# setup energy calculation server from config

ep_host = config['Calculation.Energy'].get('host')
ep_port = config['Calculation.Energy'].getint('port')
server = RemoteConnection(host=ep_host, port=ep_port)
ENERGY_CALCULATION = EnergyPlusSimulation(typ='remote', remote_server=server)

ENERGY_STEADY_STATE = SteadyStateCalculation()

# setup idf_serializer -> from config (no need for shared data)
idd_path_string = config['Firepy'].get('idd_path')
idd_path = Path(idd_path_string)
IDF_PARSER = IdfSerializer(idd_path=idd_path)

# Setup result database from config
db_host = config['Database.Result'].get('host')
db_port = config['Database.Result'].getint('port')
db_user = config['Database.Result'].get('user')
db_pw = config['Database.Result'].get('password')
db_name = config['Database.Result'].get('database')
connection_string = 'postgresql://{user}:{pw}@{host}:{port}'.format(
    user=db_user,
    pw=db_pw,
    host=db_host,
    port=db_port
)
RESULT_DB = sqlalchemy.create_engine(connection_string)


@app.route("/setup", methods=["POST"])
def setup():

    # get calculation name
    setup_name = request.args.get('name')

    R.set('{name}:running'.format(name=setup_name), 0)

    # get type of data
    content_type = request.args.get('type')

    msg = None
    # epw -> Redis
    if content_type == 'epw':
        app.logger.info('Setting up epw for: {n}'.format(n=setup_name))
        content = request.get_data(as_text=True)
        R.set('{name}:epw'.format(name=setup_name), content)
        app.logger.debug('Setting up epw on simulation server')
        ENERGY_CALCULATION.setup_server(name=setup_name, epw=content)

    # idf -> Redis
    if content_type == 'idf':
        app.logger.info('Setting up idf for: {n}'.format(n=setup_name))
        content = request.get_data(as_text=True)
        R.set('{name}:idf'.format(name=setup_name), content)

    # model -> Redis
    if content_type == 'model':
        app.logger.info('Setting up model for: {n}'.format(n=setup_name))
        content = request.get_data()  # serialized Building object
        R.set('{name}:model'.format(name=setup_name), content)

    # parameters -> Redis
    if content_type == 'parameters':
        app.logger.info('Setting up parameters for: {n}'.format(n=setup_name))
        content = request.get_data()  # serialized MutableMapping[str, Parameter] object
        R.set('{name}:parameters'.format(name=setup_name), content)

    # lca calculation -> Redis
    if content_type == 'lca_calculation':
        app.logger.info('Setting up LCA Calculation for: {n}'.format(n=setup_name))
        content = request.get_data()  # serialized LCACalculation object
        R.set('{name}:lca_calculation'.format(name=setup_name), content)

    # cost calculation -> Redis
    if content_type == 'cost_calculation':
        app.logger.info('Setting up Cost Calculation for: {n}'.format(n=setup_name))
        content = request.get_data()  # serialized CostCalculation object
        R.set('{name}:cost_calculation'.format(name=setup_name), content)

    # Check table in database
    if content_type == 'database':
        if RESULT_DB.has_table(setup_name):
            app.logger.info('Result database has table named {n}, results will be appended'.format(n=setup_name))
            msg = 'Results will be appended to existing table {n}'.format(n=setup_name)
        else:
            app.logger.info('New table named {n} will be created in database'.format(n=setup_name))

    # Energy calculation option
    if content_type == 'energy_calculation':
        app.logger.info('Setting up Energy Calculation for: {n}'.format(n=setup_name))
        mode = request.args.get('mode')
        R.set('{name}:energy_calculation'.format(name=setup_name), mode)

    # Energy calculation option
    if content_type == 'weather_data':
        app.logger.info('Setting up Weather Data for: {n}'.format(n=setup_name))
        content = request.get_data()  # serialized pd.DataFrame
        R.set('{name}:weather_data'.format(name=setup_name), content)

    if msg:
        return 'OK - {m}'.format(m=msg)
    else:
        return 'OK'


@app.route("/calculate", methods=['GET'])
def calculate():
    """
    Calculate the impact based on the parameters sent in the args of the request
    Model is updated, calculations are made and results are written in the database
    This is the entry point for external optimization algorithms
    :return: json representation of the calculated objectives
    """

    # get the name of the calculation setup
    name = request.args.get('name')
    if name is None:
        return "Please provide 'name' argument to specify which model to calculate"

    R.incr('{name}:running'.format(name=name))
    parameters, msg = update_params(name=name)
    if msg is not None:
        R.decr('{name}:running'.format(name=name))
        return msg

    reload(firepy.setup.functions)
    from firepy.setup.functions import evaluate

    try:
        # ----------------------- MODEL UPDATE --------------------------------
        model, idf = update_model(name=name, parameters=parameters)

        # ----------------------- CALCULATIONS --------------------------------
        tic = time.perf_counter()
        impact_result, cost_result, energy_result, sim_id = run(name=name, model=model, idf=idf, drop_sim_result=True)
        toc = time.perf_counter()

        # measure execution time
        exec_time = toc - tic

        # ----------------------- EVALUATION --------------------------------
        result = evaluate(impacts=impact_result.impacts, costs=cost_result.costs, energy=energy_result)

    except Exception as err:
        # if anything goes wrong return an invalid result value (e.g. infinity)
        app.logger.info('Calculation failed with error: {e}: {r}'.format(e=sys.exc_info()[0], r=err))
        app.logger.debug('Traceback:')
        if gunicorn_logger.level < 15:
            traceback.print_tb(sys.exc_info()[2])
        result = evaluate()
        sim_id = 'failed'
        exec_time = float('inf')

    # -------------------- WRITE RESULTS TO DATABASE --------------------
    app.logger.info('Saving results to database for: {id}'.format(id=sim_id))

    # collect updated parameters
    data = {p.name: p.value for p in parameters.values()}

    # Create pandas Series from parameters and results
    result_series = pd.Series(data=data, name=sim_id)
    result_series = result_series.append(result)
    result_series['calculation_id'] = sim_id
    result_series['calculation_time'] = exec_time
    result_series['timestamp'] = time.perf_counter()
    result_frame = result_series.to_frame().transpose()

    result_frame.to_sql(name=name, con=RESULT_DB, if_exists='append', index=False)

    timestamp = datetime.now().strftime('%Y.%m.%d %H:%M:%S')
    R.set('{name}:last_calculation'.format(name=name), timestamp)
    R.decr('{name}:running'.format(name=name))
    return jsonify(result.to_dict())


@app.route("/status", methods=['GET'])
def status():
    """
    Get status information of the server
    :return: json
    """
    name = request.args.get('name')

    if name is None:
        setups = [k.decode() for k in R.keys()]
        setups = [s.split(':')[0] for s in setups if ':' in s]
        setups = list(set(setups))

        result_tables = RESULT_DB.table_names()
        running = [R.get('{name}:running'.format(name=name)) for name in setups]
        running = [n.decode() for n in running if n is not None]
        info = {
            'setups': sorted(setups),
            'results': sorted(result_tables),
            'running': running
        }

    else:
        # return info for specific setup
        query = 'SELECT COUNT(*) FROM "{tbl}"'.format(tbl=name)
        if not RESULT_DB.has_table(name):
            result_count = 0
        else:
            result = RESULT_DB.execute(query)
            result_count = result.fetchone()[0]
            result.close()

        last_calc = R.get('{name}:last_calculation'.format(name=name))
        running = R.get('{name}:running'.format(name=name))
        if last_calc is not None:
            last_calc = last_calc.decode()
        else:
            last_calc = 'never'
        if running is not None:
            running = running.decode()
        else:
            running = 'unknown'

        info = {
            'result_count': result_count,
            'last_calc': last_calc,
            'running': running
        }

    return jsonify(info)


@app.route("/results", methods=['GET'])
def results():
    """
    get the results from the database
    :return:
    """
    # get the name of the calculation setup
    name = request.args.get('name')
    if name is None:
        return "Please provide 'name' argument to specify which results to return"

    query = 'SELECT * FROM "{tbl}"'.format(
        tbl=name,
    )

    if not RESULT_DB.has_table(name):
        return 'No result found for name: {n}'.format(n=name)

    result = pd.read_sql_query(query, RESULT_DB)

    return jsonify(result.to_json(orient='split'))

@app.route("/results/upload", methods=['POST'])
def results_upload():
    name = request.args.get('name')
    if name is None:
        return "Please provide 'name' argument to specify which results to return"

    # read pandas DataFrame
    content = request.get_json()
    result_frame = pd.read_json(content, orient='split')

    try:
        result_frame.to_sql(name=name, con=RESULT_DB, if_exists='fail', index=False)
    except ValueError:
        return "Result table with name {n} exists".format(n=name)

    return 'Results uploaded to database'

@app.route("/instate", methods=['GET', 'POST'])
def instate():
    """
    Same as calculate() but the results are not saved to the database
    Use this to update the state of the server to further analyse the model
    :return: id of the simulation and result of the calculation
    """

    # get the name of the calculation setup
    name = request.args.get('name')
    if name is None:
        return "Please provide 'name' argument to specify which model to instate"

    if request.method == 'POST':
        options = request.get_data(as_text=True)
        options = json.loads(options)
    else:
        options = None

    # ----------------------- MODEL UPDATE --------------------------------
    parameters, msg = update_params(name=name)
    if msg is not None:
        return msg

    model, idf = update_model(name=name, parameters=parameters)

    # ----------------------- CALCULATIONS --------------------------------
    tic = time.perf_counter()
    impact_result, cost_result, energy_result, sim_id = run(name=name, model=model, idf=idf, simulation_options=options)
    toc = time.perf_counter()

    # measure execution time
    exec_time = toc - tic

    # ----------------------- EVALUATION --------------------------------
    reload(firepy.setup.functions)
    from firepy.setup.functions import evaluate

    result = evaluate(impacts=impact_result.impacts, costs=cost_result.costs, energy=energy_result)

    data = {
        'result': result.to_dict(),
        'simulation_id': sim_id,
        'calculation_time': exec_time
    }
    return jsonify(data)


@app.route("/reinstate", methods=['GET'])
def reinstate():
    """
    Same as calculate() but the results are not saved to the database and the parameters are
    retrieved from the result database based on the calculation id
    Use this to update the state of the server to further analyse the model
    :return: result of the calculation
    """

    # get the name of the calculation setup
    name = request.args.get('name')
    if name is None:
        return "Please provide 'name' argument to specify which model to reinstate"

    # ----------------------- MODEL UPDATE --------------------------------
    calc_id = request.args.get('id')
    # update parameters in the server based on results db

    parameters, msg = update_params(name=name, calculation_id=calc_id)

    if msg is not None:
        return msg

    model, idf = update_model(name=name, parameters=parameters)

    # ----------------------- CALCULATIONS --------------------------------
    impact_result, cost_result, energy_result, sim_id = run(name=name, model=model, idf=idf, simulation_id=calc_id)

    # ----------------------- EVALUATION --------------------------------
    reload(firepy.setup.functions)
    from firepy.setup.functions import evaluate

    result = evaluate(impacts=impact_result.impacts, costs=cost_result.costs, energy=energy_result)

    return jsonify(result.to_dict())


@app.route("/model", methods=['GET'])
def get_model():
    """
    Return the actual model from the server
    :return: Serialized model
    """
    # get the name of the calculation setup
    name = request.args.get('name')
    if name is None:
        return "Please provide 'name' argument to specify which model to return"

    if not R.exists('{name}:model'.format(name=name)):
        return "No model found for name: {s}".format(s=name)

    model_dump = R.get('{name}:model'.format(name=name))
    return model_dump


@app.route("/parameters", methods=['GET'])
def get_parameters():
    """
    Return the actual parameters from the server
    :return: json
    """
    # get the name of the calculation setup
    name = request.args.get('name')
    if name is None:
        return "Please provide 'name' argument to specify which model to return the parameters of"

    if not R.exists('{name}:parameters'.format(name=name)):
        return "No parameters found for name: {s}".format(s=name)

    params: MutableMapping[str, Parameter] = dill.loads(R.get('{name}:parameters'.format(name=name)))
    param_dict = {name: par.value for name, par in params.items()}
    return jsonify(param_dict)


@app.route("/parameters/full", methods=['GET'])
def get_parameters_full():
    """
    Return the actual parameters from the server
    :return: Serialized List of Parameters
    """
    # get the name of the calculation setup
    name = request.args.get('name')
    if name is None:
        return "Please provide 'name' argument to specify which model to return the parameters of"

    if not R.exists('{name}:parameters'.format(name=name)):
        return "No parameters found for name: {s}".format(s=name)

    param_dump = R.get('{name}:parameters'.format(name=name))
    return param_dump


@app.route("/lca", methods=['GET'])
def get_lca_calculation():
    """
    Return the last lca_calculation from the server
    :return: Serialized LCACalculation
    """
    # get the name of the calculation setup
    name = request.args.get('name')
    if name is None:
        return "Please provide 'name' argument to specify which model to return the calculation of"

    if not R.exists('{name}:lca_calculation'.format(name=name)):
        return "No LCA calculation found for name: {s}".format(s=name)

    calc_dump = R.get('{name}:lca_calculation'.format(name=name))
    return calc_dump


@app.route("/cost", methods=['GET'])
def get_cost_calculation():
    """
    Return the actual model from the server
    :return: Serialized CostCalculation
    """
    # get the name of the calculation setup
    name = request.args.get('name')
    if name is None:
        return "Please provide 'name' argument to specify which model to return the calculation of"

    if not R.exists('{name}:cost_calculation'.format(name=name)):
        return "No Cost calculation found for name: {s}".format(s=name)

    calc_dump = R.get('{name}:cost_calculation'.format(name=name))
    return calc_dump


@app.route("/energy", methods=['GET'])
def get_energy_results():
    """
    Return the energy calculation results for a given simulation id
    :return:
    """
    # get the name of the calculation setup
    name = request.args.get('name')
    if name is None:
        return "Please provide 'name' argument to specify which model to return the calculation of"

    energy_calculation = R.get('{name}:energy_calculation'.format(name=name)).decode()

    if energy_calculation == 'simulation':

        simulation_id = request.args.get('id')
        if simulation_id is None:
            return "Please provide 'id' argument to specify which model to return the energy results of"

        variables = request.args.getlist('variables')
        typ = request.args.get('type')
        period = request.args.get('period')

        app.logger.info('Getting results for simulation with id: {sid}'.format(sid=simulation_id))

        energy_calc_results = ENERGY_CALCULATION.results(variables=variables,
                                                         name=name,
                                                         sim_id=simulation_id,
                                                         typ=typ, period=period)

    elif energy_calculation == 'steady_state':
        model: Building = dill.loads(R.get('{name}:model'.format(name=name)))
        if ENERGY_STEADY_STATE.weather_data is None:
            # setup data in energy calculation
            app.logger.debug('Setting up weather data in steady state calculation.')
            wd_dump = R.get('{name}:weather_data'.format(name=name))
            weather_data = dill.loads(wd_dump)
            ENERGY_STEADY_STATE.weather_data = weather_data
        energy_calc_results = ENERGY_STEADY_STATE.calculate(model)

    else:
        raise Exception('Energy calculation option "{ec}" not implemented.'.format(ec=energy_calculation))

    return jsonify(energy_calc_results.to_json(orient='split'))


@app.route("/energy/detailed", methods=['GET'])
def get_energy_results_detailed():
    """
    Return the energy calculation results for a given simulation id
    :return:
    """
    # get the name of the calculation setup
    name = request.args.get('name')
    if name is None:
        return "Please provide 'name' argument to specify which model to return the calculation of"

    simulation_id = request.args.get('id')
    if simulation_id is None:
        return "Please provide 'id' argument to specify which model to return the energy results of"

    variable = request.args.get('variable')
    typ = request.args.get('type')
    period = request.args.get('period')

    app.logger.info('Getting detailed results for simulation with id: {sid}'.format(sid=simulation_id))

    energy_calc_results = ENERGY_CALCULATION.results_detailed(variable=variable,
                                                              name=name,
                                                              sim_id=simulation_id,
                                                              typ=typ, period=period)

    return jsonify(energy_calc_results.to_json(orient='split'))


@app.route("/idf", methods=['GET'])
def get_idf():
    """
        Return the eppy IDF of the model
        :return:
        """
    # get the name of the calculation setup
    name = request.args.get('name')
    if name is None:
        return "Please provide 'name' argument to specify which model to return the IDF of"

    idf_string = R.get('{name}:idf'.format(name=name))
    return idf_string


@app.route("/cleanup", methods=['GET'])
def cleanup():
    name = request.args.get('name')
    if name is None:
        return "Please provide 'name' argument to specify which setup to cleanup"

    target = request.args.get('target')
    msg = ''
    if target == 'results' or target is None:

        if not RESULT_DB.has_table(name):
            return 'No result found for name: {n}'.format(n=name)

        query = 'DROP TABLE "{n}"'.format(n=name)
        res = RESULT_DB.execute(query)
        res.close()

        app.logger.info('Result table {n} has been cleared'.format(n=name))
        msg += 'Existing table {n} has been cleared; '.format(n=name)

    if target == 'individuals':

        if not RESULT_DB.has_table(name):
            return 'No result found for name: {n}'.format(n=name)

        iid = request.args.get('id')
        if iid is None:
            query = 'DELETE FROM "{n}"'.format(n=name)
        else:
            query = 'DELETE FROM "{n}" WHERE "calculation_id"=\'{pid}\''.format(n=name, pid=iid)

        res = RESULT_DB.execute(query)
        res.close()

        app.logger.info('Individual result with id {id} from table {n} has been cleared'.format(n=name, id=iid))
        msg += 'Existing table {n} has been cleared; '.format(n=name)

    if target == 'simulations' or target is None:
        app.logger.info('Deleting simulation results for {n}; '.format(n=name))
        msg += ENERGY_CALCULATION.server.clean_up(name=name)

    if target == 'setups' or target is None:
        app.logger.info('Deleting setup for {n}'.format(n=name))
        setups = [k.decode() for k in R.keys()]
        sp_to_remove = [s for s in setups if name in s]
        if sp_to_remove:
            n_deleted = R.delete(*sp_to_remove)
            msg += 'Setups items ({i}) has been deleted for {n}; '.format(n=name, i=n_deleted)
        else:
            return 'No setup found for name: {n}'.format(n=name)

    if not msg:
        return 'Unknown target: {t}'.format(t=target)
    else:
        return msg


def update_params(name: str,
                  calculation_id: str = None) -> Tuple[MutableMapping[str, Parameter], Union[str, None]]:
    # update parameters in the setup
    app.logger.debug('Loading parameters for {n} from redis'.format(n=name))

    param_dump = R.get('{name}:parameters'.format(name=name))
    if param_dump is None:
        msg = 'Unable to update model, no setup found for name: {n}'.format(n=name)
        return {}, msg

    parameters: MutableMapping[str, Parameter] = dill.loads(param_dump)
    msg = None

    if calculation_id is not None:
        query = 'SELECT * FROM "{tbl}"'.format(
            tbl=name
        )
        result = pd.read_sql_query(query, RESULT_DB, index_col='calculation_id')
        try:
            db_values = result.loc[calculation_id, :]
        except KeyError:
            msg = 'No previous calculation found for id: {id}'.format(id=calculation_id)
            return parameters, msg
    else:
        db_values = {n: None for n in parameters.keys()}

    for par_name, param in parameters.items():
        if calculation_id is not None:
            # get value from previous calculations
            value = db_values[par_name]
        else:
            # get value from request argument
            value = request.args.get(par_name)

        if value is None:
            msg = 'Missing value for parameter: {p}'.format(p=par_name)
            return parameters, msg

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
                return parameters, msg
        except ValueError as e:
            msg = 'Parameter conversion failed: {e}'.format(e=e)
            return parameters, msg

        if param.limits != (None, None):
            minimum, maximum = param.limits
            if not minimum <= value <= maximum:
                msg = 'Parameter value {v} of {p} exceeds its limits: {lim}'.format(
                    v=value, p=param.name, lim=param.limits
                )
                return parameters, msg

        if param.type == 'str' and param.options is not None:
            if value not in param.options:
                msg = 'Parameter value {v} of {p} is invalid, options are: {o}'.format(
                    v=value, p=param.name, o=param.options
                )
                return parameters, msg

        # update parameter value
        param.value = value

    R.set('{name}:parameters'.format(name=name), dill.dumps(parameters))
    return parameters, msg


def update_model(name: str, parameters: MutableMapping[str, Parameter]) -> Tuple[Building, IDF]:

    app.logger.info('Updating model: {n}'.format(n=name))
    param_values = ['{n}: {v}'.format(n=name, v=p.value) for name, p in parameters.items()]
    app.logger.debug('Parameters: ' + '; '.join(param_values))

    # update the model
    model: Building = dill.loads(R.get('{name}:model'.format(name=name)))

    reload(firepy.setup.functions)
    from firepy.setup.functions import update_model, idf_update_options

    model = update_model(parameters=parameters, model=model)

    R.set('{name}:model'.format(name=name), dill.dumps(model))

    # update idf too along with the model
    app.logger.debug('Updating idf based on model: {n}'.format(n=name))

    idf_string = R.get('{name}:idf'.format(name=name))
    IDF_PARSER.idf = idf_string.decode()
    IDF_PARSER.update_idf(model=model, **idf_update_options)
    R.set('{name}:idf'.format(name=name), IDF_PARSER.idf.idfstr())

    return model, IDF_PARSER.idf


def run(name: str,
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
    :return: impact result and cost result
    """
    energy_calculation = R.get('{name}:energy_calculation'.format(name=name)).decode()

    def run_simulation(options, sim_id=None):
        app.logger.debug('Running simulation')

        frequency = options['output_resolution']
        if frequency is not None:
            ENERGY_CALCULATION.output_frequency = frequency

        ENERGY_CALCULATION.idf = idf

        if options['clear_existing_variables']:
            ENERGY_CALCULATION.clear_outputs()

        zone_outputs: List = options['outputs']['zone']
        app.logger.debug('Setting zone outputs: {}'.format(zone_outputs))
        if zone_outputs:  # not an empty list
            ENERGY_CALCULATION.set_outputs(*zone_outputs, typ='zone')
        else:  # this would never happen since we set the defaults above
            ENERGY_CALCULATION.set_outputs('heating', 'cooling', 'lights', typ='zone')

        surface_outputs: List = options['outputs']['surface']
        app.logger.debug('Setting surface outputs: {}'.format(surface_outputs))
        if surface_outputs:  # not an empty list
            ENERGY_CALCULATION.set_outputs(*surface_outputs, typ='surface')

        if sim_id is not None:
            sim_id = ENERGY_CALCULATION.run(name=name, sim_id=sim_id)
        else:
            sim_id = ENERGY_CALCULATION.run(name=name)

        app.logger.info('Simulation ready, id: {sid}'.format(sid=sim_id))
        return sim_id

    # energy calculation
    if energy_calculation == 'simulation':
        if simulation_options is None:
            reload(firepy.setup.functions)
            from firepy.setup.functions import energy_calculation_options
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
            app.logger.info('Getting previous results for simulation with id: {sid}'.format(sid=simulation_id))
            response = ENERGY_CALCULATION.results(variables=['heating', 'cooling', 'lights'],
                                                  name=name,
                                                  sim_id=simulation_id,
                                                  typ='zone', period='runperiod')

            if 'No result directory' in response:
                app.logger.info('No results found for id: {sid}, rerunning simulation...'.format(sid=simulation_id))
                simulation_id = run_simulation(options=energy_calculation_options, sim_id=simulation_id)

                app.logger.info('Getting results for simulation with id: {sid}'.format(sid=simulation_id))
                response = ENERGY_CALCULATION.results(variables=['heating', 'cooling', 'lights'],
                                                      name=name,
                                                      sim_id=simulation_id,
                                                      typ='zone', period='runperiod')

        else:
            simulation_id = run_simulation(options=energy_calculation_options)

            app.logger.info('Getting results for simulation with id: {sid}'.format(sid=simulation_id))
            response = ENERGY_CALCULATION.results(variables=['heating', 'cooling', 'lights'],
                                                  name=name,
                                                  sim_id=simulation_id,
                                                  typ='zone', period='runperiod')

        if isinstance(response, pd.DataFrame):
            energy_calc_results = response
            if drop_sim_result:
                app.logger.debug('Disposing result of simulation: {sid}'.format(sid=simulation_id))
                ENERGY_CALCULATION.server.drop_result(name=name, sim_id=simulation_id)
        else:
            error_message = 'EnergyPlus error: {t}'.format(t=response)
            app.logger.info(error_message)
            raise Exception(error_message)
        calculation_id = simulation_id

    elif energy_calculation == 'steady_state':
        calculation_id = str(uuid.uuid1())
        app.logger.info('Running steady state energy calculation with id: {id}'.format(id=calculation_id))
        if ENERGY_STEADY_STATE.weather_data is None:
            # setup data in energy calculation
            app.logger.debug('Setting up weather data in steady state calculation.')
            wd_dump = R.get('{name}:weather_data'.format(name=name))
            weather_data = dill.loads(wd_dump)
            ENERGY_STEADY_STATE.weather_data = weather_data
        energy_calc_results = ENERGY_STEADY_STATE.calculate(model)

    else:
        raise Exception('Energy calculation option "{ec}" not implemented.'.format(ec=energy_calculation))

    # TODO make impact and cost calculation optional
    # impact calculation
    app.logger.info('Calculating life cycle impact for: {id}'.format(id=calculation_id))

    lca_calculation: LCACalculation = dill.loads(R.get('{name}:lca_calculation'.format(name=name)))
    lca_calculation.clear_cache()

    lca_result = lca_calculation.calculate_impact(model, demands=energy_calc_results)
    R.set('{name}:lca_calculation'.format(name=name), dill.dumps(lca_calculation))

    # cost calculation
    app.logger.info('Calculating life cycle costs for: {id}'.format(id=calculation_id))

    cost_calculation: CostCalculation = dill.loads(R.get('{name}:cost_calculation'.format(name=name)))
    cost_calculation.clear_cache()

    cost_result = cost_calculation.calculate_cost(model, demands=energy_calc_results)
    R.set('{name}:cost_calculation'.format(name=name), dill.dumps(cost_calculation))

    return lca_result, cost_result, energy_calc_results, calculation_id


# use only for development:
if __name__ == '__main__':
    app.run(debug=True, port=9091, host='0.0.0.0')
