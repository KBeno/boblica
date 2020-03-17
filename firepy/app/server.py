import logging
import os
import configparser
import json
from importlib import reload
from pathlib import Path
from typing import MutableMapping, Tuple, Union, List, Mapping

import redis
import dill
import pandas as pd
import sqlalchemy
from flask import Flask, request, jsonify
from eppy.modeleditor import IDF

import firepy.setup.functions
from firepy.app.settings import Parameter
from firepy.tools.serializer import IdfSerializer
from firepy.calculation.energy import RemoteConnection, EnergyPlusSimulation
from firepy.model.building import Building
from firepy.calculation.lca import ImpactResult, LCACalculation
from firepy.calculation.cost import CostResult, CostCalculation

app = Flask(__name__)

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
#   calculation_name:cost_calculation]

# setup energy calculation server from config

ep_host = config['Calculation.Energy'].get('host')
ep_port = config['Calculation.Energy'].getint('port')
server = RemoteConnection(host=ep_host, port=ep_port)
ENERGY_CALCULATION = EnergyPlusSimulation(typ='remote', remote_server=server)

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

    parameters, msg = update_params(name=name)
    if msg is not None:
        return msg

    reload(firepy.setup.functions)
    from firepy.setup.functions import evaluate

    try:
        # ----------------------- MODEL UPDATE --------------------------------
        model, idf = update_model(name=name, parameters=parameters)

        # ----------------------- CALCULATIONS --------------------------------
        impact_result, cost_result, sim_id = run(name=name, model=model, idf=idf)

        # ----------------------- EVALUATION --------------------------------
        result = evaluate(impacts=impact_result.impacts, costs=cost_result.costs)

    except Exception as e:
        # if anything goes wrong return an invalid result value (e.g. infinity)
        app.logger.info('Calculation failed with error: {e}'.format(e=e))
        result = evaluate()
        sim_id = 'failed'

    # -------------------- WRITE RESULTS TO DATABASE --------------------
    app.logger.info('Saving results to database for: {id}'.format(id=sim_id))

    # collect updated parameters
    data = {p.name: p.value for p in parameters.values()}

    # Create pandas Series from parameters and results
    result_series = pd.Series(data=data, name=sim_id)
    result_series = result_series.append(result)
    result_series['calculation_id'] = sim_id
    result_frame = result_series.to_frame().transpose()

    result_frame.to_sql(name=name, con=RESULT_DB, if_exists='append', index=False)

    return jsonify(result.to_dict())


@app.route("/status", methods=['GET'])
def status():
    """
    Get status information of the server
    :return: json
    """

    setups = [k.decode() for k in R.keys()]
    setups = [s.split(':')[0] for s in setups]
    setups = list(set(setups))

    result_tables = RESULT_DB.table_names()
    info = {
        'setups': setups,
        'results': result_tables
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

    query = 'SELECT * FROM {tbl}'.format(
        tbl=name,
    )

    if not RESULT_DB.has_table(name):
        return 'No result found for name: {n}'.format(n=name)

    result = pd.read_sql_query(query, RESULT_DB)

    return jsonify(result.to_json(orient='split'))


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
    impact_result, cost_result, sim_id = run(name=name, model=model, idf=idf, simulation_options=options)

    # ----------------------- EVALUATION --------------------------------
    reload(firepy.setup.functions)
    from firepy.setup.functions import evaluate

    result = evaluate(impacts=impact_result.impacts, costs=cost_result.costs)

    data = {
        'result': result.to_dict(),
        'simulation_id': sim_id
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
    impact_result, cost_result, sim_id = run(name=name, model=model, simulation_id=calc_id)

    # ----------------------- EVALUATION --------------------------------
    reload(firepy.setup.functions)
    from firepy.setup.functions import evaluate

    result = evaluate(impacts=impact_result.impacts, costs=cost_result.costs)

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


@app.route("/cleanup", methods=['GET'])
def cleanup():
    name = request.args.get('name')
    if name is None:
        return "Please provide 'name' argument to specify which setup tp cleanup"

    target = request.args.get('target')
    msg = ''
    if target == 'results' or target is None:

        if not RESULT_DB.has_table(name):
            return 'No result found for name: {n}'.format(n=name)

        query = 'DROP TABLE {n}'.format(n=name)
        RESULT_DB.execute(query)

        app.logger.info('Result table {n} has been cleared'.format(n=name))
        msg += 'Existing table {n} has been cleared; '.format(n=name)

    if target == 'simulations' or target is None:
        app.logger.info('Deleting simulation results for {n}'.format(n=name))
        msg += ENERGY_CALCULATION.server.clean_up(name=name)

    if not msg:
        return 'Unknown target: {t}'.format(t=target)
    else:
        return msg


def update_params(name: str,
                  calculation_id: str = None) -> Tuple[MutableMapping[str, Parameter], Union[str, None]]:
    # update parameters in the setup
    app.logger.info('Loading parameters for {n} from redis'.format(n=name))

    param_dump = R.get('{name}:parameters'.format(name=name))
    if param_dump is None:
        msg = 'Unable to update model, no setup found for name: {n}'.format(n=name)
        return {}, msg

    parameters: MutableMapping[str, Parameter] = dill.loads(param_dump)
    msg = None

    if calculation_id is not None:
        query = 'SELECT * FROM {tbl}'.format(
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
            if not minimum < value < maximum:
                msg = 'Parameter value {v} of {p} exceeds its limits: {lim}'.format(
                    v=value, p=param.name, lim=param.limits
                )
                return parameters, msg

        # update parameter value
        param.value = value

    R.set('{name}:parameters'.format(name=name), dill.dumps(parameters))
    return parameters, msg


def update_model(name: str, parameters: MutableMapping[str, Parameter]) -> Tuple[Building, IDF]:

    app.logger.info('Updating model: {n}'.format(n=name))

    # update the model
    model: Building = dill.loads(R.get('{name}:model'.format(name=name)))

    reload(firepy.setup.functions)
    from firepy.setup.functions import update_model, update_model, idf_update_options

    model = update_model(parameters=parameters, model=model)

    R.set('{name}:model'.format(name=name), dill.dumps(model))

    # update idf too along with the model
    app.logger.info('Updating idf based on model: {n}'.format(n=name))

    idf_string = R.get('{name}:idf'.format(name=name))
    IDF_PARSER.idf = idf_string.decode()
    IDF_PARSER.update_idf(model=model, **idf_update_options)
    R.set('{name}:idf'.format(name=name), IDF_PARSER.idf.idfstr())

    return model, IDF_PARSER.idf


def run(name: str,
        model: Building,
        idf: IDF = None,
        simulation_id: str = None,
        simulation_options: Mapping = None) -> Tuple[ImpactResult, CostResult, str]:
    """
    Run calculations with the model. Either idf or simulation_id is needed. If simulation_id is given, no
    simulation will run, existing results will be read
    :param name: name of the calculation setup
    :param model: Building model tu run calculation on
    :param idf: IDF representing the same model to use in simulation
    :param simulation_id: if simulation has been made before, the id of the simulation
    :param simulation_options: optional dictionary to pass to customize the simulation
    :return: impact result and cost result
    """

    # energy calculation
    if simulation_options is None:
        reload(firepy.setup.functions)
        from firepy.setup.functions import energy_calculation_options
    else:
        energy_calculation_options = simulation_options

    if simulation_id is None:
        app.logger.info('Running simulation')

        frequency = energy_calculation_options['output_resolution']
        if frequency is not None:
            ENERGY_CALCULATION.output_frequency = frequency

        ENERGY_CALCULATION.idf = idf

        if energy_calculation_options['clear_existing_variables']:
            ENERGY_CALCULATION.clear_outputs()

        zone_outputs: List = energy_calculation_options['outputs']['zone']
        if zone_outputs:  # not an empty list
            ENERGY_CALCULATION.set_outputs(*zone_outputs, typ='zone')
        else:
            ENERGY_CALCULATION.set_outputs('heating', 'cooling', 'lights', typ='zone')

        surface_outputs: List = energy_calculation_options['outputs']['surface']
        if surface_outputs:  # not an empty list
            ENERGY_CALCULATION.set_outputs(*surface_outputs, typ='surface')

        simulation_id = ENERGY_CALCULATION.run(name=name)

        app.logger.info('Simulation ready, id: {sid}'.format(sid=simulation_id))

    app.logger.info('Getting results for simulation with id: {sid}'.format(sid=simulation_id))

    energy_calc_results = ENERGY_CALCULATION.results(variables=['heating', 'cooling', 'lights'],
                                                     name=name,
                                                     sim_id=simulation_id,
                                                     typ='zone', period='runperiod')

    # impact calculation
    app.logger.info('Calculating life cycle impact for: {id}'.format(id=simulation_id))

    lca_calculation: LCACalculation = dill.loads(R.get('{name}:lca_calculation'.format(name=name)))
    lca_calculation.clear_cache()

    lca_result = lca_calculation.calculate_impact(model, demands=energy_calc_results)
    R.set('{name}:lca_calculation'.format(name=name), dill.dumps(lca_calculation))

    # cost calculation
    app.logger.info('Calculating life cycle costs for: {id}'.format(id=simulation_id))

    cost_calculation: CostCalculation = dill.loads(R.get('{name}:cost_calculation'.format(name=name)))
    cost_calculation.clear_cache()

    cost_result = cost_calculation.calculate_cost(model, demands=energy_calc_results)
    R.set('{name}:cost_calculation'.format(name=name), dill.dumps(cost_calculation))

    return lca_result, cost_result, simulation_id


# use only for development:
if __name__ == '__main__':
    app.run(debug=True, port=9091, host='0.0.0.0')
