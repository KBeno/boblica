import os.path
import logging
from pathlib import Path

from flask import Flask, request, jsonify
import redis
import dill
import pandas as pd
import sqlalchemy

from firepy.app.settings import CalculationSetup
from firepy.tools.create import FenestrationCreator

app = Flask(__name__)


# initiate redis client to store and get objects
r = redis.Redis(host='127.0.0.1', port=6379)
# use redis to share between workers in future
CALCULATION_SETUP = None


# @app.before_first_request
# def setup_logging():
#     app.logger.addHandler(logging.StreamHandler())
#     app.logger.setLevel(logging.DEBUG)


@app.route("/setup", methods=["POST"])
def setup():

    # get CalculationSetup from request
    setup_dump = request.get_data()  # CalculationSetup instance from pickle
    calc_setup = dill.loads(setup_dump)

    app.logger.info('Setting up: {n}'.format(n=calc_setup.name))
    # # make folder for the calculation
    # setup_path = BASE_PATH / 'calculations' / calc_setup.name
    # if not setup_path.exists():
    #     setup_path.mkdir(parents=True)
    #
    # # serialize setup and save it to the file (in future to redis)
    # pickle_path = setup_path / 'setup.pickle'
    # app.logger.info('Saving setup file at: {p}'.format(p=pickle_path))
    # with pickle_path.open('wb') as setup_pickle:
    #     dill.dump(calc_setup, setup_pickle)

    # serialize setup and save it to redis
    app.logger.info('Saving setup file to redis')
    r.set(calc_setup.name, setup_dump)

    # TODO log for each calculation separately

    # # Initiate empty file
    # result_csv_path = setup_path + '/results.csv'
    # # make sure we do not overwrite existing files
    # if os.path.isfile(result_csv_path):
    #     i = 0
    #     result_csv_name = result_csv_path.split('.csv')[0]
    #     if '_' in result_csv_name[-3:]:
    #         result_csv_name = result_csv_name.split('_')[0]
    #     while os.path.isfile(result_csv_name + '_{i}.csv'.format(i=i)):
    #         i += 1
    #     result_csv_path = result_csv_name + '_{i}.csv'.format(i=i)
    #
    # app.logger.info('Initiating result file at: {p}'.format(p=result_csv_path))
    # calc_setup.results.to_csv(result_csv_path)

    # additional steps from setup() function of CalculationSetup
    calc_setup.setup(logger=app.logger)

    global CALCULATION_SETUP
    CALCULATION_SETUP = calc_setup

    return 'Setup for {n}: Done'.format(n=calc_setup.name)


@app.route("/calculate", methods=['GET'])
def calculate():

    # get the name of the calculation setup
    name = request.args.get('name')
    if name is None:
        return "Please provide 'name' argument to specify which model to calculate"

    global CALCULATION_SETUP
    if CALCULATION_SETUP is None or CALCULATION_SETUP.name != name:
        # # read setup from file (in future from redis)
        # setup_path = BASE_PATH / 'calculations' / name
        # if not setup_path.exists():
        #     return "No setup found for name: {s}".format(s=name)
        #
        # pickle_path = setup_path / 'setup.pickle'
        #
        # # deserialize setup
        # app.logger.info('Loading setup from file: {p}'.format(p=pickle_path))
        # with pickle_path.open('rb') as setup_pickle:
        #     CALCULATION_SETUP = dill.load(setup_pickle)

        # read setup from redis
        setup_dump = r.get(name)
        if setup_dump is None:
            return "No setup found for name: {s}".format(s=name)

        app.logger.info('Loading setup from {n} redis'.format(n=name))
        calc_setup = dill.loads(setup_dump)
        CALCULATION_SETUP = calc_setup

    # update parameters in the setup
    for name, param in CALCULATION_SETUP.parameters.items():
        # get value from request argument
        value = request.args.get(name)
        if value is None:
            return 'Missing value for parameter: {p}'.format(p=name)

        # convert type of parameter
        try:
            if param.type == 'float':
                value = float(value)
            elif param.type == 'str':
                value = str(value)
            else:
                return 'Parameter type of {p} needs to be one of ["str", "float"], not {pt}'.format(
                    pt=param.type, p=param.name
                )
        except ValueError as e:
            return 'Parameter conversion failed: {e}'.format(e=e)

        if param.limits != (None, None):
            minimum, maximum = param.limits
            if not minimum < value < maximum:
                return 'Parameter value {v} of {p} exceeds its limits: {lim}'.format(
                    v=value, p=param.name, lim=param.limits
                )

        # update parameter value
        param.value = value

    # parameters = {
    #     'fen_ratio_N': float(request.args.get('fen_rat_N')),
    #     'fen_ratio_W': float(request.args.get('fen_rat_W')),
    #     'fen_ratio_S': float(request.args.get('fen_rat_S')),
    #     'fen_ratio_E': float(request.args.get('fen_rat_E')),
    #     'glazing_type_N': request.args.get('glazing_N'),
    #     'glazing_type_W': request.args.get('glazing_W'),
    #     'glazing_type_S': request.args.get('glazing_S'),
    #     'glazing_type_E': request.args.get('glazing_E'),
    #     'frame_type': request.args.get('frame'),
    #     'wall_ins_material': request.args.get('wall_ins_mat'),
    #     'wall_ins_thickness': float(request.args.get('wall_ins_thick')),
    #     'roof_ins_material': request.args.get('roof_ins_mat'),
    #     'roof_ins_thickness': float(request.args.get('roof_ins_thick')),
    #     'floor_ins_thickness': float(request.args.get('floor_ins_thick'))
    # }

    # now that the parameters are updated, we can update the model itself
    CALCULATION_SETUP.update_model(logger=app.logger)

    result = CALCULATION_SETUP.calculate(logger=app.logger)

    # result_csv_path = setup_path + '/results.csv'
    # app.logger.info('Updating results file at: {}'.format(result_csv_path))
    # # TODO this was not okay if working parallel!
    # setup.results.to_csv(result_csv_path)

    return jsonify(result)


@app.route("/status", methods=['GET'])
def status():
    """
    Get status information of the server
    :return: json
    """

    # cal_path = BASE_PATH / 'calculations'
    # if not cal_path.exists():
    #     return 'No setup found'
    # setups = [p.name for p in cal_path.iterdir() if p.is_dir()]

    setups = [k.decode() for k in r.keys()]
    info = {
        'setups': setups
    }

    return jsonify(info)


@app.route("/results", methods=['GET'])
def results():

    # get the name of the calculation setup
    name = request.args.get('name')
    if name is None:
        return "Please provide 'name' argument to specify which results to return"

    global CALCULATION_SETUP
    if CALCULATION_SETUP is None or CALCULATION_SETUP.name != name:
        # # read setup from file (in future from redis)
        # setup_path = BASE_PATH / 'calculations' / name
        # if not setup_path.exists():
        #     return "No setup found for name: {s}".format(s=name)
        #
        # pickle_path = setup_path / 'setup.pickle'
        #
        # # deserialize setup
        # app.logger.info('Loading setup from file: {p}'.format(p=pickle_path))
        # with pickle_path.open('rb') as setup_pickle:
        #     CALCULATION_SETUP = dill.load(setup_pickle)

        # read setup from redis
        setup_dump = r.get(name)
        if setup_dump is None:
            return "No setup found for name: {s}".format(s=name)

        app.logger.info('Loading setup from {n} redis'.format(n=name))
        calc_setup = dill.loads(setup_dump)
        CALCULATION_SETUP = calc_setup

    query = 'SELECT * FROM {tbl}'.format(
        tbl=name,
    )

    if CALCULATION_SETUP.result_db is None:
        CALCULATION_SETUP.result_db = sqlalchemy.create_engine(CALCULATION_SETUP.result_db_url)

    if not CALCULATION_SETUP.result_db.has_table(CALCULATION_SETUP.name):
        return 'No result found for name: {n}'.format(n=CALCULATION_SETUP.name)

    result = pd.read_sql_query(query, CALCULATION_SETUP.result_db)

    return jsonify(result.to_json(orient='split'))

@app.route("/instate", methods=['GET'])
def instate():

    # get the name of the calculation setup
    name = request.args.get('name')
    if name is None:
        return "Please provide 'name' argument to specify which model to calculate"

    global CALCULATION_SETUP
    if CALCULATION_SETUP is None or CALCULATION_SETUP.name != name:

        # read setup from redis
        setup_dump = r.get(name)
        if setup_dump is None:
            return "No setup found for name: {s}".format(s=name)

        app.logger.info('Loading setup from {n} redis'.format(n=name))
        calc_setup = dill.loads(setup_dump)
        CALCULATION_SETUP = calc_setup

    # update parameters in the setup
    for name, param in CALCULATION_SETUP.parameters.items():
        # get value from request argument
        value = request.args.get(name)
        if value is None:
            return 'Missing value for parameter: {p}'.format(p=name)

        # convert type of parameter
        try:
            if param.type == 'float':
                value = float(value)
            elif param.type == 'str':
                value = str(value)
            else:
                return 'Parameter type of {p} needs to be one of ["str", "float"], not {pt}'.format(
                    pt=param.type, p=param.name
                )
        except ValueError as e:
            return 'Parameter conversion failed: {e}'.format(e=e)

        if param.limits != (None, None):
            minimum, maximum = param.limits
            if not minimum < value < maximum:
                return 'Parameter value {v} of {p} exceeds its limits: {lim}'.format(
                    v=value, p=param.name, lim=param.limits
                )

        # update parameter value
        param.value = value

    # now that the parameters are updated, we can update the model itself
    CALCULATION_SETUP.update_model(logger=app.logger)

    CALCULATION_SETUP.calculate(logger=app.logger, save=False)

    setup_dump = dill.dumps(CALCULATION_SETUP)

    return setup_dump

@app.route("/reinstate", methods=['GET'])
def reinstate():

    # get the name of the calculation setup
    name = request.args.get('name')
    if name is None:
        return "Please provide 'name' argument to specify which model to calculate"

    global CALCULATION_SETUP
    if CALCULATION_SETUP is None or CALCULATION_SETUP.name != name:

        # read setup from redis
        setup_dump = r.get(name)
        if setup_dump is None:
            return "No setup found for name: {s}".format(s=name)

        app.logger.info('Loading setup from {n} redis'.format(n=name))
        calc_setup = dill.loads(setup_dump)
        CALCULATION_SETUP = calc_setup

    calc_id = request.args.get('id')
    # update parameters in the setup based on results db
    query = 'SELECT * FROM {tbl}'.format(
        tbl=name
    )

    if CALCULATION_SETUP.result_db is None:
        CALCULATION_SETUP.result_db = sqlalchemy.create_engine(CALCULATION_SETUP.result_db_url)

    if not CALCULATION_SETUP.result_db.has_table(CALCULATION_SETUP.name):
        return 'No result found for name: {n}'.format(n=CALCULATION_SETUP.name)

    result = pd.read_sql_query(query, CALCULATION_SETUP.result_db, index_col='index')
    for name, param in CALCULATION_SETUP.parameters.items():
        # get value from result db
        value = result.loc[calc_id, name]
        if value is None:
            return 'Missing value for parameter: {p}'.format(p=name)

        # convert type of parameter
        try:
            if param.type == 'float':
                value = float(value)
            elif param.type == 'str':
                value = str(value)
            else:
                return 'Parameter type of {p} needs to be one of ["str", "float"], not {pt}'.format(
                    pt=param.type, p=param.name
                )
        except ValueError as e:
            return 'Parameter conversion failed: {e}'.format(e=e)

        if param.limits != (None, None):
            minimum, maximum = param.limits
            if not minimum < value < maximum:
                return 'Parameter value {v} of {p} exceeds its limits: {lim}'.format(
                    v=value, p=param.name, lim=param.limits
                )

        # update parameter value
        param.value = value

    # now that the parameters are updated, we can update the model itself
    CALCULATION_SETUP.update_model(logger=app.logger)

    CALCULATION_SETUP.calculate(logger=app.logger, save=False, sim_id=calc_id)

    setup_dump = dill.dumps(CALCULATION_SETUP)

    return setup_dump


# use only for development:
if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(name)-25s: %(levelname)-8s %(message)s', "%Y-%m-%d %H:%M:%S")
    console.setFormatter(formatter)
    logger.addHandler(console)

    app.run(debug=True, port=9091, host='0.0.0.0')
