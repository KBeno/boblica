import requests
import logging
import json
from typing import Mapping, Union, MutableMapping, List
from json import JSONDecodeError
from pathlib import Path
from pprint import pformat

import dill
import pandas as pd
from eppy.modeleditor import IDF

from firepy.app.settings import Parameter
from firepy.model.building import Building
from firepy.calculation.lca import LCACalculation
from firepy.calculation.cost import CostCalculation

logger = logging.getLogger(__name__)

class RemoteClient:

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.url = '{host}:{port}'.format(host=self.host, port=self.port)

    def setup(self,
              name: str,
              epw: Path = None,
              idf: IDF = None,
              model: Building = None,
              parameters: MutableMapping[str, Parameter] = None,
              lca_calculation: LCACalculation = None,
              cost_calculation: CostCalculation = None) -> str:
        url = self.url + '/setup'

        logger.info('Setting up calculation: {n} at: {u}'.format(n=name, u=self.url))

        success = {}

        if epw is not None:
            logger.debug('Setting up EPW on server')
            with epw.open('r') as epw_file:
                epw_text = epw_file.read()
            # change windows type newline characters to unix type
            epw_text = epw_text.replace('\r\n', '\n')
            response = requests.post(url=url, params={'name': name, 'type': 'epw'}, data=epw_text)
            logger.debug('Response from server: ' + response.text)
            if response.text != 'OK':
                return response.text
            else:
                success['epw'] = response.text

        if idf is not None:
            logger.debug('Setting up IDF on server')
            idf_text = idf.idfstr()
            response = requests.post(url=url, params={'name': name, 'type': 'idf'}, data=idf_text)
            logger.debug('Response from server: ' + response.text)
            if response.text != 'OK':
                return response.text
            else:
                success['idf'] = response.text

        if model is not None:
            logger.debug('Setting up model on server')
            model_dump = dill.dumps(model)
            response = requests.post(url=url, params={'name': name, 'type': 'model'}, data=model_dump)
            logger.debug('Response from server: ' + response.text)
            if response.text != 'OK':
                return response.text
            else:
                success['model'] = response.text

        if parameters is not None:
            logger.debug('Setting up parameters on server')
            param_dump = dill.dumps(parameters)
            response = requests.post(url=url, params={'name': name, 'type': 'parameters'}, data=param_dump)
            logger.debug('Response from server: ' + response.text)
            if response.text != 'OK':
                return response.text
            else:
                success['parameters'] = response.text

        if lca_calculation is not None:
            logger.debug('Setting up LCA Calculation on server')
            lca_dump = dill.dumps(lca_calculation)
            response = requests.post(url=url, params={'name': name, 'type': 'lca_calculation'}, data=lca_dump)
            logger.debug('Response from server: ' + response.text)
            if response.text != 'OK':
                return response.text
            else:
                success['LCA Calculation'] = response.text

        if cost_calculation is not None:
            logger.debug('Setting up Cost Calculation on server')
            cost_dump = dill.dumps(cost_calculation)
            response = requests.post(url=url, params={'name': name, 'type': 'cost_calculation'}, data=cost_dump)
            logger.debug('Response from server: ' + response.text)
            if response.text != 'OK':
                return response.text
            else:
                success['Cost Calculation'] = response.text

        logger.debug('Initiating result database on server')
        response = requests.post(url=url, params={'name': name, 'type': 'database'})
        logger.debug('Response from server: ' + response.text)
        if response.text != 'OK':
            return response.text
        else:
            success['Database'] = response.text

        return pformat(success)

    def calculate(self, name: str, parameters: Mapping[str, Union[float, int, str]]):
        url = self.url + '/calculate'
        payload = {'name': name}
        payload.update(parameters)
        response = requests.get(url=url, params=payload)
        try:
            return response.json()
        except ValueError:
            return response.text

    def status(self):
        url = self.url + '/status'
        response = requests.get(url=url)
        try:
            return response.json()
        except ValueError:
            return response.text

    def results(self, name: str) -> pd.DataFrame:
        url = self.url + '/results'
        response = requests.get(url=url, params={'name': name})
        try:
            df = pd.read_json(response.json(), orient='split')
            return df
        except JSONDecodeError:
            return response.text

    def reinstate(self, name: str, calc_id: str) -> Mapping:
        url = self.url + '/reinstate'
        response = requests.get(url=url, params={'name': name, 'id': calc_id})
        try:
            return response.json()
        except JSONDecodeError:
            return response.text

    def instate(self, name: str, parameters: Mapping[str, Union[float, int, str]],
                options: Mapping = None) -> Mapping:
        url = self.url + '/instate'
        payload = {'name': name}
        payload.update(parameters)
        if options is not None:
            response = requests.post(url=url, params=payload, data=json.dumps(options))
        else:
            response = requests.get(url=url, params=payload)
        try:
            return response.json()
        except JSONDecodeError:
            return response.text

    def get_model(self, name: str) -> Building:
        url = self.url + '/model'
        response = requests.get(url=url, params={'name': name})
        try:
            model: Building = dill.loads(response.content)
            return model
        except dill.UnpicklingError:
            return response.text

    def get_params(self, name: str) -> Mapping:
        url = self.url + '/parameters'
        response = requests.get(url=url, params={'name': name})
        try:
            return response.json()
        except JSONDecodeError:
            return response.text

    def get_lca(self, name: str) -> LCACalculation:
        url = self.url + '/lca'
        response = requests.get(url=url, params={'name': name})
        try:
            calc: LCACalculation = dill.loads(response.content)
            return calc
        except dill.UnpicklingError:
            return response.text

    def get_cost(self, name: str) -> CostCalculation:
        url = self.url + '/cost'
        response = requests.get(url=url, params={'name': name})
        try:
            calc: CostCalculation = dill.loads(response.content)
            return calc
        except dill.UnpicklingError:
            return response.text

    def get_energy(self, name: str, calc_id: str,
                   variables: List[str] = None,
                   typ: str = 'zone',
                   period: str = 'runperiod') -> pd.DataFrame:
        url = self.url + '/energy'
        if variables is None:
            variables = ['heating', 'cooling']
        response = requests.get(url=url, params={'name': name,
                                                 'id': calc_id,
                                                 'variables': variables,
                                                 'type': typ,
                                                 'period': period})
        try:
            df = pd.read_json(response.json(), orient='split')
            return df
        except JSONDecodeError:
            return response.text

    def get_energy_detailed(self, name: str, calc_id: str,
                            variable: str,
                            typ: str,
                            period: str) -> pd.DataFrame:
        url = self.url + '/energy/detailed'

        response = requests.get(url=url, params={'name': name,
                                                 'id': calc_id,
                                                 'variable': variable,
                                                 'type': typ,
                                                 'period': period})
        try:
            df = pd.read_json(response.json(), orient='split')
            return df
        except JSONDecodeError:
            return response.text