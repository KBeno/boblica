import requests
import logging
from typing import Mapping, Union
from json import JSONDecodeError

import dill
import pandas as pd

from firepy.app.settings import CalculationSetup

logger = logging.getLogger(__name__)

class RemoteClient:

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.url = '{host}:{port}'.format(host=self.host, port=self.port)

    def setup(self, setup: CalculationSetup) -> str:
        name = setup.name
        logging.info('Setting up calculation: {n} at: {u}'.format(n=name, u=self.url))
        setup_dump = dill.dumps(setup)
        url = self.url + '/setup'
        response = requests.post(url=url, params={'name': name}, data=setup_dump)
        return response.text

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

    def reinstate(self, name: str, calc_id: str) -> CalculationSetup:
        url = self.url + '/reinstate'
        response = requests.get(url=url, params={'name': name, 'id': calc_id})
        calc_setup = dill.loads(response.content)
        return calc_setup

    def instate(self, name: str, parameters: Mapping[str, Union[float, int, str]]) -> CalculationSetup:
        url = self.url + '/instate'
        payload = {'name': name}
        payload.update(parameters)
        response = requests.get(url=url, params=payload)
        calc_setup = dill.loads(response.content)
        return calc_setup
