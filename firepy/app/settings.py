import os.path
import logging
from typing import Union, List, MutableMapping, Callable
from types import MethodType
# from __future__ import annotations

import pandas as pd
from eppy.modeleditor import rename
import sqlalchemy

from firepy.tools.serializer import IdfSerializer
from firepy.tools.create import FenestrationCreator
from firepy.calculation.energy import EnergyPlusSimulation, RemoteConnection
from firepy.calculation.lca import LCACalculation
from firepy.calculation.cost import CostCalculation
from firepy.model.hvac import *
from firepy.model.building import Building


logger = logging.getLogger(__name__)


class Parameter:

    def __init__(self, name: str, typ: str, value: Union[str, float, int] = None, limits: tuple = (None, None)):
        self.name = name
        self.value = value
        self.type = typ
        self.limits = limits

# DEPRECATED
class CalculationSetup:
    """
    Deprecated Class
    """

    def __init__(self, name, idf_parser: IdfSerializer, model: Building, parameters: MutableMapping[str, Parameter],
                 energy_calculation: EnergyPlusSimulation, lca_calculation: LCACalculation,
                 cost_calculation: CostCalculation, objectives: List[str],
                 setup_function: Callable[['CalculationSetup'], 'CalculationSetup'],
                 update_function: Callable[['CalculationSetup'], None],
                 calculate_function: Callable[['CalculationSetup'], dict],
                 result_db_url: str):
        """

        :param name:
        :param idf_parser:
        :param model:
        :param parameters:
        :param energy_calculation:
        :param lca_calculation:
        :param cost_calculation:
        :param objectives: The names of the objectives as a list of strings
        :param update_function: Update function that is used to update the model
            It takes a CalculationSetup instance and a list of Parameter instances as input and updates
            the model based on the parameters. It does not return anything
        :param calculate_function: The function that does the effective calculation.
            It should return a dict containing the calculated values of the objectives as {obj_name: obj_value, ...}
        :param setup_function: The function to be called during setup on the server
        """
        self.name = name
        self.idf_parser = idf_parser
        self.model = model
        self.parameters = parameters
        self.objectives = objectives
        self.energy_calculation = energy_calculation
        self.lca_calculation = lca_calculation
        self.cost_calculation = cost_calculation
        self.update_model = MethodType(update_function, self)
        self.calculate = MethodType(calculate_function, self)
        self.setup = MethodType(setup_function, self)
        self.result_db_url = result_db_url
        self.result_db = None

        # # Initiate result csv file
        # columns = [p.name for p in self.parameters.values()] + self.objectives
        #
        # self.results = pd.DataFrame(columns=columns)
        # self.results.index.name = 'result_id'

