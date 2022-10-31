from typing import Union, List, Tuple
import os
import logging

import math
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class Parameter:

    def __init__(self, name: str, typ: str, value: Union[str, float, int] = None,
                 limits: Tuple[float, float] = (None, None), step: float = None, options: List[str] = None,
                 precision: int = 4):
        """

        :param name:
        :param typ: 'str' or 'float'
        :param value:
        :param limits:
        :param options:
        :param precision: for float parameters set the number of decimals
        """
        self.name = name
        self.value = value
        self.type = typ
        self.limits = limits
        self.step = step
        self.precision = precision
        if options is not None:
            self.options = options
        elif step is not None:
            lower, upper = limits
            self.options = np.arange(lower, upper + step, step)

    def random(self):
        if self.type == 'str':
            if self.options is not None:
                return np.random.choice(self.options)
            else:
                raise Exception('No options given for parameter: {}'.format(self.name))
        elif self.type == 'float':
            if self.step is None:
                raise Exception('No limits given for parameter {}'.format(self.name))
            else:
                return np.random.choice(self.options)

    def encode(self, value):
        """
        Encodes the given value to a positive integer
        :param value:
        :return:
        """
        if self.type == 'str':
            return self.options.index(value)
        elif self.type == 'float':
            lower, upper = self.limits
            if value < lower:
                logger.info('Value {v} is out of bounds ({l} - {u}) in parameter {p}'.format(
                    v=value, l=lower, u=upper, p=self.name
                ))
                return 0
            elif value > upper:
                logger.info('Value {v} is out of bounds ({l} - {u}) in parameter {p}'.format(
                    v=value, l=lower, u=upper, p=self.name
                ))
                return len(self.options)
            return int((value - lower) // self.step)

    def decode(self, value: float):
        if value == len(self.options):
            value -= 1
        else:
            value = math.floor(value)
        if self.type == 'str':
            return self.options[value]
        elif self.type == 'float':
            lower, upper = self.limits
            return lower + value * self.step

    def limits_encoded(self):
        return 0, len(self.options)

    def normalize(self, encoded_value: int, bounds: Tuple[float, float] = (0, 1)):
        min, max = bounds
        low, high = self.limits_encoded()
        return (encoded_value - low) / (high - 1 - low) * (max - min) + min


class MonteCarloSimulation:

    def __init__(self, client: Union[RemoteClient, LocalClient], name: str):
        self.client = client
        self.name = name

    @property
    def parameters(self) -> List[Parameter]:
        return self._parameters

    # TODO parameter elolszlas tipus megadasa parameterenkent( egyenletes, normal, stb)

    @parameters.setter
    def parameters(self, param_list: List[Parameter]):
        self._parameters = param_list

    def setup(self, name: str):
        self.parameters = self.client.get_full_params(name=name)
        self.name = name

    def next(self, seed: int = None):
        if seed is None:
            # seed the generator from the computer to enable parallel runs
            seed = int.from_bytes(os.urandom(4), byteorder='little')
            # else use the provided seed to enable reproducibility
        np.random.seed(seed)
        params = {p.name: p.random() for p in self.parameters}

        self.client.calculate(name=self.name, parameters=params)

        return seed


def pareto_dominance(df: pd.DataFrame, objectives: list, non_dom: bool = True,
                     dom: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Select pareto non-dominated / dominated solutions from a list of solutions

    :param df: the input DataFrame
    :param objectives: columns names of the objectives to evaluate pareto dominance
    :param non_dom: if True (default) return non-dominated solutions
    :param dom: if True return dominated solutions default is False
    :return: DataFrame of the selected results or both results as (non-dominated, dominated)
    """
    # drop NA values
    df = df.dropna(axis='index', subset=objectives)

    dominated = pd.Index([])
    non_dominated = pd.Index([])
    evaluated = pd.Index([])
    df_objectives = df[objectives]

    while True:

        not_evaluated = df_objectives.drop(index=evaluated)
        candidate = not_evaluated.iloc[0, :]  # first item
        candidate_index = not_evaluated.iloc[[0], :].index

        # compare candidate to all other
        compare = candidate < not_evaluated  # Series < DataFrame

        bigger = compare.all(axis='columns')
        # if any of the not evaluated solutions is dominated by the candidate
        if bigger.any():
            # add index of all rows that are dominated by the candidate to dominated indices
            dominated = dominated.union(not_evaluated[bigger].index)
            evaluated = evaluated.union(not_evaluated[bigger].index)

            # update list and candidate
            not_evaluated = df_objectives.drop(index=evaluated)
            candidate = not_evaluated.iloc[0, :]
            candidate_index = not_evaluated.iloc[[0], :].index

        # check if any dominates the candidate
        compare = candidate > not_evaluated
        smaller = compare.all(axis='columns')
        if smaller.any():
            # candidate is dominated
            dominated = dominated.append(candidate_index)
            evaluated = evaluated.append(candidate_index)
        #                 # restart loop because we don't need to compare this anymore
        #                 break
        else:
            # i is non-dominated
            non_dominated = non_dominated.append(candidate_index)
            evaluated = evaluated.append(candidate_index)

        print('Evaluated: {e} / {t}'.format(e=len(evaluated), t=len(df)), end='\r')

        if len(evaluated) == len(df):
            # we are ready
            break

    if non_dom and not dom:
        return df.loc[non_dominated, :]
    elif dom and not non_dom:
        return df.loc[dominated, :]
    else:
        return df.loc[non_dominated, :], df.loc[dominated, :]


def pareto_rank(df: pd.DataFrame, objectives: list, max_rank: int = 10) -> pd.DataFrame:
    """
    Rank solutions based on pareto dominance

    :param df:
    :param objectives:
    :return: DataFrame with additional column 'pareto_rank'
    """
    df['pareto_rank'] = np.nan
    rank = 0
    evaluating = df[objectives]

    while len(evaluating) > 0:

        non_dominated, dominated = pareto_dominance(evaluating, objectives=objectives, non_dom=True, dom=True)
        df.loc[non_dominated.index, 'pareto_rank'] = rank
        if rank >= max_rank:
            break
        rank += 1
        evaluating = dominated
        print('Rank {r}: {n} - remaining: {d}'.format(r=rank, n=len(non_dominated), d=len(dominated)))

    return df
