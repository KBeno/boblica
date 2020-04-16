from typing import Union, List, Tuple
import os

import pandas as pd
import numpy as np


class Parameter:

    def __init__(self, name: str, typ: str, value: Union[str, float, int] = None,
                 limits: Tuple[float, float] = (None, None), step: float = None, options: List[str] = None):
        """

        :param name:
        :param typ: 'str' or 'float'
        :param value:
        :param limits:
        :param options:
        """
        self.name = name
        self.value = value
        self.type = typ
        self.limits = limits
        self.step = step
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


class MonteCarloSimulation:

    def __init__(self, client, name: str):
        self.client = client
        self.name = name

    @property
    def parameters(self) -> List[Parameter]:
        return self._parameters

    @parameters.setter
    def parameters(self, param_list: List[Parameter]):
        self._parameters = param_list

    def setup(self, name: str):
        self.parameters = self.client.get_full_params(name=name)
        self.name = name

    def next(self, seed: int = None):
        if seed is None:
            # seed the generator from the computer to enable parallel runs
            np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        else:
            # use the provided seed to enable reproducibility
            np.random.seed(seed)
        params = {p.name: p.random() for p in self.parameters}
        self.client.calculate(name=self.name, parameters=params)


def pareto_non_dominated(df: pd.DataFrame, objectives: list) -> pd.DataFrame:
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

        if len(evaluated) == len(df):
            # we are ready
            break

    return df.loc[non_dominated, :]

