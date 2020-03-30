from typing import Union

import pandas as pd


class Parameter:

    def __init__(self, name: str, typ: str, value: Union[str, float, int] = None, limits: tuple = (None, None)):
        self.name = name
        self.value = value
        self.type = typ
        self.limits = limits


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