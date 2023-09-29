#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import sys
from functions import *

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.ticker import MaxNLocator
from matplotlib import cm

from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination

os.environ['PYTHONUNBUFFERED'] = '1'

directories = {
    'alexnet'   : '../AlexNet/evaluations/saves/2023-08-20-01-53-10/epoch_19_90.2_91.3.pth/',
    'mobilenet' : '../MobileNet/evaluations/saves/MobileNetV2WithExits/2023-08-20-05-20-25/epoch_19_89.7_90.9.pth',
}

'''
Fixar: Taxa de aceite

2 objetivos:
- Acurácia total do sistema
- Tempo médio de inferência

4 parâmetros -> 
Limiar de normal / ataque na primeira
Limiar de normal / ataque na segunda
'''

class MyProblem(ElementwiseProblem):
    def __init__(self, df, min_acceptance=0.7):
        super().__init__(n_var=4,
                         n_obj=2,
                         n_ieq_constr=1,
                         xl=np.array([ 0.5, 0.5, 0.5, 0.5 ]),
                         xu=np.array([ 1, 1, 1, 1 ]))
            
        self.df = df
        self.accuracy_e1, self.acceptance_e1, self.min_time = get_objectives(self.df, 0, 0, 1, 1)
        self.accuracy_e2, self.acceptance_e2, self.max_time = get_objectives(self.df, 2, 2, 0, 0)
        self.min_acceptance = min_acceptance      

    def _evaluate(self, x, out, *args, **kwargs):
        accuracy, acceptance, time = get_objectives(self.df, *x)
        out["F"] = [ 1 - accuracy, (time - self.min_time) / (self.max_time - self.min_time) ]
        out["G"] = [ self.min_acceptance - acceptance ]

def process(directory='alexnet', glob='2016_01', min_acceptance=0.7):
    directory = directories[directory]
    files = Path(directory).glob(f'*{glob}*')
    dfs = []
    for file in sorted(files):
        dfs.append(pd.read_csv(file))

    df = pd.concat(dfs, ignore_index=True)

    problem = MyProblem(df, min_acceptance=min_acceptance)

    algorithm = NSGA2(
        pop_size=100, # 100
        n_offsprings=80, # 80
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 1000) # 1000

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)

    X = res.X
    F = res.F

    print(f'{directory}')
    print(f'{glob}')

    print(f'Exit1: {problem.accuracy_e1*100:.2f}% - {problem.acceptance_e1*100:.2f}% - {problem.min_time:.2f}us')
    print(f'Exit2: {problem.accuracy_e2*100:.2f}% - {problem.acceptance_e2*100:.2f}% - {problem.max_time:.2f}us')
    print()

    for i in range(len(F)):
        f = F[i]
        x = X[i]
        print(f'{i:02d}: {100 * (1 - f[0]):.2f}% : {problem.min_time + (f[1] * (problem.max_time - problem.min_time)):.2f}us', end='')
        print(f'\t{x[0]:.4f} : {x[1]:.4f} : {x[2]:.4f} : {x[3]:.4f}')

    return X, F, problem.min_time, problem.max_time, problem.accuracy_e1, problem.acceptance_e1, problem.accuracy_e2, problem.acceptance_e2

network = sys.argv[1]
try:
    min_acceptance = float(sys.argv[2])
except e:
    min_acceptance = 0.7

print(f"Processing {network}")

X, F, min_time, max_time, accuracy_e1, acceptance_e1, accuracy_e2, acceptance_e2 = process(network, '2016_0[23]', min_acceptance)

print(f"{min_time}, {max_time}, {accuracy_e1}, {acceptance_e1}, {accuracy_e2}, {acceptance_e2}")

#with open(f'{network}_x_f_{min_acceptance}_2016_23.sav', 'wb') as f:
#    pickle.dump([ X, F, min_time, max_time, accuracy_e1, acceptance_e1, accuracy_e2, acceptance_e2 ], f)
