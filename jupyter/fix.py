#!/home/ubuntu/ppgia/bin/python3

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import sys

network = 'mobilenet'

with open(f'{network}_x_f_2016_23.sav', 'rb') as f:
    X, F, min_time, max_time = pickle.load(f)

accuracy_e1 = 89.6
acceptance_e1 = 100
accuracy_e2 = 90.08
acceptance_e2 = 100

with open(f'fix_{network}_x_f_2016_23.sav', 'wb') as f:
    pickle.dump([ X, F, min_time, max_time, accuracy_e1, acceptance_e1, accuracy_e2, acceptance_e2 ], f)
