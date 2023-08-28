#!/usr/bin/env python3

import json

with open('progression.json') as f:
    data = json.load(f)

for key in data:
    if key != 'average_times':
        accuraccy = 100 * sum(data[key]['correct_exit']) / 60000
        total_time = 0
        for t, c in zip(data['average_times'], data[key]['chosen_exit']):
            total_time += t * c
        average_time = total_time / 60000

        print(f'{key} {accuraccy:.2f} {average_time:.2f}')