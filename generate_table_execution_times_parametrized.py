
# usage: python generate_table_execution_times_parametrized.py <folder_parametrized> <folder_non_parametrized>

import sys
import os
import re
import matplotlib.pyplot as plt
import numpy as np


# The important lines of the file are these:

# read_vertical_slices: 0.002184 s (279 calls), avg: 0.007829 ms, proportion: 0.01%
# save_reduce: 0.166946 s (54 calls), avg: 3.091588 ms, proportion: 1.13%
# write_horizontal_slices: 0.006632 s (279 calls), avg: 0.023772 ms, proportion: 0.04%
# initialize: 0.016154 s (27 calls), avg: 0.598282 ms, proportion: 0.11%
# read_horizontal_slices: 0.006287 s (279 calls), avg: 0.022535 ms, proportion: 0.04%
# LBM_step: 14.558249 s (4500 calls), avg: 3.235166 ms, proportion: 98.66%
# execution_time: 16.544197 s
# initialization_time: 0.023508 s
# total_time: 16.567705 s
# LBM execution time (including PaRSEC overhead): 16.064520 s

# parse_output_file will store every important line in a dictionary, and return it
def parse_output_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    lines = content.split('\n')
    times = {}
    for line in lines:
        if line.startswith('save_reduce'):
            match = re.search(r'(\d+\.\d+) s', line)
            if match:
                times['save_reduce'] = float(match.group(1))
        if line.startswith('initialize'):
            match = re.search(r'(\d+\.\d+) s', line)
            if match:
                times['initialize'] = float(match.group(1))
        if line.startswith('read_vertical_slices'):
            match = re.search(r'(\d+\.\d+) s', line)
            if match:
                times['read_vertical_slices'] = float(match.group(1))
        if line.startswith('write_horizontal_slices'):
            match = re.search(r'(\d+\.\d+) s', line)
            if match:
                times['write_horizontal_slices'] = float(match.group(1))
        if line.startswith('read_horizontal_slices'):
            match = re.search(r'(\d+\.\d+) s', line)
            if match:
                times['read_horizontal_slices'] = float(match.group(1))
        if line.startswith('LBM_step'):
            match = re.search(r'(\d+\.\d+) s', line)
            if match:
                times['LBM_step'] = float(match.group(1))
        if line.startswith('execution_time'):
            match = re.search(r'(\d+\.\d+) s', line)
            if match:
                times['execution_time'] = float(match.group(1))
        if line.startswith('initialization_time'):
            match = re.search(r'(\d+\.\d+) s', line)
            if match:
                times['initialization_time'] = float(match.group(1))
        if line.startswith('total_time'):
            match = re.search(r'(\d+\.\d+) s', line)
            if match:
                times['total_time'] = float(match.group(1))
        if line.startswith('LBM execution time (including PaRSEC overhead)'):
            match = re.search(r'(\d+\.\d+) s', line)
            if match:
                times['LBM_execution_time'] = float(match.group(1))
    if len(times) != 10:
        return None
    return times
    

if len(sys.argv) != 3:
    print("Usage: python plot_execution_times.py <folder_parametrized> <folder_non_parametrized>")
    sys.exit(1)

folder_parametrized = sys.argv[1]
folder_non_parametrized = sys.argv[2]

assert os.path.isdir(folder_parametrized)
assert os.path.isdir(folder_non_parametrized)
assert folder_parametrized != folder_non_parametrized

times_parametrized = {}
times_non_parametrized = {}

for folder in [folder_parametrized, folder_non_parametrized]:
    target_time_dict = times_parametrized if folder == folder_parametrized else times_non_parametrized
    files = os.listdir(folder)
    files = [file for file in files if os.path.isdir(folder + file)]
    files = [file+'/output.txt' for file in files]
    files.sort()
    times = {}
    for file in files:
        time_results = parse_output_file(folder + file)
        if time_results is not None:
            experiment_id = int(file.split('/')[0].split('_')[1])
            target_time_dict[experiment_id] = time_results
        else:
            print('None', folder + file)

# Average the times across the different experiments
for folder in [folder_parametrized, folder_non_parametrized]:
    target_time_dict = times_parametrized if folder == folder_parametrized else times_non_parametrized
    average_dict_times = {}
    # initialize average_dict_times with zeros on all the keys
    for experiment_id in target_time_dict:
        for key in target_time_dict[experiment_id]:
            if key not in average_dict_times:
                average_dict_times[key] = 0
    # sum all the times
    for experiment_id in target_time_dict:
        for key in target_time_dict[experiment_id]:
            average_dict_times[key] += target_time_dict[experiment_id][key]
    # divide by the number of experiments
    for key in average_dict_times:
        average_dict_times[key] /= len(target_time_dict)
    # copy the result back to the original dictionary
    target_time_dict = {key: value for key, value in average_dict_times.items()}
    if folder == folder_parametrized:
        times_parametrized = target_time_dict
    else:
        times_non_parametrized = target_time_dict


# print(times_parametrized)
# print(times_non_parametrized)
# # {'read_vertical_slices': 0.0021629999999999996, 'write_horizontal_slices': 0.006483593749999999, 'read_horizontal_slices': 0.0062445, 'LBM_step': 14.511180562500002, 'execution_time': 16.46684421875, 'initialization_time': 0.02379475, 'total_time': 16.49063896875, 'LBM_execution_time': 15.983198937499996}
# # {'read_vertical_slices': 0.0021656562499999996, 'write_horizontal_slices': 0.00647534375, 'read_horizontal_slices': 0.0062559999999999985, 'LBM_step': 14.509349406250003, 'execution_time': 16.4732023125, 'initialization_time': 0.023451875, 'total_time': 16.496654187499995, 'LBM_execution_time': 15.990682906249996}

# Generate the table in latex format
print("%% This table was generated by generate_table_execution_times_parametrized.py")
print("%% You should not modify it manually, as it will be overwritten.")
print("\\begin{table}")
print("\\centering")
print("\\begin{tabular}{|c|c|c|c|c|c|}")
print("\\hline")
print(" & GPU kernels & initialization & whole run & PaRSEC overhead \\\\")
print("\\hline")
print("Parametrized &", end=" ")
sum_kernels=times_parametrized['read_vertical_slices']+times_parametrized['write_horizontal_slices']+times_parametrized['read_horizontal_slices']+times_parametrized['LBM_step']+times_parametrized['initialize']+times_parametrized['save_reduce']
parsec_overhead=times_parametrized['execution_time']/sum_kernels
print(f"{sum_kernels:.4f}", end=" s & ")
print(f"{times_parametrized['initialization_time']:.4f} s", end=" & ")
print(f"{times_parametrized['execution_time']:.4f} s", end=" & ")
print(f"{((parsec_overhead-1)*100):.2f}\%", end=" \\\\")
print("\\hline")
print("Non-parametrized &", end=" ")
sum_kernels=times_non_parametrized['read_vertical_slices']+times_non_parametrized['write_horizontal_slices']+times_non_parametrized['read_horizontal_slices']+times_non_parametrized['LBM_step']+times_non_parametrized['initialize']+times_non_parametrized['save_reduce']
parsec_overhead=times_non_parametrized['execution_time']/sum_kernels
print(f"{sum_kernels:.4f}", end=" s & ")
print(f"{times_non_parametrized['initialization_time']:.4f} s", end=" & ")
print(f"{times_non_parametrized['execution_time']:.4f} s", end=" & ")
print(f"{((parsec_overhead-1)*100):.2f}\%", end=" \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\caption{Execution times for the parametrized and non-parametrized versions of the code.}")
print("\\label{tab:execution_times}")
print("\\end{table}")
