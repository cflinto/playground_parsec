
# usage: python plot_execution_times.py <folder>
# <folder> is the folder containing the results of the runs

# The output files are of the form <folder>/experiment_xxx/output.txt
# The id of the experiment tells the changing parameters: overlap_x and step_kernel_id
# The id xxyyz means: experiment xx, overlap_x=yy, step_kernel_id=z
# z is the modulo 10 of the experiment id
# y is the modulo 100 of the experiment id divided by 10
# x is the experiment id divided by 10000
# (don't forget to add leading zeros to the experiment id)

# The important lines in the output file are (for example):
# read_vertical_slices: 0.016338 s (2241 calls), avg: 0.007290 ms, proportion: 0.11%
# write_horizontal_slices: 0.029638 s (2241 calls), avg: 0.013225 ms, proportion: 0.20%
# read_horizontal_slices: 0.030856 s (2241 calls), avg: 0.013769 ms, proportion: 0.21%
# LBM_step: 14.786609 s (4500 calls), avg: 3.285913 ms, proportion: 98.26%

import sys
import os
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_output_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    lines = content.split('\n')
    times = {}
    for line in lines:
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
    if len(times) != 4:
        return None
    return times

if len(sys.argv) != 2:
    print("Usage: python plot_execution_times.py <folder>")
    sys.exit(1)

folder = sys.argv[1]
files = os.listdir(folder)
files = [file for file in files if os.path.isdir(folder + file)]
files = [file+'/output.txt' for file in files]
files.sort()
times = {}
for file in files:
    time_results = parse_output_file(folder + file)
    if time_results is not None:
        experiment_id = int(file.split('/')[0].split('_')[1])
        times[experiment_id] = time_results

# Store the results in an elegant array that we can access like so: kernel_time[overlap][step_kernel_id][function] (average of the times)

# Will make the correspondence tuples (overlap, step_kernel_id, kernel_name) -> experiment_id
kernel_times={}
for experiment_id in times:
    overlap = int((experiment_id/10)%100)
    step_kernel_id = int(experiment_id%10)
    # iterate over the keys of the dictionary
    for kernel_name in times[experiment_id]:
        if (overlap, step_kernel_id, kernel_name) not in kernel_times:
            kernel_times[(overlap, step_kernel_id, kernel_name)] = []
        kernel_times[(overlap, step_kernel_id, kernel_name)].append(times[experiment_id][kernel_name])

# Now we have the average times for each kernel, for each overlap and step_kernel_id
for key in kernel_times:
    kernel_times[key] = np.mean(kernel_times[key])

print(kernel_times) # {(10, 0, 'read_vertical_slices'): 0.003415, (10, 0, 'write_horizontal_slices'): 0.009269, (10, 0, 'read_horizontal_slices'): 0.008782, (10, 0, 'LBM_step'): 14.365579, (10, 1, 'read_vertical_slices'): 0.003356, ...}

# Now we can plot the results
overlaps = []
step_kernel_ids = []
kernel_names = []
for key in kernel_times:
    if key[0] not in overlaps:
        overlaps.append(key[0])
    if key[1] not in step_kernel_ids:
        step_kernel_ids.append(key[1])
    if key[2] not in kernel_names:
        kernel_names.append(key[2])
overlaps.sort()
step_kernel_ids.sort()
kernel_names.sort()

# overlaps = [i for i in range(1, 21)]

print(overlaps)
print(step_kernel_ids)
print(kernel_names)

# Plot the results
# Bar plots
# 3 bars per group (overlap_x)
# A bar is subdivided in 4 parts (LBM_step, read_vertical_slices, read_horizontal_slices, write_horizontal_slices), depending on the time spent in each kernel
# We hence have 32 groupes of 3 bars (32 possible overlap_x values and 3 possible step_kernel_id values)

# We will different colors for the different parts of the bar

# plt.rcParams['hatch.linewidth'] = 10.0

fig, ax = plt.subplots( figsize=(12, 8) )
index = np.arange(len(overlaps))
bar_width = 0.5
opacity = 0.8

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

#for i in range(len(step_kernel_ids)):
i = 0 # Only plot the best kernel to avoid overloading
for kernel_name in 'write_horizontal_slices', 'read_horizontal_slices', 'read_vertical_slices', 'LBM_step':
    for overlap in overlaps:
        # Verify that ALL the kernels kernel_times[(overlap, step_kernel_ids[i], ....)] exist (... = any kernel_name)
        if (overlap, step_kernel_ids[i], 'write_horizontal_slices') not in kernel_times:
            continue
        if (overlap, step_kernel_ids[i], 'read_horizontal_slices') not in kernel_times:
            continue
        if (overlap, step_kernel_ids[i], 'read_vertical_slices') not in kernel_times:
            continue
        if (overlap, step_kernel_ids[i], 'LBM_step') not in kernel_times:
            continue

        pattern=''
        # if i == 0:
        #     pattern = '\\\\'
        # elif i == 1:
        #     pattern = '..'
        # elif i == 2:
        #     pattern = '//'

        height = 0
        if kernel_name == 'write_horizontal_slices':
            color = CB_color_cycle[3]
            height = kernel_times[(overlap, step_kernel_ids[i], 'write_horizontal_slices')]+kernel_times[(overlap, step_kernel_ids[i], 'read_horizontal_slices')]+kernel_times[(overlap, step_kernel_ids[i], 'read_vertical_slices')]+kernel_times[(overlap, step_kernel_ids[i], 'LBM_step')]
        elif kernel_name == 'read_horizontal_slices':
            color = CB_color_cycle[2]
            height = kernel_times[(overlap, step_kernel_ids[i], 'read_horizontal_slices')]+kernel_times[(overlap, step_kernel_ids[i], 'read_vertical_slices')]+kernel_times[(overlap, step_kernel_ids[i], 'LBM_step')]
        elif kernel_name == 'read_vertical_slices':
            color = CB_color_cycle[1]
            height = kernel_times[(overlap, step_kernel_ids[i], 'read_vertical_slices')]+kernel_times[(overlap, step_kernel_ids[i], 'LBM_step')]
        elif kernel_name == 'LBM_step':
            color = CB_color_cycle[0]
            height = kernel_times[(overlap, step_kernel_ids[i], 'LBM_step')]
        plt.bar((overlap-1) + (i+1)*bar_width, height, bar_width, alpha=opacity, color=color, hatch=pattern)


plt.xlabel('overlap_x')
plt.ylabel('Execution time (s)')
plt.title('Execution times of the D2Q9 kernels (naive LBM_step)', fontsize=24)
plt.xticks(index + bar_width, overlaps)


# Set the y min and max values to the min and max of the LBM_step kernel
ymin = 99999
ymax = 0
for overlap in overlaps:
    # for i in range(len(step_kernel_ids)):
        i = 0
        if (overlap, step_kernel_ids[i], 'LBM_step') in kernel_times:
            ymin = min(ymin, kernel_times[(overlap, step_kernel_ids[i], 'LBM_step')])
            ymax = max(ymax, kernel_times[(overlap, step_kernel_ids[i], 'LBM_step')])
plt.ylim([ymin*0.99, ymax*1.02])

# The legend is the colors of the bars
import matplotlib.patches as mpatches


red_patch = mpatches.Patch(facecolor=CB_color_cycle[0], edgecolor='black', label='LBM_step')
blue_patch = mpatches.Patch(facecolor=CB_color_cycle[1], edgecolor='black', label='read_vertical_slices')
green_patch = mpatches.Patch(facecolor=CB_color_cycle[2], edgecolor='black', label='read_horizontal_slices')
yellow_patch = mpatches.Patch(facecolor=CB_color_cycle[3], edgecolor='black', label='write_horizontal_slices')
plt.legend(handles=[red_patch, blue_patch, green_patch, yellow_patch])
# # Hatch patterns, white background, black patterns
# backslash_patch = mpatches.Patch(hatch='\\\\\\\\\\\\', color='white', edgecolor='black', label='step_kernel_id=0')
# dot_patch = mpatches.Patch(hatch='......', color='white', edgecolor='black', label='step_kernel_id=1')
# slash_patch = mpatches.Patch(hatch='//////', color='white', edgecolor='black', label='step_kernel_id=2')
# plt.legend(handles=[red_patch, blue_patch, green_patch, yellow_patch, backslash_patch, dot_patch, slash_patch])

plt.show()
