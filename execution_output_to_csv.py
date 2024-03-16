
# Input: a list of files
# Output: a csv, left column=thread num, block num, other columns=average time of kernels write_horizontal_slices, read_horizontal_slices, read_vertical_slices, LBM_step

# The interesting lines of the output files are:

# read_vertical_slices: 0.004722 s (441 calls), avg: 0.010707 ms, proportion: 0.02%
# save_reduce: 0.167772 s (54 calls), avg: 3.106892 ms, proportion: 0.78%
# write_horizontal_slices: 0.010358 s (441 calls), avg: 0.023487 ms, proportion: 0.05%
# initialize: 0.019571 s (27 calls), avg: 0.724840 ms, proportion: 0.09%
# read_horizontal_slices: 0.009500 s (441 calls), avg: 0.021541 ms, proportion: 0.04%
# LBM_step: 21.313072 s (4500 calls), avg: 4.736238 ms, proportion: 99.02%

# and

#   READ_HORIZONTAL_SLICES_BLOCK_NUM = 4096
#   READ_HORIZONTAL_SLICES_THREAD_NUM = 64
#   WRITE_HORIZONTAL_SLICES_BLOCK_NUM = 4096
#   WRITE_HORIZONTAL_SLICES_THREAD_NUM = 64
#   READ_VERTICAL_SLICES_BLOCK_NUM = 4096
#   READ_VERTICAL_SLICES_THREAD_NUM = 64
#   LBM_STEP_BLOCK_NUM = 4096
#   LBM_STEP_THREAD_NUM = 64

import sys
import os
import re


# First, read the arguments
if len(sys.argv) < 2:
    print("Usage: python execution_output_to_csv.py <output_file1> <output_file2> ...")
    sys.exit(1)

output_files = sys.argv[1:]

print("thread_num;block_num;read_horizontal_slices;read_vertical_slices;write_horizontal_slices;LBM_step")

# Then, read the files
for output_file in output_files:
    with open(output_file, 'r') as f:
        lines = f.readlines()

    # Then, parse the lines
    read_horizontal_slices = 0
    read_vertical_slices = 0
    write_horizontal_slices = 0
    LBM_step = 0
    block_num = 0
    thread_num = 0
    for line in lines:
        m = re.search(r'read_horizontal_slices: ([0-9]+\.[0-9]+) s', line)
        if m:
            read_horizontal_slices = float(m.group(1))
        m = re.search(r'read_vertical_slices: ([0-9]+\.[0-9]+) s', line)
        if m:
            read_vertical_slices = float(m.group(1))
        m = re.search(r'write_horizontal_slices: ([0-9]+\.[0-9]+) s', line)
        if m:
            write_horizontal_slices = float(m.group(1))
        m = re.search(r'LBM_step: ([0-9]+\.[0-9]+) s', line)
        # Assuming the thread_num and block_num are the same for all kernels
        if m:
            LBM_step = float(m.group(1))
        m = re.search(r'READ_HORIZONTAL_SLICES_BLOCK_NUM = ([0-9]+)', line)
        if m:
            block_num = int(m.group(1))
        m = re.search(r'READ_HORIZONTAL_SLICES_THREAD_NUM = ([0-9]+)', line)
        if m:
            thread_num = int(m.group(1))

    # Finally, print the results
    print(f"{thread_num};{block_num};{read_horizontal_slices};{read_vertical_slices};{write_horizontal_slices};{LBM_step}")

