import os
import re
import argparse
import numpy as np

def parse_log(file_path):
    if not os.path.exists(file_path):
        return 'OOM'  # Assume missing log as failure

    with open(file_path, 'r') as f:
        content = f.read()

    if 'out of memory' in content.lower():
        return 'OOM'

    # Find all relevant memory lines
    max_mem_gb = None
    # Pattern for timer.py lines: MaxMemAllocated=58.53GB
    pattern_timer = r'MaxMemAllocated=([\d\.]+)GB'
    # Pattern for early memory lines: max allocated: 59937.87841796875 (in MB)
    pattern_early = r'max allocated: ([\d\.]+)'

    last_max_mem = 0.0
    for line in content.splitlines():
        match_timer = re.search(pattern_timer, line)
        if match_timer:
            current_mem = float(match_timer.group(1))
            if current_mem > last_max_mem:
                last_max_mem = current_mem

        match_early = re.search(pattern_early, line)
        if match_early:
            current_mem_mb = float(match_early.group(1))
            current_mem_gb = current_mem_mb / 1024.0
            if current_mem_gb > last_max_mem:
                last_max_mem = current_mem_gb

    if last_max_mem > 0:
        return round(last_max_mem, 2)
    else:
        return 'OOM'  # No memory info found

def main():
    parser = argparse.ArgumentParser(description='Visualize memory from training logs.')
    parser.add_argument('num_gpus', type=int, help='Number of GPUs (e.g., 4)')
    args = parser.parse_args()

    num_gpus = args.num_gpus
    methods_file = {'X-MoE': 'XMoE', 'DeepSpeed-MoE': 'DeepSpeed-MoE', 'Tutel': 'Tutel'}
    methods_display = ['X-MoE', 'DeepSpeed-MoE', 'Tutel']
    batch_sizes = [1, 2, 4, 8, 16]

    # Assuming script is in ./visualization/, logs in ../
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

    results = {method: [] for method in methods_display}

    for method_display in methods_display:
        method_file = methods_file[method_display]
        for bs in batch_sizes:
            log_file = os.path.join(log_dir, f'n{num_gpus}-Small-{method_file}-batch{bs}.log')
            result = parse_log(log_file)
            results[method_display].append(result)

    # Print the 3x5 table
    print('Batch Size | 1      | 2      | 4      | 8      | 16     ')
    print('-----------------------------------------------------')
    for method in methods_display:
        row = f'{method:<10} | '
        for val in results[method]:
            if isinstance(val, str):
                row += f'{val:<6} | '
            else:
                row += f'{val:<6} | '
        print(row)

if __name__ == '__main__':
    main()