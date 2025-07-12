import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_log(file_path):
    if not os.path.exists(file_path):
        return 'OOM'  # Assume missing log as failure

    with open(file_path, 'r') as f:
        content = f.read()

    if 'out of memory' in content.lower():
        return 'OOM'

    tflops_list = []
    pattern = r'iteration\s+(\d+)/\s+\d+\s+\|\s+.*\|\s+TFLOPs:\s+(\d+\.\d+)\s+\|'
    for match in re.finditer(pattern, content):
        iter_num = int(match.group(1))
        tflops = float(match.group(2))
        if 5 <= iter_num <= 50:
            tflops_list.append(tflops)

    if len(tflops_list) == 0:
        return 'OOM'  # No valid iterations found

    avg_tflops = np.mean(tflops_list)
    return round(avg_tflops, 2)

def main():
    parser = argparse.ArgumentParser(description='Visualize throughputs from training logs.')
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

    # Compute max TFLOPs for each method
    max_tflops = {}
    for method in methods_display:
        vals = [v for v in results[method] if isinstance(v, (int, float))]
        max_tflops[method] = max(vals) if vals else 0

    # Plot bar chart with custom colors
    colors = ['#FFD700', '#2E8B57', '#4169E1']  # Gold for X-MoE, SeaGreen for DeepSpeed-MoE, RoyalBlue for Tutel
    fig, ax = plt.subplots()
    ax.bar(max_tflops.keys(), max_tflops.values(), color=colors)
    ax.set_ylabel('Max TFLOPs')
    ax.set_title('Max Throughput Comparison')
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'throughput_comparison.png'))
    plt.show()

if __name__ == '__main__':
    main()