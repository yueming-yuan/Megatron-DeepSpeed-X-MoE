import re
import matplotlib.pyplot as plt
import numpy as np
import sys

def parse_log(log_text):
    pattern = r"time \(ms\) \| fwd: \d+\.\d+ \(fwd_moe: (\d+\.\d+), 1st_a2a: (\d+\.\d+), experts: (\d+\.\d+), 2nd_a2a: (\d+\.\d+), top_k: (\d+\.\d+)\), dispatch: (\d+\.\d+), combine: (\d+\.\d+)"
    
    matches = re.findall(pattern, log_text)
    
    data = []
    for match in matches:
        fwd_moe, a1, experts, a2, topk, dispatch, combine = map(float, match)
        others = fwd_moe - (float(a1) + experts + float(a2) + topk + dispatch + combine)
        data.append({
            'fwd_moe': fwd_moe,
            '1st_a2a': a1,
            'experts': experts,
            '2nd_a2a': a2,
            'top_k': topk,
            'dispatch': dispatch,
            'combine': combine,
            'others': others
        })
    
    return data

def average_data(data, start_iter=5, end_iter=50):
    selected = data[start_iter-1:end_iter]
    if not selected:
        return None
    
    avg = {}
    keys = ['others', 'combine', '2nd_a2a', 'experts', '1st_a2a', 'dispatch', 'top_k']
    
    for key in keys:
        avg[key] = np.mean([d.get(key, 0) for d in selected])
    
    return avg

def plot_stacked_bar(avg_data):
    if not avg_data:
        print("No data to plot.")
        return
    
    labels = ['others', 'combine', '2nd_a2a', 'experts', '1st_a2a', 'dispatch', 'gate']
    values = [avg_data[k] for k in ['others', 'combine', '2nd_a2a', 'experts', '1st_a2a', 'dispatch', 'top_k']]
    
    # Print average times
    print("Average times (ms):")
    for label, value in zip(labels, values):
        print(f"{label}: {value:.2f}")
    
    colors = ['gray', 'purple', 'orange', 'yellow', 'darkred', 'cyan', 'navy']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bottom = 0
    for label, value, color in zip(labels, values, colors):
        ax.bar('X-MoE', value, bottom=bottom, label=label, color=color)
        bottom += value
    
    ax.set_ylabel('Time (ms)')
    ax.set_title('10.1B MoE (Small)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('time_breakdown.png')
    print("Plot saved to time_breakdown.png")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <log_file_path>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    with open(log_file, 'r') as f:
        log_text = f.read()
    
    parsed_data = parse_log(log_text)
    avg_data = average_data(parsed_data)
    plot_stacked_bar(avg_data)