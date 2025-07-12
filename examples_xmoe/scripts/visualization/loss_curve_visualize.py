import re
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def parse_log(log_text, filename):
    # Generalized pattern to match iteration / total, lm loss, moe loss
    pattern = r"iteration\s+(\d+)/\s+\d+ .* lm loss: (\d+\.\d+E[+-]\d+) \| moe loss: (\d+\.\d+E[+-]\d+)"
    
    matches = re.findall(pattern, log_text)
    
    data = []
    for match in matches:
        iteration = int(match[0])
        lm_loss = float(match[1])
        moe_loss = float(match[2])
        data.append({
            'iteration': iteration,
            'lm_loss': lm_loss,
            'moe_loss': moe_loss,
            'total_loss': lm_loss + moe_loss  # Optional total
        })
    
    sorted_data = sorted(data, key=lambda x: x['iteration'])
    print(f"Parsed {len(sorted_data)} iterations from log file: {filename}")
    if sorted_data:
        last_iter = sorted_data[-1]['iteration']
        last_lm_loss = sorted_data[-1]['lm_loss']
        print(f"File: {filename} - Last step: iteration={last_iter}, lm_loss={last_lm_loss}")
    return sorted_data

def plot_losses(data1, data2, label1, label2):
    if not data1 or not data2:
        print("No data to plot.")
        return
    
    iterations1 = [d['iteration'] for d in data1]
    lm_losses1 = [d['lm_loss'] for d in data1]
    
    iterations2 = [d['iteration'] for d in data2]
    lm_losses2 = [d['lm_loss'] for d in data2]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations1, lm_losses1, label=f'{label1} LM Loss', color='blue')
    plt.plot(iterations2, lm_losses2, label=f'{label2} LM Loss', color='red')
    
    plt.xlabel('Iteration')
    plt.ylabel('LM Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_comparison.png')
    print("Plot saved to loss_comparison.png")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <log_file1> <log_file2>")
        sys.exit(1)
    
    log_file1 = sys.argv[1]
    log_file2 = sys.argv[2]
    
    with open(log_file1, 'r') as f:
        log_text1 = f.read()
    
    with open(log_file2, 'r') as f:
        log_text2 = f.read()
    
    parsed_data1 = parse_log(log_text1, log_file1)
    parsed_data2 = parse_log(log_text2, log_file2)
    
    label1 = os.path.basename(log_file1)
    label2 = os.path.basename(log_file2)
    
    plot_losses(parsed_data1, parsed_data2, label1, label2)