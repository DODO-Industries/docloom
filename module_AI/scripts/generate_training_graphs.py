"""
Parses training log and generates ASCII / Markdown loss progression graphs.
"""
import re
import os
import sys

def parse_pretrain_log(log_path):
    steps = []
    losses = []
    val_steps = []
    val_losses = []
    
    if not os.path.exists(log_path):
        return steps, losses, val_steps, val_losses
        
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # epoch 0 step 100/2248 loss 20.9124 (avg 22.3357) lr 3.00e-04
            m = re.search(r"step\s+(\d+)/\d+\s+loss\s+([\d\.]+)", line)
            if m:
                s = int(m.group(1))
                l = float(m.group(2))
                steps.append(s)
                losses.append(l)
            # -- validation loss: 10.7314 (held-out, never trained on)
            m_val = re.search(r"validation loss:\s+([\d\.]+)", line)
            if m_val and steps:
                val_steps.append(steps[-1])
                val_losses.append(float(m_val.group(1)))
                
    return steps, losses, val_steps, val_losses

def render_ascii_sparkline(values, width=40):
    if not values:
        return ""
    ticks = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    min_v, max_v = min(values), max(values)
    rng = max_v - min_v if max_v != min_v else 1.0
    
    # Resample to width
    chunk_size = max(1, len(values) // width)
    sampled = [min(values[i:i+chunk_size]) for i in range(0, len(values), chunk_size)][:width]
    
    line = "".join(ticks[min(len(ticks)-1, int((v - min_v) / rng * (len(ticks)-1)))] for v in sampled)
    return line

if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = r"C:\Users\ratho\.gemini\antigravity-ide\brain\48b582bc-4d34-4b36-ad9c-efff2d2438f8\.system_generated\tasks\task-1709.log"
    steps, losses, v_steps, v_losses = parse_pretrain_log(log_file)
    print(f"Parsed {len(steps)} training steps, {len(v_losses)} validation points.")
    if losses:
        print(f"Initial Loss: {losses[0]:.4f} -> Latest Loss: {losses[-1]:.4f}")
        print("Loss Curve (downward slope = healthy learning):")
        print(render_ascii_sparkline(losses))
