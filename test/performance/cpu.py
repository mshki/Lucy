import subprocess
import time
import psutil
import csv

# Start the C++ binary and pipe the inputs
process = subprocess.Popen(
    ['../build/bin/PokerBotMAIF'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True
)

# Send the inputs (Mode 1, 100,000 iterations)
process.stdin.write("1\n100000\n")
process.stdin.flush()

ps_process = psutil.Process(process.pid)

print("Profiling started... Logging to benchmark_log.csv")

with open('benchmark_log.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Time (s)', 'CPU Usage (%)', 'RAM Usage (MB)'])
    
    start_time = time.time()
    
    # Loop until the C++ process finishes
    while process.poll() is None:
        try:
            # Get CPU and Memory for this specific process
            cpu = ps_process.cpu_percent(interval=0.5) 
            mem = ps_process.memory_info().rss / (1024 * 1024) # Convert bytes to MB
            current_time = round(time.time() - start_time, 2)
            
            writer.writerow([current_time, cpu, mem])
            
            # Note: For GPU profiling via Python, you'd typically query `nvidia-smi` 
            # using subprocess here, or use a library like `pynvml`.
            
        except psutil.NoSuchProcess:
            break

print("Training complete. Data saved.")