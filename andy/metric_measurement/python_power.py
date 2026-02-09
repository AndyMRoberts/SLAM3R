import subprocess
import time

scene = 'example_1'
# start logging power consumption
try: 
    power_process = subprocess.Popen(['sudo', './power', f'power_log_{scene}.csv'])  # Start power measurement
except Exception as e:
    print(f"Failed to start power measurement: {e}")


time.sleep(10)

# Stop logging power consumption
print("Terminating power measurement...")
power_process.terminate()
