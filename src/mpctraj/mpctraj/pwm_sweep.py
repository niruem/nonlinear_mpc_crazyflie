import time
from pathlib import Path
from crazyflie_py import Crazyswarm
import csv
from datetime import datetime

# THIS FILE WAS USED TO SWEEP THROUGH PWM VALUES IN ORDER TO MEASURE THE WEIGHT DIFFERENCE ON THE SCALE

COMMAND_RATE = 50   # Hz - send commands at 50Hz to keep watchdog happy
DWELL_TIME = 6.0    # seconds per step

def send_for_duration(cf, cmd_val, duration, rate=COMMAND_RATE):
    """Send a constant cmdVel repeatedly for `duration` seconds at `rate` Hz."""
    interval = 1.0 / rate
    end_time = time.time() + duration
    while time.time() < end_time:
        cf.cmdVel(0.0, 0.0, 0.0, float(cmd_val))
        time.sleep(interval)

def main():
    swarm = Crazyswarm()
    cf = swarm.allcfs.crazyflies[0]

    # Zero out before starting
    send_for_duration(cf, 0.0, 2.0)

    calibration_values = range(10000, 60001, 5000)
    
    print("Starting thrust calibration sweep...")
    print(f"Total steps: {len(list(calibration_values))} | Dwell: {DWELL_TIME}s each")
    print(f"Estimated total time: {len(calibration_values) * DWELL_TIME / 60:.1f} minutes")
    print("-" * 40)

    results = []

    try:
        for cmd_val in calibration_values:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Sending {cmd_val} — record thrust now!")
            send_for_duration(cf, cmd_val, DWELL_TIME)
            results.append({
                "cmd_val": cmd_val,
                "timestamp": datetime.now().isoformat(),
                "thrust_grams": ""
            })
        

    except KeyboardInterrupt:
        print("\nSweep interrupted by user.")

    finally:
        send_for_duration(cf, 0.0, 1.0)  # Ramp down cleanly before cutting
        cf.cmdVel(0.0, 0.0, 0.0, 0.0)
        print("Motors stopped.")

        # if results:
        #     timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     csv_path = Path(__file__).parent / f"calibration_{timestamp_str}.csv"
        #     with open(csv_path, "w", newline="") as f:
        #         writer = csv.DictWriter(f, fieldnames=["cmd_val", "timestamp", "thrust_grams"])
        #         writer.writeheader()
        #         writer.writerows(results)
        #     print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()