import subprocess
import time
import json
import os
import sys

LOG_FILE = ".rsi_state/run_log.jsonl"

def monitor():
    print(f"[MONITOR] Starting evolution_loop.py...")
    # Start evolution_loop.py in the background
    with open("evolution.log", "w") as out:
        process = subprocess.Popen([sys.executable, "evolution_loop.py"], stdout=out, stderr=out)

    print(f"[MONITOR] evolution_loop.py started with PID {process.pid}")
    print(f"[MONITOR] Watching {LOG_FILE} for Discovery Signal...")

    f = None
    last_inode = None

    try:
        while True:
            # Check if process is still running
            if process.poll() is not None:
                print("[MONITOR] evolution_loop.py exited unexpectedly.")
                break

            # Handle file opening/reopening
            if f is None:
                if os.path.exists(LOG_FILE):
                    try:
                        f = open(LOG_FILE, "r")
                        stat = os.fstat(f.fileno())
                        last_inode = stat.st_ino
                        # print(f"[MONITOR] Opened log file (inode {last_inode})")
                    except Exception:
                        time.sleep(0.5)
                        continue
                else:
                    time.sleep(0.5)
                    continue

            # Read new lines
            where = f.tell()
            line = f.readline()
            if not line:
                time.sleep(0.1)
                f.seek(where) # Reset position to try again

                # Check for file rotation/deletion
                if not os.path.exists(LOG_FILE):
                    f.close()
                    f = None
                    last_inode = None
                    continue

                try:
                    current_stat = os.stat(LOG_FILE)
                    if current_stat.st_ino != last_inode:
                        # File replaced
                        f.close()
                        f = None
                        last_inode = None
                        continue
                    # Also check if file was truncated (size smaller than current position)
                    if current_stat.st_size < where:
                         f.close()
                         f = None
                         last_inode = None
                         continue
                except FileNotFoundError:
                    f.close()
                    f = None
                    last_inode = None
                    continue
            else:
                # Process the line
                try:
                    data = json.loads(line)
                    # Check for signal
                    # "accepted": true AND "novelty": 1.0
                    if data.get("accepted") is True and data.get("novelty") == 1.0:
                        print("\n[SIGNAL DETECTED] Discovery Signal Found!")
                        print(json.dumps(data, indent=2))
                        print("[MONITOR] Terminating evolution_loop.py...")
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        return
                except json.JSONDecodeError:
                    pass
    except KeyboardInterrupt:
        print("\n[MONITOR] Stopping...")
        if process:
            process.terminate()

if __name__ == "__main__":
    monitor()
