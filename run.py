"""One-command launcher with a process-level two-hour watchdog."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def supervise(command, seconds=7140):
    env = dict(os.environ, GTSRB_HARD_DEADLINE=str(time.time()+seconds))
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        return process.wait(timeout=seconds)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        print("Stopping run; completed-epoch checkpoints remain available.", flush=True)
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        return 124


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    raise SystemExit(supervise([sys.executable, "-u", "-m", "gtsrb.run", *sys.argv[1:]]))
