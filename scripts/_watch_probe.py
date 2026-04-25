#!/usr/bin/env python
"""Wait for subproof probe results to appear."""
import time, os
start = time.time()
path = "working/subproof_probe_results"
while time.time() - start < 900:
    if os.path.isdir(path) and os.listdir(path):
        print(f"DONE after {int(time.time()-start)}s! Files: {os.listdir(path)}")
        break
    elapsed = int(time.time() - start)
    print(f"  {elapsed}s elapsed, waiting for results...", flush=True)
    time.sleep(30)
else:
    print(f"TIMEOUT after 900s")