"""Run a command, write wall time and peak RSS of the child as JSON: measure.py OUT.json cmd..."""
import json, resource, subprocess, sys, time
t = time.time(); rc = subprocess.call(sys.argv[2:]); wall = time.time() - t
rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
json.dump({"cmd": sys.argv[2:], "rc": rc, "wall_s": round(wall, 2), "peak_rss_mb": round(rss, 1)}, open(sys.argv[1], "w"))
sys.exit(rc)
