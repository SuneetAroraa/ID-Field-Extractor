import json
from datetime import datetime,timezone
 
def log(trace_path, stage, data):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        **data
    }
    with open(trace_path, "a") as f:
        f.write(json.dumps(entry) + "\n")