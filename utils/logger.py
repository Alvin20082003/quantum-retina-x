import csv
from datetime import datetime
from pathlib import Path

FIELDS = ["timestamp","run_name","dataset","model","accuracy",
          "macro_f1","auc_roc","kappa","grade1_sensitivity","notes"]

class ExperimentLogger:
    def __init__(self, results_dir="experiments/results", filename="results_log.csv"):
        self.path = Path(results_dir) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def log(self, **row):
        row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
        clean = {k: row.get(k, "") for k in FIELDS}
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(clean)
        print(f"[logger] appended '{clean['run_name']}' -> {self.path}")
