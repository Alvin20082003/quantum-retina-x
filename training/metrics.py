import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             cohen_kappa_score, recall_score)

def compute_metrics(y_true, y_prob, num_classes=5):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(1)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    per_grade = recall_score(y_true, y_pred, labels=list(range(num_classes)),
                             average=None, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro",
                            labels=list(range(num_classes)))
    except Exception:
        auc = float("nan")
    return {"accuracy": acc, "macro_f1": macro_f1, "auc_roc": auc, "kappa": kappa,
            "per_grade_sensitivity": [float(s) for s in per_grade],
            "grade1_sensitivity": float(per_grade[1])}
