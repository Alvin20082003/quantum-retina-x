import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import numpy as np
from scipy import stats

with open("results_revision/data/phase0_results.json") as f: p0=json.load(f)
with open("results_revision/data/A1_distribution.json") as f: a1=json.load(f)
with open("experiments/results/multiseed_results.json") as f: ms=json.load(f)

va=[r["accuracy"] for r in ms["vit"]]; ha=[r["accuracy"] for r in ms["hybrid"]]
def tci(v): m=np.mean(v);s=np.std(v,ddof=1);se=s/3**0.5;t=stats.t.ppf(0.975,2);return m,s,m-t*se,m+t*se
vm,vs,vl,vu=tci(va); hm,hs,hl,hu=tci(ha)
g1=a1["100_sample_grade_distribution"].get("1",10)
v=p0["vit_seed42"]; h=p0["hyb_seed42"]; vl_=p0["vit_low"]; hl_=p0["hyb_low"]

lines=[
"# Results Summary - Revision",
"",
"## Phase 0: Reproduced Results (seed 42, APTOS 550)",
"| Model | Accuracy | Macro-F1 | AUC | Kappa | G1-Sens |",
"|---|---|---|---|---|---|",
f"| ViT-only | {v['accuracy']:.4f} | {v['macro_f1']:.4f} | {v['auc_roc']:.4f} | {v['kappa']:.4f} | {v['grade1_sensitivity']:.4f} |",
f"| ViT+VQC  | {h['accuracy']:.4f} | {h['macro_f1']:.4f} | {h['auc_roc']:.4f} | {h['kappa']:.4f} | {h['grade1_sensitivity']:.4f} |",
"",
"## Low-Data (100 samples, seed 42)",
"| Model | Acc | Kappa | G1-Sens |",
"|---|---|---|---|",
f"| ViT-only | {vl_['accuracy']:.4f} | {vl_['kappa']:.4f} | {vl_['grade1_sensitivity']:.4f} |",
f"| ViT+VQC  | {hl_['accuracy']:.4f} | {hl_['kappa']:.4f} | {hl_['grade1_sensitivity']:.4f} |",
"",
f"## A1: 100-Sample Grade Distribution",
str(a1["100_sample_grade_distribution"]),
f"G1 count in 100-sample subset: {g1}",
"",
"## A2: 56 True G1 Predicted As",
"ViT-only-low: G0=34, G1=4, G2=8, G3=9, G4=1",
"ViT+VQC-low:  G0=18, G1=0, G2=26, G3=0, G4=12",
"",
"## C1: Trainable Parameters",
"ViT-only=86,393,093 | VQC-only=5,269 | ViT+VQC=3,125 | Classical-A=3,101 | Classical-B=3,117",
"",
"## C2: Classical Baselines (mean+-std, 3 seeds)",
f"Classical-A acc: 0.8121+-0.0058 | Classical-B acc: 0.8158+-0.0110",
f"ViT+VQC mean acc: {hm:.4f}+-{hs:.4f}",
"",
"## C3: Runtime (NVIDIA RTX 3060 Laptop GPU)",
"Head-only ViT: 0.096 ms/img | Head-only ViT+VQC: 16.173 ms/img",
"Full ViT: 19.290 ms/img | Full ViT+VQC: 38.983 ms/img",
"VQC simulator: default.qubit (PennyLane), parameter-shift, CPU",
"",
"## C4: Statistics",
"McNemar (seed42): b=42, c=24, p=0.0364 (significant at 0.05)",
"Bootstrap 95% CI Acc ViT-only: [0.7491, 0.8182]",
"Bootstrap 95% CI Acc ViT+VQC:  [0.7855, 0.8491]",
"Bootstrap 95% CI Kap ViT-only: [0.8494, 0.9095]",
"Bootstrap 95% CI Kap ViT+VQC:  [0.8368, 0.9034]",
f"3-seed t-CI ViT-only acc: {vm:.4f}+-{vs:.4f} [{vl:.4f}, {vu:.4f}]",
f"3-seed t-CI ViT+VQC  acc: {hm:.4f}+-{hs:.4f} [{hl:.4f}, {hu:.4f}]",
"Paired t-test (acc, 3 seeds): t=3.7937, p=0.0630",
"",
"## B: EyePACS Version Confirmed = v1",
"Reproduced: ViT-only 0.6422/0.3520/0.7245/0.4119/0.1810",
"Reproduced: ViT+VQC  0.6304/0.3249/0.6033/0.4123/0.1897",
"",
"## B: Domain Adaptation Delta (accuracy vs zero-shot)",
"IDRiD   ViT-only: ZS=0.4286 AdaBN=+0.0330 CORAL=-0.0857",
"IDRiD   ViT+VQC:  ZS=0.4352 AdaBN=-0.0110 CORAL=-0.2549",
"Messidor ViT-only: ZS=0.5940 AdaBN=-0.0103 CORAL=-0.0109",
"Messidor ViT+VQC:  ZS=0.5831 AdaBN=-0.0419 CORAL=-0.2001",
"EyePACS ViT-only: ZS=0.6422 AdaBN=-0.0520 CORAL=+0.0920",
"EyePACS ViT+VQC:  ZS=0.6304 AdaBN=-0.0410 CORAL=-0.5882",
"",
"## Skipped (time constraint)",
"B1 colour normalisation (Reinhard/Ben Graham/CLAHE): SKIPPED",
"B3 fine-tuning with target labels: SKIPPED",
"A3 class-balancing strategies: SKIPPED",
"See run_log.md for details.",
]
with open("results_revision/reports/results_summary.md","w",encoding="utf-8") as f:
    f.write("\n".join(lines))

kn={
    "reproduction":{"vit_seed42":{k:p0["vit_seed42"][k] for k in ["accuracy","macro_f1","auc_roc","kappa","grade1_sensitivity"]},
                    "hyb_seed42":{k:p0["hyb_seed42"][k] for k in ["accuracy","macro_f1","auc_roc","kappa","grade1_sensitivity"]}},
    "lowdata":{"vit_100":{k:p0["vit_low"][k] for k in ["accuracy","kappa","grade1_sensitivity"]},
               "hyb_100":{k:p0["hyb_low"][k] for k in ["accuracy","kappa","grade1_sensitivity"]}},
    "eyepacs_version":"v1",
    "eyepacs_vit":{"acc":0.6422,"f1":0.3520,"auc":0.7245,"kappa":0.4119,"g1":0.1810},
    "eyepacs_hyb":{"acc":0.6304,"f1":0.3249,"auc":0.6033,"kappa":0.4123,"g1":0.1897},
    "statistics":{"mcnemar_b":42,"mcnemar_c":24,"mcnemar_p":0.0364,
                  "bootstrap_ci_acc_vit":[0.7491,0.8182],"bootstrap_ci_acc_hyb":[0.7855,0.8491],
                  "bootstrap_ci_kap_vit":[0.8494,0.9095],"bootstrap_ci_kap_hyb":[0.8368,0.9034],
                  "ttest_t":3.7937,"ttest_p":0.0630},
    "params":{"vit":86393093,"vqc_only":5269,"hyb_trainable":3125,"classA":3101,"classB":3117},
    "runtime_ms":{"head_vit":0.096,"head_hyb":16.173,"full_vit":19.290,"full_hyb":38.983},
    "a1_g1_in_100":g1,
    "a2_g1_preds":{"vit_low":{"G0":34,"G1":4,"G2":8,"G3":9,"G4":1},
                   "hyb_low":{"G0":18,"G1":0,"G2":26,"G3":0,"G4":12}},
}
with open("results_revision/data/key_numbers.json","w") as f: json.dump(kn,f,indent=2)
print("All files saved.")
