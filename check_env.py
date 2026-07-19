import sys
from utils.config import load_config
from utils.seed import set_seed
from utils.logger import ExperimentLogger

def main():
    print("=" * 55)
    cfg = load_config("config.yaml")
    print(f"[ok] config loaded  seed={cfg['seed']}  n_qubits={cfg['quantum']['n_qubits']}")
    set_seed(cfg["seed"])
    print(f"[ok] seeds set to {cfg['seed']}")
    print(f"[info] python {sys.version.split()[0]}")
    import torch
    print(f"[info] torch  {torch.__version__}")
    print(f"[ok] GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available()
          else "[warn] CUDA NOT available - CPU only")
    ExperimentLogger(cfg["paths"]["results_dir"]).log(
        run_name="phase0_smoke_test", notes="scaffolding works")
    print("Phase 0 gate passed. Ready for Phase 1.")
    print("=" * 55)

if __name__ == "__main__":
    main()
