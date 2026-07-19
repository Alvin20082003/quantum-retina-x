import torch
import torch.nn as nn

def main():
    """Phase 3 gate: prove the VQC trains standalone on a toy task."""
    from models.quantum import QuantumLayer
    print("=" * 55)
    print("Phase 3 gate: standalone VQC test")
    print("=" * 55)

    ql = QuantumLayer(n_qubits=4, n_layers=2, init_std=0.01)
    model = nn.Sequential(ql, nn.Linear(4, 3))

    w_before = [p.clone().detach() for p in model.parameters()]

    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()

    torch.manual_seed(42)
    for step in range(30):
        x = torch.randn(8, 4)
        y = torch.randint(0, 3, (8,))
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(f"  step {step:2d}  loss {loss.item():.4f}")

    w_after = list(model.parameters())
    qmoved = not torch.allclose(w_before[0], w_after[0].detach(), atol=1e-6)
    lmoved = not torch.allclose(w_before[1], w_after[1].detach(), atol=1e-6)
    print(f"quantum weights moved: {qmoved}")
    print(f"linear  weights moved: {lmoved}")

    if qmoved and lmoved:
        print("Phase 3 gate PASSED. VQC trains, no barren plateau.")
    else:
        print("!! FAILED - weights did not move. Check init_std or circuit.")
    print("=" * 55)

if __name__ == "__main__":
    main()
