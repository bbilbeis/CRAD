import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List

# -----------------------------
# 1) Toy student network
# -----------------------------
class StudentNet(nn.Module):
    def __init__(self, in_dim=784, num_classes=10, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: [B, 1, 28, 28] -> flatten
        h = x.view(x.size(0), -1)
        x = F.relu(self.fc1(h))
        return self.fc2(x),h  # logits

# -----------------------------
# 2) Proxy (public) unlabeled data for server distillation
#    Here we just synthesize random data for demo.
# -----------------------------
def make_proxy_loader(num_samples=2048, batch_size=128, num_classes=10):
    # Fake 28x28 images; in practice, use a real public/unlabeled dataset.
    x = torch.randn(num_samples, 1, 28, 28)
    # No labels needed; placeholder zeros
    y = torch.zeros(num_samples, dtype=torch.long)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True), num_classes

# -----------------------------
# 3) Client: produce logits on proxy data using its local model
#    In a real system, this runs remotely; here we simulate with given "client models".
# -----------------------------
@torch.no_grad()
def client_compute_logits(client_model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    client_model.eval()
    return client_model(inputs)  # logits [B, C]

# -----------------------------
# 4) Server: weighted aggregation of client logits
#    logits_list: List[Tensor [B, C]], weights: Tensor [num_clients] that sums to 1
# -----------------------------
def aggregate_client_logits(logits_list: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    # Stack: [N, B, C] -> weighted sum over N
    stacked = torch.stack(logits_list, dim=0)
    # weights: [N] -> [N, 1, 1] for broadcast
    w = weights.view(-1, 1, 1)
    agg = (w * stacked).sum(dim=0)  # [B, C]
    return agg

# -----------------------------
# 5) KD loss on logits (KLDiv with temperature)
#    student learns from aggregated teacher logits
# -----------------------------
def kd_loss_on_logits(student_logits, teacher_logits, T: float = 2.0):
    # KLDivLoss expects log-prob for input and prob for target
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    p_t    = F.softmax(teacher_logits / T, dim=-1)
    kl = F.kl_div(log_p_s, p_t, reduction='batchmean') * (T * T)
    return kl

# -----------------------------
# 6) End-to-end demo training loop
# -----------------------------
def train_federated_kd(
    student: nn.Module,
    client_models: List[nn.Module],
    client_weights: List[float], ##you can use uncertatiy or anything you think it is good for re-weighting.
    rounds: int = 5,
    lr: float = 1e-3,
    T: float = 2.0,
    device: str = "cpu"
):
    student.to(device)
    for m in client_models:
        m.to(device)

    # normalize weights -> tensor on device
    w = torch.tensor(client_weights, dtype=torch.float32, device=device)
    w = w / w.sum()

    proxy_loader, _ = make_proxy_loader()  # replace with real public data
    opt = torch.optim.Adam(student.parameters(), lr=lr)

    for r in range(1, rounds + 1):
        student.train()
        total_loss = 0.0
        n_batches = 0

        for (x, _) in proxy_loader:
            x = x.to(device)

            # --- Clients compute logits on the SAME proxy batch ---
            client_logits_list = [client_compute_logits(cm, x) for cm in client_models]

            # --- Weighted aggregation at the server ---
            with torch.no_grad():
                agg_teacher_logits = aggregate_client_logits(client_logits_list, w)  # [B, C]

            # --- Student forward & KD loss ---
            opt.zero_grad()
            student_logits = student(x)  # [B, C]
            loss = kd_loss_on_logits(student_logits, agg_teacher_logits, T=T)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"[Round {r}] KD loss: {avg_loss:.4f}")

# -----------------------------
# 7) Example usage
#    We simulate client models as stronger teachers.
#    In practice, each client would have its own trained local model.
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Create a student and a few "client teacher" models (simulated)
    student = StudentNet()

    # Simulate 3 clients with different capacities (as teachers)
    client1 = StudentNet(hidden=512)  # larger hidden -> pretend stronger
    client2 = StudentNet(hidden=384)
    client3 = StudentNet(hidden=256)

    # Optionally, pretend they were trained by nudging weights
    with torch.no_grad():
        for m, scale in [(client1, 1.2), (client2, 1.1), (client3, 1.0)]:
            for p in m.parameters():
                p.add_(0.01 * scale * torch.randn_like(p))

    # Client weights (e.g., proportional to data size, quality, or trust)
    client_weights = [0.5, 0.3, 0.2]  # sum need not be 1; we normalize internally

    # Train server-side KD
    train_federated_kd(
        student=student,
        client_models=[client1, client2, client3],
        client_weights=client_weights,
        rounds=5,
        lr=1e-3,
        T=2.0,
        device="cpu"
    )
