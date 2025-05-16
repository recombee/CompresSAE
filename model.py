import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True)


class CompresSAE(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, k: int):
        super().__init__()
        self.k = k
        self.encoder_w = nn.Parameter(nn.init.kaiming_uniform_(torch.empty([input_dim, embedding_dim])))
        self.encoder_b = nn.Parameter(torch.zeros(embedding_dim))
        self.decoder_w = nn.Parameter(nn.init.kaiming_uniform_(torch.empty([embedding_dim, input_dim])))
        self.normalize_decoder()

    def encode(self, x: torch.Tensor, apply_activation: bool = True) -> torch.Tensor:
        e_pre = l2_normalize(x) @ self.encoder_w + self.encoder_b
        return self.topk_mask(e_pre, self.k) if apply_activation else e_pre

    def decode(self, e: torch.Tensor) -> torch.Tensor:
        return e @ self.decoder_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder_w.data = l2_normalize(self.decoder_w.data)
        if self.decoder_w.grad is not None:
            self.decoder_w.grad -= (self.decoder_w.grad * self.decoder_w.data).sum(-1, keepdim=True) * self.decoder_w.data

    @staticmethod
    def topk_mask(e: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
        e_topk = torch.topk(torch.abs(e), k, dim)
        return torch.zeros_like(e).scatter(dim, e_topk.indices, e_topk.values) * torch.sign(e)

    def compute_loss_dict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e_pre = self.encode(x, apply_activation=False)
        e, e_4k = self.topk_mask(e_pre, self.k), self.topk_mask(e_pre, 4 * self.k)
        x_out, x_out_4k = self.decode(e), self.decode(e_4k)
        losses = {
            "L2": (x_out - x).pow(2).mean(),
            "L2_4k": (x_out_4k - x).pow(2).mean(),
            "L1": e.abs().sum(-1).mean(),
            "L0": (e != 0).float().sum(-1).mean(),
            "Cosine": (1 - F.cosine_similarity(x, x_out, 1)).mean(),
            "Cosine_4k": (1 - F.cosine_similarity(x, x_out_4k, 1)).mean(),
        }
        losses["Loss"] = losses["Cosine"] + losses["Cosine_4k"] / 8
        return losses

    def train_step(self, optimizer: optim.Optimizer, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        losses = self.compute_loss_dict(batch)
        optimizer.zero_grad()
        losses["Loss"].backward()
        self.normalize_decoder()
        optimizer.step()
        return losses
