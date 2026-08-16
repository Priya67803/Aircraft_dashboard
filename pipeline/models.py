"""
models.py
=========
All architectural building blocks of the predictive-maintenance
framework, consolidated: positional encoding, sensor Graph Attention
Network, HI cross-attention, four autoencoders (MC-Dropout capable),
adaptive attention fusion, dynamic health index, the shared temporal
encoder, task heads, adaptive gating ensemble, physics-informed +
multi-task losses, self-supervised pretraining, EVT/POT adaptive
thresholding, and Integrated-Gradients explainability.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import genpareto


# ------------------------------------------------------------------ #
# Positional encoding
# ------------------------------------------------------------------ #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ------------------------------------------------------------------ #
# Sensor Graph Attention Network
# ------------------------------------------------------------------ #
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.2):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, n_heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(n_heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(n_heads, out_dim))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_bias=None):
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.n_heads, self.out_dim).permute(0, 2, 1, 3)
        src = (h * self.a_src.view(1, self.n_heads, 1, self.out_dim)).sum(-1)
        dst = (h * self.a_dst.view(1, self.n_heads, 1, self.out_dim)).sum(-1)
        e = self.leaky_relu(src.unsqueeze(-1) + dst.unsqueeze(-2))
        if adj_bias is not None:
            e = e + adj_bias.unsqueeze(1)
        alpha = self.dropout(F.softmax(e, dim=-1))
        out = torch.matmul(alpha, h).permute(0, 2, 1, 3).reshape(B, N, self.n_heads * self.out_dim)
        return out, alpha


def correlation_adjacency(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xc = x - x.mean(dim=1, keepdim=True)
    std = xc.std(dim=1, keepdim=True) + eps
    xn = xc / std
    corr = torch.einsum("bln,blm->bnm", xn, xn) / x.size(1)
    return torch.clamp(corr, -1.0, 1.0) * 2.0


class HICrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.2):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, hi_query, sensor_kv):
        attn_out, attn_weights = self.mha(hi_query, sensor_kv, sensor_kv, need_weights=True)
        return self.norm(hi_query + attn_out), attn_weights


# ------------------------------------------------------------------ #
# Autoencoders (MC-Dropout capable)
# ------------------------------------------------------------------ #
class _MCDropoutMixin:
    def mc_predict(self, x, n_samples=20):
        was_training = self.training
        self.train()
        errs = []
        with torch.no_grad():
            for _ in range(n_samples):
                recon = self.forward(x)
                err = torch.mean(torch.abs(recon - x), dim=tuple(range(1, x.dim())))
                errs.append(err)
        if not was_training:
            self.eval()
        errs = torch.stack(errs, dim=0)
        mean, var, std = errs.mean(0), errs.var(0, unbiased=False), errs.std(0, unbiased=False)
        return {"mean": mean, "var": var, "std": std,
                "ci_low": mean - 1.96 * std, "ci_high": mean + 1.96 * std, "samples": errs}


class DenseAE(nn.Module, _MCDropoutMixin):
    def __init__(self, seq_len, n_feat, dropout=0.2):
        super().__init__()
        in_dim = seq_len * n_feat
        self.seq_len, self.n_feat = seq_len, n_feat
        self.enc = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(32, 16), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(32, 64), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(64, in_dim))

    def forward(self, x):
        b = x.size(0)
        z = self.enc(x.reshape(b, -1))
        return self.dec(z).reshape(b, self.seq_len, self.n_feat)


class CNNAE(nn.Module, _MCDropoutMixin):
    def __init__(self, n_feat, dropout=0.2):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(n_feat, 32, 3, padding=1), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Conv1d(32, 16, 3, padding=1), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Conv1d(16, 8, 3, padding=1), nn.ReLU())
        self.dec = nn.Sequential(nn.Conv1d(8, 16, 3, padding=1), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Conv1d(32, n_feat, 1))

    def forward(self, x):
        h = self.enc(x.transpose(1, 2))
        return self.dec(h).transpose(1, 2)


class LSTMAE(nn.Module, _MCDropoutMixin):
    def __init__(self, seq_len, n_feat, latent=32, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.encoder_lstm = nn.LSTM(n_feat, 32, batch_first=True)
        self.enc_dropout = nn.Dropout(dropout)
        self.bottleneck = nn.Linear(32, latent)
        self.decoder_lstm = nn.LSTM(latent, 32, batch_first=True)
        self.dec_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(32, n_feat)

    def forward(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h = self.enc_dropout(h_n[-1])
        z = torch.relu(self.bottleneck(h))
        z_rep = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.decoder_lstm(z_rep)
        return self.out_proj(self.dec_dropout(dec_out))


class TransformerAE(nn.Module, _MCDropoutMixin):
    def __init__(self, n_feat, d_model=64, n_heads=4, ff_dim=128, n_layers=2, dropout=0.2):
        super().__init__()
        self.in_proj = nn.Linear(n_feat, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
                                                dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, n_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.pos_enc(self.in_proj(x))
        h = self.encoder(h)
        return self.out_proj(self.dropout(h))


# ------------------------------------------------------------------ #
# Fusion + Dynamic Health Index
# ------------------------------------------------------------------ #
class AttentionFusion(nn.Module):
    def __init__(self, condition_dim, n_models=4, hidden=32):
        super().__init__()
        self.score_net = nn.Sequential(nn.Linear(condition_dim + n_models, hidden), nn.ReLU(),
                                        nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_models))

    def forward(self, errors, condition):
        logits = self.score_net(torch.cat([condition, errors], dim=-1))
        alpha = torch.softmax(logits, dim=-1)
        return torch.sum(alpha * errors, dim=-1), alpha


class DynamicHealthIndex(nn.Module):
    def __init__(self, n_feat, hidden=32, init_beta_logit=0.0):
        super().__init__()
        self.neural_encoder = nn.Sequential(nn.Linear(n_feat, hidden), nn.ReLU(),
                                             nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.beta_logit = nn.Parameter(torch.tensor(float(init_beta_logit)))

    @property
    def beta(self):
        return torch.sigmoid(self.beta_logit)

    def forward_sequence(self, x_seq, pca_hi_seq):
        B, L, F_ = x_seq.shape
        neural_hi = self.neural_encoder(x_seq.reshape(B * L, F_)).reshape(B, L)
        beta = self.beta
        return beta * pca_hi_seq + (1.0 - beta) * neural_hi


# ------------------------------------------------------------------ #
# Shared temporal encoder
# ------------------------------------------------------------------ #
class SharedTemporalEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_feat, d_model = cfg.NUM_FEATURES, cfg.D_MODEL
        gat_dim = cfg.GAT_HEADS * cfg.GAT_OUT_DIM
        self.node_embed = nn.Linear(1, 16)
        self.gat = GraphAttentionLayer(16, cfg.GAT_OUT_DIM, cfg.GAT_HEADS, cfg.DROPOUT)
        self.gat_to_dmodel = nn.Linear(gat_dim, d_model)
        self._raw_sensor_proj = nn.Linear(n_feat, d_model)
        self.dynamic_hi = DynamicHealthIndex(n_feat=n_feat)
        self.hi_proj = nn.Linear(1, d_model)
        self.cross_attn = HICrossAttention(d_model, cfg.N_HEADS, cfg.DROPOUT)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=cfg.N_HEADS, dim_feedforward=cfg.FF_DIM,
                                                dropout=cfg.DROPOUT, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.N_TRANSFORMER_LAYERS)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x_seq, pca_hi_seq, use_gat=True, use_cross_attention=True, use_dynamic_hi=True):
        B, L, F_ = x_seq.shape
        if use_gat:
            nodes = self.node_embed(x_seq.reshape(B * L, F_, 1))
            adj_bias = correlation_adjacency(x_seq).unsqueeze(1).expand(-1, L, -1, -1).reshape(B * L, F_, F_)
            gat_out, gat_attn = self.gat(nodes, adj_bias=adj_bias)
            gat_pooled = gat_out.mean(dim=1).reshape(B, L, -1)
            sensor_kv = self.gat_to_dmodel(gat_pooled)
        else:
            gat_attn = torch.zeros(B * L, self.gat.n_heads, F_, F_, device=x_seq.device)
            sensor_kv = self._raw_sensor_proj(x_seq)

        if use_dynamic_hi:
            dyn_hi_seq = self.dynamic_hi.forward_sequence(x_seq, pca_hi_seq)
        else:
            dyn_hi_seq = pca_hi_seq
        hi_query = self.hi_proj(dyn_hi_seq.unsqueeze(-1))

        if use_cross_attention:
            fused, cross_attn_w = self.cross_attn(hi_query, sensor_kv)
        else:
            fused = sensor_kv
            cross_attn_w = torch.zeros(B, L, L, device=x_seq.device)

        h = self.out_norm(self.transformer(self.pos_enc(fused)))
        pooled = h.mean(dim=1)
        return {"repr": pooled, "seq_repr": h,
                "gat_attn": gat_attn.reshape(B, L, gat_attn.size(1), F_, F_),
                "cross_attn_w": cross_attn_w, "dynamic_hi_seq": dyn_hi_seq}


# ------------------------------------------------------------------ #
# Task heads + gating ensemble
# ------------------------------------------------------------------ #
def _mlp_head(in_dim, out_dim, hidden=48, dropout=0.2):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                          nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, out_dim))


class AnomalyHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = _mlp_head(d_model, 1)

    def forward(self, r):
        return self.net(r).squeeze(-1)


class FaultClassificationHead(nn.Module):
    def __init__(self, d_model, n_classes=3):
        super().__init__()
        self.net = _mlp_head(d_model, n_classes)

    def forward(self, r):
        return self.net(r)


class RULRegressionHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = _mlp_head(d_model, 1)

    def forward(self, r):
        return torch.relu(self.net(r).squeeze(-1))


class AdaptiveGatingEnsemble(nn.Module):
    def __init__(self, condition_dim, n_models, hidden=32):
        super().__init__()
        self.gate_net = nn.Sequential(nn.Linear(condition_dim, hidden), nn.ReLU(),
                                       nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_models))

    def forward(self, predictions, condition):
        gate_weights = torch.softmax(self.gate_net(condition), dim=-1)
        return torch.sum(gate_weights * predictions, dim=-1), gate_weights


# ------------------------------------------------------------------ #
# Losses
# ------------------------------------------------------------------ #
class PhysicsInformedLoss(nn.Module):
    def rul_monotonicity(self, rul_t, rul_t1):
        return F.relu(rul_t1 - rul_t).mean()

    def hi_monotonicity(self, hi_seq):
        return F.relu(hi_seq[:, 1:] - hi_seq[:, :-1]).mean()

    def forward(self, rul_pred_t, rul_pred_t1, hi_seq):
        l_rul = self.rul_monotonicity(rul_pred_t, rul_pred_t1)
        l_hi = self.hi_monotonicity(hi_seq)
        return l_rul + l_hi, {"rul_monotonicity": l_rul.item(), "hi_monotonicity": l_hi.item()}


class MultiTaskLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.la, self.lf, self.lr, self.lp = cfg.LAMBDA_ANOMALY, cfg.LAMBDA_FAULT, cfg.LAMBDA_RUL, cfg.LAMBDA_PHYSICS
        self.physics = PhysicsInformedLoss()
        self.bce, self.ce, self.mse = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(), nn.MSELoss()

    def forward(self, outputs, targets):
        l_anom = self.bce(outputs["anomaly_logit"], targets["y_anomaly"])
        l_fault = self.ce(outputs["fault_logits"], targets["y_fault"])
        l_rul = self.mse(outputs["rul_pred"], targets["y_rul"])
        l_phys, phys_log = self.physics(outputs["rul_pred"], outputs["rul_pred_next"], outputs["dynamic_hi_seq"])
        total = self.la * l_anom + self.lf * l_fault + self.lr * l_rul + self.lp * l_phys
        logs = {"anomaly": l_anom.item(), "fault": l_fault.item(), "rul": l_rul.item(),
                "physics": l_phys.item(), **phys_log, "total": total.item()}
        return total, logs


# ------------------------------------------------------------------ #
# Self-supervised pretraining
# ------------------------------------------------------------------ #
def random_mask(x, mask_ratio=0.15):
    mask = (torch.rand_like(x) < mask_ratio)
    return x.masked_fill(mask, 0.0), mask


def jitter(x, sigma=0.05):
    return x + torch.randn_like(x) * sigma


def time_mask(x, max_span=4):
    B, L, F_ = x.shape
    out = x.clone()
    for b in range(B):
        span = torch.randint(1, max_span + 1, (1,)).item()
        start = torch.randint(0, max(1, L - span), (1,)).item()
        out[b, start:start + span, :] = 0.0
    return out


class MaskedReconstructionHead(nn.Module):
    def __init__(self, d_model, n_feat):
        super().__init__()
        self.proj = nn.Linear(d_model, n_feat)

    def forward(self, seq_repr):
        return self.proj(seq_repr)


class ContrastiveProjectionHead(nn.Module):
    def __init__(self, d_model, proj_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, proj_dim))

    def forward(self, r):
        return F.normalize(self.net(r), dim=-1)


class TemporalOrderHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, r):
        return self.net(r).squeeze(-1)


def nt_xent_loss(z1, z2, temperature=0.5):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t()) / temperature
    sim.fill_diagonal_(-1e9)
    targets = torch.arange(B, device=z.device)
    targets = torch.cat([targets + B, targets], dim=0)
    return F.cross_entropy(sim, targets)


def make_swapped_halves(x):
    B, L, F_ = x.shape
    mid = L // 2
    swap = (torch.rand(B, device=x.device) < 0.5)
    out = x.clone()
    first, second = x[:, :mid, :], x[:, mid:2 * mid, :]
    swapped_seq = torch.cat([second, first], dim=1)
    if L % 2 == 1:
        swapped_seq = torch.cat([swapped_seq, x[:, 2 * mid:, :]], dim=1)
    out[swap] = swapped_seq[swap]
    return out, swap.float()


class SelfSupervisedPretrainer:
    def __init__(self, encoder, n_feat, d_model, device, mask_ratio=0.15, lr=1e-3):
        self.encoder, self.device, self.mask_ratio = encoder, device, mask_ratio
        self.recon_head = MaskedReconstructionHead(d_model, n_feat).to(device)
        self.contrastive_head = ContrastiveProjectionHead(d_model).to(device)
        self.order_head = TemporalOrderHead(d_model).to(device)
        params = (list(encoder.parameters()) + list(self.recon_head.parameters())
                  + list(self.contrastive_head.parameters()) + list(self.order_head.parameters()))
        self.opt = torch.optim.Adam(params, lr=lr)

    def step(self, x_seq, pca_hi_seq):
        self.encoder.train()
        self.opt.zero_grad()
        x_masked, mask = random_mask(x_seq, self.mask_ratio)
        out_masked = self.encoder(x_masked, pca_hi_seq)
        recon = self.recon_head(out_masked["seq_repr"])
        l_recon = F.mse_loss(recon[mask], x_seq[mask]) if mask.any() else recon.sum() * 0.0

        view1, view2 = jitter(time_mask(x_seq)), jitter(time_mask(x_seq))
        z1 = self.contrastive_head(self.encoder(view1, pca_hi_seq)["repr"])
        z2 = self.contrastive_head(self.encoder(view2, pca_hi_seq)["repr"])
        l_contrast = nt_xent_loss(z1, z2)

        x_ordered, order_label = make_swapped_halves(x_seq)
        order_logit = self.order_head(self.encoder(x_ordered, pca_hi_seq)["repr"])
        l_order = F.binary_cross_entropy_with_logits(order_logit, order_label)

        loss = l_recon + l_contrast + l_order
        loss.backward()
        self.opt.step()
        return {"recon": l_recon.item(), "contrastive": l_contrast.item(),
                "order": l_order.item(), "total": loss.item()}


# ------------------------------------------------------------------ #
# EVT / POT adaptive threshold
# ------------------------------------------------------------------ #
class AdaptiveThreshold:
    def __init__(self, tail_quantile=0.90, target_tail_prob=0.01):
        self.tail_quantile, self.target_tail_prob = tail_quantile, target_tail_prob
        self.threshold_, self.method_, self.gpd_params_ = None, None, None

    def fit(self, healthy_scores):
        healthy_scores = np.asarray(healthy_scores, dtype=np.float64)
        u = np.quantile(healthy_scores, self.tail_quantile)
        exceedances = healthy_scores[healthy_scores > u] - u
        if len(exceedances) >= 15:
            try:
                shape, loc, scale = genpareto.fit(exceedances, floc=0)
                p_exceed = len(exceedances) / len(healthy_scores)
                ratio = self.target_tail_prob / max(p_exceed, 1e-6)
                z = (scale / shape) * (ratio ** (-shape) - 1.0) if abs(shape) > 1e-6 else -scale * np.log(ratio)
                self.threshold_ = float(u + z)
                self.method_ = "EVT-POT (GPD)"
                self.gpd_params_ = {"shape": shape, "loc": loc, "scale": scale, "u": float(u)}
                return self
            except Exception:
                pass
        self.threshold_ = float(np.quantile(healthy_scores, 1.0 - self.target_tail_prob))
        self.method_ = "percentile (fallback)"
        return self

    def predict(self, scores):
        return (np.asarray(scores) > self.threshold_).astype(int)

    def summary(self):
        return {"threshold": self.threshold_, "method": self.method_, "gpd_params": self.gpd_params_}


# ------------------------------------------------------------------ #
# Explainability: Integrated Gradients
# ------------------------------------------------------------------ #
class IntegratedGradients:
    def __init__(self, model_fn, baseline=None, steps=32):
        self.model_fn, self.baseline, self.steps = model_fn, baseline, steps

    def attribute(self, x):
        baseline = self.baseline if self.baseline is not None else torch.zeros_like(x)
        alphas = torch.linspace(0, 1, self.steps, device=x.device).view(-1, 1, 1, 1)
        interpolated = baseline.unsqueeze(0) + alphas * (x.unsqueeze(0) - baseline.unsqueeze(0))
        interpolated = interpolated.reshape(-1, x.size(1), x.size(2)).clone().requires_grad_(True)
        outputs = self.model_fn(interpolated)
        grads = torch.autograd.grad(outputs.sum(), interpolated, create_graph=False)[0]
        grads = grads.reshape(self.steps, x.size(0), x.size(1), x.size(2))
        avg_grads = grads.mean(dim=0)
        return ((x - baseline) * avg_grads).detach()


def feature_importance(attributions, feature_names):
    scores = attributions.abs().mean(dim=(0, 1)).cpu().numpy()
    return dict(sorted(zip(feature_names, scores.tolist()), key=lambda kv: -kv[1]))


def temporal_importance(attributions):
    return attributions.abs().mean(dim=(0, 2)).cpu().numpy()


# ------------------------------------------------------------------ #
# Full framework
# ------------------------------------------------------------------ #
def _recon_error(recon, x):
    return torch.mean(torch.abs(recon - x), dim=(1, 2))


def _norm01(x, eps=1e-8):
    return (x - x.min()) / (x.max() - x.min() + eps)


class PredictiveMaintenanceFramework(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.seq_len, self.n_feat = cfg.SEQ_LEN, cfg.NUM_FEATURES
        self.encoder = SharedTemporalEncoder(cfg)
        self.lstm_ae = LSTMAE(cfg.SEQ_LEN, cfg.NUM_FEATURES, latent=cfg.LATENT_DIM, dropout=cfg.DROPOUT)
        self.dense_ae = DenseAE(cfg.SEQ_LEN, cfg.NUM_FEATURES, dropout=cfg.DROPOUT)
        self.transformer_ae = TransformerAE(cfg.NUM_FEATURES, d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS,
                                             ff_dim=cfg.FF_DIM, dropout=cfg.DROPOUT)
        self.cnn_ae = CNNAE(cfg.NUM_FEATURES, dropout=cfg.DROPOUT)
        self.fusion = AttentionFusion(condition_dim=cfg.D_MODEL, n_models=4)
        self.anomaly_head = AnomalyHead(cfg.D_MODEL)
        self.fault_head = FaultClassificationHead(cfg.D_MODEL)
        self.rul_head = RULRegressionHead(cfg.D_MODEL)

    def autoencoder_errors(self, x_seq):
        le = _recon_error(self.lstm_ae(x_seq), x_seq)
        de = _recon_error(self.dense_ae(x_seq), x_seq)
        te = _recon_error(self.transformer_ae(x_seq), x_seq)
        ce = _recon_error(self.cnn_ae(x_seq), x_seq)
        return torch.stack([_norm01(le), _norm01(de), _norm01(te), _norm01(ce)], dim=-1)

    def mc_autoencoder_uncertainty(self, x_seq, n_samples=20):
        return {
            "LSTM AE": self.lstm_ae.mc_predict(x_seq, n_samples),
            "Dense AE": self.dense_ae.mc_predict(x_seq, n_samples),
            "Transformer AE": self.transformer_ae.mc_predict(x_seq, n_samples),
            "CNN AE": self.cnn_ae.mc_predict(x_seq, n_samples),
        }

    def forward(self, x_seq, pca_hi_seq, use_gat=True, use_cross_attention=True,
                use_dynamic_hi=True, use_fusion_attention=True):
        enc_out = self.encoder(x_seq, pca_hi_seq, use_gat, use_cross_attention, use_dynamic_hi)
        repr_ = enc_out["repr"]
        errs = self.autoencoder_errors(x_seq)
        if use_fusion_attention:
            fused_score, alpha = self.fusion(errs, repr_)
        else:
            fused_score, alpha = errs.mean(dim=-1), torch.full_like(errs, 1.0 / errs.size(-1))

        anomaly_logit = self.anomaly_head(repr_)
        fault_logits = self.fault_head(repr_)
        rul_pred = self.rul_head(repr_)

        rolled_hi = enc_out["dynamic_hi_seq"].roll(shifts=-1, dims=1)
        rolled_hi[:, -1] = enc_out["dynamic_hi_seq"][:, -1]
        enc_out_next = self.encoder(x_seq, rolled_hi, use_gat, use_cross_attention, use_dynamic_hi)
        rul_pred_next = self.rul_head(enc_out_next["repr"])

        return {"repr": repr_, "ae_errors": errs, "fused_anomaly_score": fused_score, "fusion_alpha": alpha,
                "anomaly_logit": anomaly_logit, "fault_logits": fault_logits, "rul_pred": rul_pred,
                "rul_pred_next": rul_pred_next, "dynamic_hi_seq": enc_out["dynamic_hi_seq"],
                "encoder_output": enc_out}
