"""
runner.py
=========
Orchestrates the end-to-end run and is the only module the Flask app
talks to. `run_pipeline(mode, progress_cb)` executes every stage of the
reference framework (data -> classical baselines -> self-supervised
pretraining -> multi-task fine-tuning with physics loss -> MC-Dropout +
EVT threshold calibration -> adaptive gating ensemble -> evaluation ->
ablation -> explainability -> figures) and streams progress through
`progress_cb(event: dict)`. Figures are rendered to base64 PNGs (no
files written) so they can be dropped straight into the dashboard.
"""
import io
import base64
import traceback
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, chi2
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, mean_squared_error, roc_curve

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from .config import Config, set_seed
from . import data as data_mod
from .models import (
    PredictiveMaintenanceFramework, MultiTaskLoss, SelfSupervisedPretrainer,
    AdaptiveThreshold, AdaptiveGatingEnsemble, IntegratedGradients,
    feature_importance, temporal_importance,
)

STAGES = [
    "Preparing data",
    "Classical baselines (GBR / XGBoost)",
    "Self-supervised pretraining",
    "Multi-task fine-tuning (physics-informed loss)",
    "MC-Dropout + EVT threshold calibration",
    "Adaptive gating ensemble",
    "Holdout evaluation",
    "Statistical significance tests",
    "Architectural ablation study",
    "Explainability (Integrated Gradients)",
    "Rendering figures",
]


def _fig_to_b64():
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=130, facecolor="#0e1526")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _subset(seqs, ix):
    return {k: (v[ix] if isinstance(v, np.ndarray) else v) for k, v in seqs.items()}


def _make_loader(seqs, batch_size, shuffle=True):
    from torch.utils.data import TensorDataset, DataLoader
    X = torch.tensor(seqs["X"])
    HI = torch.tensor(seqs["hi"])
    y_bin = torch.tensor((seqs["y_cls"] != "healthy").astype(np.float32))
    y_cls = torch.tensor(seqs["y_cls_idx"])
    y_rul = torch.tensor(seqs["y_rul"])
    y_rul_next = torch.tensor(seqs["y_rul_next"])
    ds = TensorDataset(X, HI, y_bin, y_cls, y_rul, y_rul_next)
    n = len(ds)
    bs = min(batch_size, max(1, n))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=(shuffle and n > bs))


def _batched_forward(model, X, HI, batch_size, device, **fwd_kwargs):
    model.eval()
    n = len(X)
    collected = {}
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = torch.tensor(X[start:end]).to(device)
            hib = torch.tensor(HI[start:end]).to(device)
            ob = model(xb, hib, **fwd_kwargs)
            for k, v in ob.items():
                if isinstance(v, torch.Tensor):
                    collected.setdefault(k, []).append(v.detach().cpu())
            del xb, hib, ob
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return {k: torch.cat(v_list, dim=0) for k, v_list in collected.items()}


def _mcnemar_test(y_true, pred_a, pred_b):
    correct_a, correct_b = (pred_a == y_true), (pred_b == y_true)
    n01 = int(np.sum(correct_a & ~correct_b))
    n10 = int(np.sum(~correct_a & correct_b))
    table = np.array([[int(np.sum(correct_a & correct_b)), n01], [n10, int(np.sum(~correct_a & ~correct_b))]])
    if n01 + n10 == 0:
        return 0.0, 1.0, table
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p = 1 - chi2.cdf(stat, df=1)
    return float(stat), float(p), table


def _expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


ABLATION_COMPONENTS = ["use_gat", "use_cross_attention", "use_dynamic_hi", "use_fusion_attention"]
FRIENDLY_NAMES = {
    "use_gat": "Sensor Graph Attention (GAT)",
    "use_cross_attention": "HI Cross-Attention",
    "use_dynamic_hi": "Dynamic Health Index",
    "use_fusion_attention": "Attention-Based AE Fusion",
}


def run_pipeline(mode="quick", progress_cb=None, seed=None):
    """Runs every stage. progress_cb(dict) is called with
    {"type": "stage", "index", "total", "name"} and
    {"type": "log", "text"} events. Returns the final results dict."""

    def emit(event):
        if progress_cb:
            progress_cb(event)

    def log(text):
        emit({"type": "log", "text": str(text)})

    def stage(i):
        emit({"type": "stage", "index": i, "total": len(STAGES), "name": STAGES[i]})
        log(f"--- Stage {i + 1}/{len(STAGES)}: {STAGES[i]} ---")

    cfg = Config(mode=mode)
    if seed is not None:
        cfg.SEED = int(seed)
    set_seed(cfg.SEED)
    device = cfg.DEVICE
    log(f"Mode: {mode}  |  Device: {device}")

    # ---------------- Stage 0: data ----------------
    stage(0)
    seqs, data_source = data_mod.prepare_fd001(cfg, log=log)
    log(f"Data source: {data_source}")
    log(f"Total sequences: {len(seqs['X'])}  |  seq_len={cfg.SEQ_LEN}  |  features={cfg.NUM_FEATURES}")

    n = len(seqs["X"])
    rng = np.random.default_rng(cfg.SEED)
    idx = rng.permutation(n)
    tr_idx, val_idx, ho_idx = np.split(idx, [int(0.7 * n), int(0.85 * n)])
    src_train, src_val, src_holdout = _subset(seqs, tr_idx), _subset(seqs, val_idx), _subset(seqs, ho_idx)
    log(f"Split -> train: {len(tr_idx)}, val: {len(val_idx)}, holdout: {len(ho_idx)}")

    class_counts = pd.Series(seqs["y_cls"]).value_counts().to_dict()

    # ---------------- Stage 1: classical baselines ----------------
    stage(1)
    X_last = src_train["X"][:, -1, :]
    scaler = StandardScaler().fit(X_last)
    X_s = scaler.transform(X_last)
    gbr = GradientBoostingRegressor(n_estimators=120, max_depth=4, random_state=cfg.SEED)
    gbr.fit(X_s, src_train["y_rul"])
    if HAS_XGB:
        xgb = XGBRegressor(n_estimators=120, max_depth=4, random_state=cfg.SEED, verbosity=0)
        xgb.fit(X_s, src_train["y_rul"])
        log("Trained GradientBoostingRegressor and XGBRegressor baselines.")
    else:
        xgb = None
        log("xgboost not installed - using GBR as stand-in for both classical baselines.")
    classical = {"gbr": gbr, "xgb": xgb, "scaler": scaler}

    # ---------------- Stage 2: self-supervised pretraining ----------------
    stage(2)
    model = PredictiveMaintenanceFramework(cfg).to(device)
    pretrainer = SelfSupervisedPretrainer(model.encoder, n_feat=model.n_feat, d_model=cfg.D_MODEL, device=device, lr=cfg.LR)
    loader = _make_loader(src_train, cfg.BATCH_SIZE)
    pretrain_history = []
    for epoch in range(cfg.PRETRAIN_EPOCHS):
        epoch_logs = []
        for X, HI, *_ in loader:
            logs = pretrainer.step(X.to(device), HI.to(device))
            epoch_logs.append(logs)
        avg = {k: float(np.mean([l[k] for l in epoch_logs])) for k in epoch_logs[0]}
        pretrain_history.append(avg)
        log(f"[pretrain] epoch {epoch + 1}/{cfg.PRETRAIN_EPOCHS}  "
            f"recon={avg['recon']:.4f}  contrastive={avg['contrastive']:.4f}  "
            f"order={avg['order']:.4f}  total={avg['total']:.4f}")

    # ---------------- Stage 3: multi-task fine-tuning ----------------
    stage(3)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    criterion = MultiTaskLoss(cfg)
    source_loader = _make_loader(src_train, cfg.BATCH_SIZE)
    finetune_history = []
    for epoch in range(cfg.FINETUNE_EPOCHS):
        model.train()
        epoch_logs = []
        for X, HI, y_bin, y_cls, y_rul, y_rul_next in source_loader:
            X, HI = X.to(device), HI.to(device)
            y_bin, y_cls, y_rul = y_bin.to(device), y_cls.to(device), y_rul.to(device)
            outputs = model(X, HI)
            targets = {"y_anomaly": y_bin, "y_fault": y_cls, "y_rul": y_rul}
            loss, logs = criterion(outputs, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_logs.append(logs)
        avg = {k: float(np.mean([l[k] for l in epoch_logs if k in l])) for k in epoch_logs[0]}
        finetune_history.append(avg)
        log(f"[finetune] epoch {epoch + 1}/{cfg.FINETUNE_EPOCHS}  total={avg['total']:.4f}  "
            f"anomaly={avg['anomaly']:.4f}  fault={avg['fault']:.4f}  rul={avg['rul']:.4f}  physics={avg['physics']:.4f}")

    # ---------------- Stage 4: MC-Dropout + EVT threshold ----------------
    stage(4)
    outputs_val = _batched_forward(model, src_val["X"], src_val["hi"], cfg.EVAL_BATCH_SIZE, device)
    fused_scores_val = outputs_val["fused_anomaly_score"].numpy()
    healthy_mask = (src_val["y_cls"] == "healthy")
    threshold = AdaptiveThreshold(cfg.EVT_TAIL_QUANTILE)
    if healthy_mask.sum() >= 5:
        threshold.fit(fused_scores_val[healthy_mask])
    else:
        threshold.fit(fused_scores_val)
    log(f"Adaptive threshold: {threshold.summary()}")

    n_mc = min(64, len(src_val["X"]))
    X_mc = torch.tensor(src_val["X"][:n_mc]).to(device)
    mc_out = model.mc_autoencoder_uncertainty(X_mc, n_samples=cfg.MC_SAMPLES)
    uncertainty_summary = {name: float(r["std"].mean().cpu()) for name, r in mc_out.items()}
    log(f"MC-Dropout mean uncertainty per autoencoder: {uncertainty_summary}")

    # ---------------- Stage 5: adaptive gating ensemble ----------------
    stage(5)
    deep_rul_val = outputs_val["rul_pred"].to(device)
    condition_val = outputs_val["repr"].to(device)
    y_rul_val = torch.tensor(src_val["y_rul"], device=device)
    X_last_val_s = classical["scaler"].transform(src_val["X"][:, -1, :])
    gbr_pred_val = torch.tensor(classical["gbr"].predict(X_last_val_s), dtype=torch.float32, device=device)
    xgb_pred_val = (torch.tensor(classical["xgb"].predict(X_last_val_s), dtype=torch.float32, device=device)
                     if classical["xgb"] is not None else gbr_pred_val.clone())
    preds_val = torch.stack([deep_rul_val, gbr_pred_val, xgb_pred_val], dim=-1)
    gate = AdaptiveGatingEnsemble(condition_dim=condition_val.size(-1), n_models=3).to(device)
    gate_opt = torch.optim.Adam(gate.parameters(), lr=1e-3)
    gate_epochs = max(10, cfg.FINETUNE_EPOCHS)
    for _ in range(gate_epochs):
        gate_opt.zero_grad()
        gated, _ = gate(preds_val, condition_val)
        gate_loss = torch.mean((gated - y_rul_val) ** 2)
        gate_loss.backward()
        gate_opt.step()
    log(f"Gating ensemble trained  |  final val RUL MSE={gate_loss.item():.3f}")

    # ---------------- Stage 6: holdout evaluation ----------------
    stage(6)
    out_ho = _batched_forward(model, src_holdout["X"], src_holdout["hi"], cfg.EVAL_BATCH_SIZE, device)
    fused_scores = out_ho["fused_anomaly_score"].numpy()
    y_bin_ho = (src_holdout["y_cls"] != "healthy").astype(int)
    anom_preds = threshold.predict(fused_scores)

    auc_roc = roc_auc_score(y_bin_ho, fused_scores) if len(np.unique(y_bin_ho)) > 1 else float("nan")
    ap = average_precision_score(y_bin_ho, fused_scores) if len(np.unique(y_bin_ho)) > 1 else float("nan")
    f1 = f1_score(y_bin_ho, anom_preds, zero_division=0)
    log(f"Anomaly detection: AUC-ROC={auc_roc:.4f}  AP={ap:.4f}  F1(adaptive-thr)={f1:.4f}")

    fault_pred = out_ho["fault_logits"].argmax(-1).numpy()
    fault_f1 = f1_score(src_holdout["y_cls_idx"], fault_pred, average="macro", zero_division=0)
    log(f"Fault classification macro-F1: {fault_f1:.4f}")

    deep_rul = out_ho["rul_pred"].numpy()
    X_last_s = classical["scaler"].transform(src_holdout["X"][:, -1, :])
    gbr_rul = classical["gbr"].predict(X_last_s)
    xgb_rul = classical["xgb"].predict(X_last_s) if classical["xgb"] is not None else gbr_rul
    with torch.no_grad():
        preds3 = torch.stack([out_ho["rul_pred"].to(device),
                               torch.tensor(gbr_rul, dtype=torch.float32, device=device),
                               torch.tensor(xgb_rul, dtype=torch.float32, device=device)], dim=-1)
        gated_rul, _ = gate(preds3, out_ho["repr"].to(device))
    gated_rul = gated_rul.detach().cpu().numpy()

    rmse_deep = float(np.sqrt(mean_squared_error(src_holdout["y_rul"], deep_rul)))
    rmse_gbr = float(np.sqrt(mean_squared_error(src_holdout["y_rul"], gbr_rul)))
    rmse_gated = float(np.sqrt(mean_squared_error(src_holdout["y_rul"], gated_rul)))
    log(f"RUL RMSE -- deep head: {rmse_deep:.2f}  GBR: {rmse_gbr:.2f}  gated ensemble: {rmse_gated:.2f}")

    # ---------------- Stage 7: statistical significance ----------------
    stage(7)
    thresh_gbr = np.quantile(gbr_rul, 0.1)
    gbr_anom_pred = (gbr_rul < thresh_gbr).astype(int)
    mstat, mp, mtable = _mcnemar_test(y_bin_ho, anom_preds, gbr_anom_pred)
    log(f"McNemar (framework vs GBR-derived anomaly flag): stat={mstat:.3f}, p={mp:.4f}")

    resid_deep = np.abs(deep_rul - src_holdout["y_rul"])
    resid_gbr = np.abs(gbr_rul - src_holdout["y_rul"])
    try:
        w_stat, w_p = wilcoxon(resid_deep, resid_gbr)
        w_stat, w_p = float(w_stat), float(w_p)
    except Exception:
        w_stat, w_p = float("nan"), float("nan")
    log(f"Wilcoxon signed-rank (|RUL residual| deep vs GBR): stat={w_stat:.3f}, p={w_p:.4f}")

    fault_probs = torch.softmax(out_ho["fault_logits"], dim=-1).numpy()
    y_fail_bin = (src_holdout["y_cls_idx"] == 2).astype(int)
    ece = _expected_calibration_error(y_fail_bin, fault_probs[:, 2])
    log(f"Expected Calibration Error (failure class): {ece:.4f}")

    # ---------------- Stage 8: ablation study ----------------
    stage(8)

    def eval_fn(flags):
        o = _batched_forward(model, src_holdout["X"], src_holdout["hi"], cfg.EVAL_BATCH_SIZE, device, **flags)
        rmse = float(np.sqrt(mean_squared_error(src_holdout["y_rul"], o["rul_pred"].numpy())))
        auc_ = (roc_auc_score(y_bin_ho, o["fused_anomaly_score"].numpy())
                if len(np.unique(y_bin_ho)) > 1 else float("nan"))
        return {"RUL_RMSE": rmse, "Anomaly_AUC": auc_}

    full_flags = {c: True for c in ABLATION_COMPONENTS}
    ablation_rows = [{"variant": "Full model", **eval_fn(full_flags)}]
    for comp in ABLATION_COMPONENTS:
        flags = dict(full_flags)
        flags[comp] = False
        ablation_rows.append({"variant": f"– without {FRIENDLY_NAMES[comp]}", **eval_fn(flags)})
        log(f"Ablation [{comp}=False]: RMSE={ablation_rows[-1]['RUL_RMSE']:.2f}  AUC={ablation_rows[-1]['Anomaly_AUC']:.3f}")
    ablation_df = pd.DataFrame(ablation_rows)

    # ---------------- Stage 9: explainability ----------------
    stage(9)
    n_explain = min(24, len(src_holdout["X"]))
    Xe = torch.tensor(src_holdout["X"][:n_explain]).to(device)
    HIe = torch.tensor(src_holdout["hi"][:n_explain]).to(device)

    def rul_model_fn(x_in):
        reps = x_in.size(0) // n_explain
        hi_in = HIe.repeat(reps, 1) if reps > 1 else HIe
        return model(x_in, hi_in)["rul_pred"]

    ig = IntegratedGradients(rul_model_fn, steps=24)
    attributions = ig.attribute(Xe)
    fi = feature_importance(attributions, cfg.FEATURES)
    ti = temporal_importance(attributions)
    top5 = list(fi.items())[:5]
    log("Top-5 sensors by Integrated-Gradients importance: " + ", ".join(f"{n}={s:.4f}" for n, s in top5))

    # ---------------- Stage 10: figures ----------------
    stage(10)
    plt.style.use("dark_background")
    figs = {}

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot([h["total"] for h in pretrain_history], marker="o", color="#5eead4")
    axes[0].set_title("Self-Supervised Pretraining Loss", fontsize=10)
    axes[0].set_xlabel("Epoch")
    axes[1].plot([h["total"] for h in finetune_history], marker="o", label="total", color="#f5a623")
    axes[1].plot([h["rul"] for h in finetune_history], marker="o", label="RUL (MSE)", color="#5eead4")
    axes[1].plot([h["physics"] for h in finetune_history], marker="o", label="physics", color="#ff6b6b")
    axes[1].set_title("Multi-Task Fine-Tuning Loss", fontsize=10)
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=8)
    figs["training_curves"] = _fig_to_b64()

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    if len(np.unique(y_bin_ho)) > 1:
        fpr, tpr, _ = roc_curve(y_bin_ho, fused_scores)
        ax.plot(fpr, tpr, label=f"Attention-Fused Ensemble (AUC={auc_roc:.3f})", lw=2, color="#f5a623")
    ax.plot([0, 1], [0, 1], "--", lw=1, color="#8892a6", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — FD001 Holdout", fontsize=10)
    ax.legend(fontsize=8)
    figs["roc_curve"] = _fig_to_b64()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, preds, name in zip(axes, [deep_rul, gated_rul], ["Deep RUL Head", "Adaptive Gated Ensemble"]):
        ax.scatter(src_holdout["y_rul"], preds, alpha=0.5, s=14, color="#5eead4")
        lims = [0, max(float(src_holdout["y_rul"].max()), float(preds.max())) + 5]
        ax.plot(lims, lims, "--", lw=1.2, color="#ff6b6b")
        ax.set_xlabel("True RUL")
        ax.set_ylabel("Predicted RUL")
        ax.set_title(name, fontsize=10)
    figs["rul_scatter"] = _fig_to_b64()

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    alpha = out_ho["fusion_alpha"].numpy()
    bp = ax.boxplot(alpha, labels=["LSTM AE", "Dense AE", "Transformer AE", "CNN AE"], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#1d2740")
        patch.set_edgecolor("#f5a623")
    ax.set_title("Adaptive Attention-Fusion Weights", fontsize=10)
    ax.set_ylabel(r"$\alpha_i$")
    figs["fusion_weights"] = _fig_to_b64()

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh(ablation_df["variant"], ablation_df["RUL_RMSE"], color="#5eead4")
    ax.set_xlabel("RUL RMSE (lower is better)")
    ax.set_title("Ablation Study", fontsize=10)
    figs["ablation"] = _fig_to_b64()

    fig, ax = plt.subplots(figsize=(7.2, 5))
    names, scores = zip(*sorted(fi.items(), key=lambda kv: kv[1]))
    ax.barh(names, scores, color="#f5a623")
    ax.set_title("Sensor Ranking — Integrated Gradients", fontsize=10)
    figs["sensor_importance"] = _fig_to_b64()

    log("All figures rendered.")

    results = {
        "data_source": data_source,
        "mode": mode,
        "device": str(device),
        "n_sequences": int(n),
        "splits": {"train": int(len(tr_idx)), "val": int(len(val_idx)), "holdout": int(len(ho_idx))},
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "metrics": {
            "anomaly_auc_roc": None if np.isnan(auc_roc) else round(float(auc_roc), 4),
            "anomaly_ap": None if np.isnan(ap) else round(float(ap), 4),
            "anomaly_f1": round(float(f1), 4),
            "fault_macro_f1": round(float(fault_f1), 4),
            "rul_rmse_deep_head": round(rmse_deep, 3),
            "rul_rmse_gbr": round(rmse_gbr, 3),
            "rul_rmse_gated_ensemble": round(rmse_gated, 3),
            "ece_failure_class": round(ece, 4),
        },
        "threshold": threshold.summary(),
        "uncertainty": {k: round(v, 5) for k, v in uncertainty_summary.items()},
        "significance": {
            "mcnemar_stat": round(mstat, 3), "mcnemar_p": round(mp, 4),
            "wilcoxon_stat": None if np.isnan(w_stat) else round(w_stat, 3),
            "wilcoxon_p": None if np.isnan(w_p) else round(w_p, 4),
        },
        "ablation_table": ablation_df.round(4).to_dict(orient="records"),
        "feature_importance": [{"sensor": k, "importance": round(float(v), 5)} for k, v in list(fi.items())[:10]],
        "figures": figs,
    }
    return results


def run_pipeline_safe(mode="quick", progress_cb=None, seed=None):
    try:
        return {"ok": True, "results": run_pipeline(mode=mode, progress_cb=progress_cb, seed=seed)}
    except Exception as e:
        tb = traceback.format_exc()
        if progress_cb:
            progress_cb({"type": "log", "text": f"ERROR: {e}"})
            progress_cb({"type": "log", "text": tb})
        return {"ok": False, "error": str(e), "traceback": tb}
