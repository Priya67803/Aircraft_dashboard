"""
data.py
=======
NASA CMAPSS loading / preprocessing, scoped to FD001.

`prepare_fd001()` tries the real dataset via kagglehub first. If that
fails (no internet to Kaggle, no kaggle.json credentials, offline
sandbox, etc.) it transparently falls back to a physically-plausible
SYNTHETIC turbofan-degradation dataset with the same schema, so the
rest of the pipeline (and the dashboard) always has something real to
train and show results on. The fallback is clearly flagged in the
returned dict as `data_source`.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

COLUMNS = (
    ['unit_number', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3']
    + [f'sensor_{i}' for i in range(1, 22)]
)


def _find_file(base_path: str, filename: str) -> str:
    for root, _, files in os.walk(base_path):
        for f in files:
            if filename in f:
                return os.path.join(root, f)
    raise FileNotFoundError(f"Could not locate '{filename}' under {base_path}")


def download_cmapss() -> str:
    import kagglehub
    return kagglehub.dataset_download("behrad3d/nasa-cmaps")


def load_raw_real(domain: str = "FD001", split: str = "train") -> pd.DataFrame:
    base_path = download_cmapss()
    fname = f"{split}_{domain}.txt"
    fpath = _find_file(base_path, fname)
    df = pd.read_csv(fpath, sep=r"\s+", header=None, names=COLUMNS)
    return df


def generate_synthetic_cmapss(n_engines: int = 24, seed: int = 42) -> pd.DataFrame:
    """Physically-motivated synthetic stand-in for FD001: each engine
    runs from healthy to failure with monotonic sensor drift + noise,
    following the same column schema as the real dataset so every
    downstream function (RUL labeling, scaling, sequencing) works
    unmodified."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_engines + 1):
        life = int(rng.integers(130, 260))
        t = np.arange(1, life + 1)
        frac = t / life  # 0 (healthy) -> 1 (failure)

        op1 = rng.normal(20, 2, size=life)
        op2 = rng.normal(0.6, 0.05, size=life)
        op3 = np.full(life, 100.0)

        row = {
            'unit_number': unit, 'time_cycles': t,
            'op_setting_1': op1, 'op_setting_2': op2, 'op_setting_3': op3,
        }
        # sensors 1..21: mix of drifting (degradation-sensitive) and flat/noisy sensors
        degrading = {2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21}
        for s in range(1, 22):
            base = rng.normal(400 + s * 5, 3)
            noise = rng.normal(0, 0.8, size=life)
            if s in degrading:
                drift_dir = 1 if s % 2 == 0 else -1
                trend = drift_dir * (frac ** 1.6) * rng.uniform(8, 22)
                curve = base + trend + noise
            else:
                curve = base + noise
            row[f'sensor_{s}'] = curve
        rows.append(pd.DataFrame(row))
    df = pd.concat(rows, ignore_index=True)
    return df[COLUMNS]


def _label_fault(rul: float) -> str:
    if rul > 50:
        return "healthy"
    elif rul > 20:
        return "degradation"
    return "failure"


def compute_rul(df: pd.DataFrame, rul_cap: int) -> pd.DataFrame:
    max_c = df.groupby('unit_number')['time_cycles'].max()
    df = df.merge(max_c.rename("max_cycles"), on="unit_number")
    df["RUL"] = df["max_cycles"] - df["time_cycles"]
    df["RUL_capped"] = df["RUL"].clip(upper=rul_cap)
    df.drop(columns=["max_cycles"], inplace=True)
    df["fault_label"] = df["RUL"].apply(_label_fault)
    return df


def scale_per_engine(df: pd.DataFrame, features) -> pd.DataFrame:
    scaled = []
    for eng in df['unit_number'].unique():
        edf = df[df['unit_number'] == eng].copy()
        edf[features] = RobustScaler().fit_transform(edf[features])
        scaled.append(edf)
    return pd.concat(scaled).reset_index(drop=True)


def fit_pca_health_index(df: pd.DataFrame, features, seed: int, n_components: int = 1) -> PCA:
    healthy = df[df["fault_label"] == "healthy"]
    if len(healthy) < 5:
        healthy = df
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(healthy[features].values)
    return pca


def pca_health_index(pca: PCA, X: np.ndarray) -> np.ndarray:
    proj = pca.transform(X)[:, 0]
    if len(proj) > 1 and np.corrcoef(proj, np.arange(len(proj)))[0, 1] > 0:
        proj = -proj
    return proj


def build_sequences(df: pd.DataFrame, features, seq_len: int, seed: int):
    pca = fit_pca_health_index(df, features, seed)
    Xs, ycls, yrul, yrul_next, his, units = [], [], [], [], [], []
    for unit in df['unit_number'].unique():
        ud = df[df['unit_number'] == unit].sort_values("time_cycles")
        v = ud[features].values
        rv = ud["RUL"].values
        rc = ud["RUL_capped"].values
        hi_full = pca_health_index(pca, v)
        n = len(v)
        for i in range(n - seq_len - 1):
            Xs.append(v[i:i + seq_len])
            ycls.append(_label_fault(np.min(rv[i:i + seq_len])))
            yrul.append(rc[i + seq_len - 1])
            yrul_next.append(rc[i + seq_len])
            his.append(hi_full[i:i + seq_len])
            units.append(unit)
    return dict(
        X=np.array(Xs, dtype=np.float32),
        y_cls=np.array(ycls),
        y_rul=np.array(yrul, dtype=np.float32),
        y_rul_next=np.array(yrul_next, dtype=np.float32),
        hi=np.array(his, dtype=np.float32),
        unit=np.array(units),
        pca=pca,
    )


CLASS_TO_IDX = {"healthy": 0, "degradation": 1, "failure": 2}


def encode_labels(y_cls: np.ndarray) -> np.ndarray:
    return np.array([CLASS_TO_IDX[c] for c in y_cls], dtype=np.int64)


def prepare_fd001(cfg, log=print):
    """Returns (seqs_dict, data_source_str)."""
    try:
        log("Attempting to download real NASA CMAPSS (FD001) via kagglehub ...")
        raw = load_raw_real(cfg.DOMAIN, split="train")
        source = "NASA CMAPSS (real, via kagglehub)"
        log("Real dataset downloaded successfully.")
    except Exception as e:
        log(f"Real dataset unavailable in this environment ({type(e).__name__}: {e}).")
        log(f"Falling back to a synthetic FD001-like turbofan degradation dataset "
            f"({cfg.N_ENGINES_SYNTH} engines) so the pipeline can still run end-to-end.")
        raw = generate_synthetic_cmapss(n_engines=cfg.N_ENGINES_SYNTH, seed=cfg.SEED)
        source = "Synthetic FD001-like data (offline fallback)"

    raw = compute_rul(raw, cfg.RUL_CAP)
    scaled = scale_per_engine(raw, cfg.FEATURES)
    seqs = build_sequences(scaled, cfg.FEATURES, cfg.SEQ_LEN, cfg.SEED)
    seqs["y_cls_idx"] = encode_labels(seqs["y_cls"])
    return seqs, source
