"""
AwareLGBM: LightGBM multi-output custom objective benchmark.

Compares 6 custom objective functions against vanilla LightGBM (k independent models)
on synthetic data with known cross-output correlation structure.

Cases:
  0. Vanilla LightGBM (baseline) — k independent LGBMRegressors
  1. Independent MSE            — custom obj, no cross-output coupling
  2. Mahalanobis                — precision-weighted, gradient couples outputs
  3. Asymmetric Quantile        — penalize under-prediction more (tau=0.7)
  4. Shared-Norm Regularization — L2 norm penalty across outputs
  5. Cholesky Multi-Output      — numerically stable Mahalanobis via Cholesky
  6. Pairwise LambdaRank        — cross-sectional ranking (separate benchmark)

LightGBM 4.6 API:
  - Custom objectives go in params['objective'] as a callable
  - Multi-output via num_class=k with dummy label of shape (n,)
  - preds arrive as (n, k) ndarray; return (grad, hess) each (n, k)

Usage:
    python tests/AwareLGBM.py
    python tests/AwareLGBM.py --n_samples 8000 --k 5 --seed 42
"""

import argparse
import os
import time
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="An input array is constant")

import lightgbm as lgb


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_correlated_regression(n=5000, d=20, k=3, noise=0.5, seed=0):
    """Generate multi-output regression data with known correlation structure."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float64)

    W = rng.randn(d, k) * 0.5
    # Correlation structure via Cholesky factor
    A = rng.randn(k, k) * 0.3
    L_true = np.eye(k) + np.tril(A, -1)
    Sigma_true = L_true @ L_true.T

    Y = X @ W + noise * rng.randn(n, k) @ L_true.T

    split = int(0.8 * n)
    return {
        "X_train": X[:split], "Y_train": Y[:split],
        "X_test": X[split:], "Y_test": Y[split:],
        "Sigma_true": Sigma_true, "L_true": L_true,
        "n_train": split, "n_test": n - split, "d": d, "k": k,
    }


def make_ranking_data(n_queries=200, n_items=50, d=15, seed=0):
    """Generate cross-sectional ranking data (queries = timestamps, items = assets)."""
    rng = np.random.RandomState(seed)
    n_total = n_queries * n_items
    X = rng.randn(n_total, d).astype(np.float64)

    true_scores = X[:, :4].sum(axis=1) + 0.5 * rng.randn(n_total)
    labels = np.digitize(true_scores, np.percentile(true_scores, [20, 40, 60, 80]))

    groups = [n_items] * n_queries
    split_q = int(0.8 * n_queries)
    split_i = split_q * n_items

    return {
        "X_train": X[:split_i], "labels_train": labels[:split_i].astype(np.float64),
        "groups_train": groups[:split_q],
        "X_test": X[split_i:], "labels_test": labels[split_i:].astype(np.float64),
        "groups_test": groups[split_q:],
        "n_items": n_items, "n_queries_test": n_queries - split_q,
    }


# ---------------------------------------------------------------------------
# Custom objectives (LightGBM 4.6: preds shape is (n, k), return (n, k))
# ---------------------------------------------------------------------------

def make_independent_mse_obj(Y_train):
    """Case 1: Independent MSE — no cross-output coupling."""
    Y = Y_train.copy()
    def obj(preds, dataset):
        grad = -2.0 * (Y - preds)        # (n, k)
        hess = np.full_like(grad, 2.0)
        return grad, hess
    return obj


def make_mahalanobis_obj(Y_train, P):
    """Case 2: Mahalanobis loss — gradient couples outputs via precision matrix."""
    Y = Y_train.copy()
    diag_P = np.diag(P).copy()
    n = Y.shape[0]
    def obj(preds, dataset):
        r = Y - preds                     # (n, k)
        grad = -2.0 * (r @ P)             # coupled across outputs
        hess = np.broadcast_to(2.0 * diag_P, (n, preds.shape[1])).copy()
        return grad, hess
    return obj


def make_asymmetric_quantile_obj(Y_train, tau=0.7):
    """Case 3: Asymmetric quantile (pinball) loss.

    Pinball loss: L = tau * r  if r >= 0,  (tau-1) * r  if r < 0   (r = y - yhat)
    grad = dL/d(yhat) = -tau  if r >= 0,  (1-tau)  if r < 0
    hess = 1.0 (constant — gradient descent, standard for quantile GBDT)
    """
    Y = Y_train.copy()
    def obj(preds, dataset):
        r = Y - preds
        grad = np.where(r >= 0, -tau, (1.0 - tau))
        hess = np.ones_like(grad)
        return grad, hess
    return obj


def make_shared_norm_obj(Y_train, lam=0.1):
    """Case 4: MSE + lambda * ||y_hat||_2 regularization across outputs."""
    Y = Y_train.copy()
    eps_norm = 1e-8
    def obj(preds, dataset):
        r = Y - preds
        norms = np.sqrt(np.sum(preds ** 2, axis=1, keepdims=True) + eps_norm)
        grad = -2.0 * r + lam * preds / norms
        hess_mse = 2.0
        hess_reg = lam * (norms ** 2 - preds ** 2) / (norms ** 3 + eps_norm)
        hess = np.maximum(hess_mse + hess_reg, 1e-4)
        return grad, hess
    return obj


def make_cholesky_obj(Y_train, Sigma):
    """Case 5: Cholesky-decomposed Mahalanobis — numerically stable."""
    Y = Y_train.copy()
    L_cho, low = cho_factor(Sigma, lower=True)
    P = cho_solve((L_cho, low), np.eye(Sigma.shape[0]))
    diag_P = np.diag(P).copy()
    n = Y.shape[0]
    def obj(preds, dataset):
        r = Y - preds
        grad = -2.0 * (r @ P)
        hess = np.broadcast_to(2.0 * diag_P, (n, preds.shape[1])).copy()
        return grad, hess
    return obj


# ---------------------------------------------------------------------------
# Case 6: Pairwise LambdaRank (single-output, group-aware)
# ---------------------------------------------------------------------------

def _compute_dcg(rels):
    positions = np.arange(1, len(rels) + 1)
    return np.sum((2.0 ** rels - 1.0) / np.log2(positions + 1.0))


def lambdarank_obj(preds, dataset):
    """Case 6: Pairwise LambdaRank with |delta NDCG| weighting."""
    labels = dataset.get_label()
    groups = dataset.get_group()
    sigma = 1.0

    grad = np.zeros_like(preds)
    hess = np.zeros_like(preds)

    idx = 0
    for g_size in groups:
        g_lab = labels[idx:idx + g_size]
        g_pred = preds[idx:idx + g_size]

        sorted_idx = np.argsort(-g_pred)
        ranks = np.empty(g_size, dtype=np.int64)
        ranks[sorted_idx] = np.arange(g_size)

        ideal_order = np.sort(g_lab)[::-1]
        idcg = _compute_dcg(ideal_order)
        if idcg < 1e-12:
            idx += g_size
            continue

        # Vectorized pairwise: meshgrid over group
        lab_i, lab_j = np.meshgrid(g_lab, g_lab, indexing="ij")
        pred_i, pred_j = np.meshgrid(g_pred, g_pred, indexing="ij")
        rank_i, rank_j = np.meshgrid(ranks, ranks, indexing="ij")

        mask = lab_i > lab_j
        delta_s = pred_i - pred_j

        with np.errstate(over="ignore"):
            exp_val = np.exp(sigma * delta_s)
        p_lambda = 1.0 / (1.0 + exp_val)

        gain_diff = np.abs(2.0 ** lab_i - 2.0 ** lab_j)
        disc_diff = np.abs(
            1.0 / np.log2(rank_i + 2.0) - 1.0 / np.log2(rank_j + 2.0)
        )
        delta_ndcg = (gain_diff * disc_diff) / idcg

        lam = -sigma * p_lambda * delta_ndcg * mask
        w = sigma ** 2 * p_lambda * (1.0 - p_lambda) * delta_ndcg * mask

        g_grad = lam.sum(axis=1) - lam.sum(axis=0)
        g_hess = w.sum(axis=1) + w.sum(axis=0)
        g_hess = np.maximum(g_hess, 1e-6)

        grad[idx:idx + g_size] = g_grad
        hess[idx:idx + g_size] = g_hess
        idx += g_size

    return grad, hess


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def eval_multioutput(Y_true, Y_pred, P=None, tau=0.7):
    """Evaluate multi-output predictions."""
    mse_per_output = np.mean((Y_true - Y_pred) ** 2, axis=0)
    mse_total = mse_per_output.mean()
    r = Y_true - Y_pred
    mahal = float(np.mean(np.sum((r @ P) * r, axis=1))) if P is not None else np.nan
    # Quantile (pinball) loss
    qloss = np.mean(np.where(r >= 0, tau * r, (tau - 1.0) * r))
    corr_per_output = np.array([
        np.corrcoef(Y_true[:, j], Y_pred[:, j])[0, 1]
        for j in range(Y_true.shape[1])
    ])
    return {
        "mse_total": mse_total,
        "mse_per_output": mse_per_output,
        "mahal_dist": mahal,
        "qloss": qloss,
        "corr_per_output": corr_per_output,
        "corr_mean": corr_per_output.mean(),
    }


def eval_ranking(labels, preds, groups, n_items):
    """Evaluate ranking predictions via Spearman rho per query."""
    corrs = []
    idx = 0
    for g_size in groups:
        sl = slice(idx, idx + g_size)
        rho, _ = spearmanr(labels[sl], preds[sl])
        if not np.isnan(rho):
            corrs.append(rho)
        idx += g_size
    return {"spearman_mean": np.mean(corrs), "spearman_std": np.std(corrs)}


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

LGB_PARAMS_BASE = {
    "verbose": -1,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
}

N_ROUNDS = 300


def train_vanilla_baseline(data):
    """Case 0: Train k independent LGBMRegressors (vanilla baseline)."""
    k = data["k"]
    models = []
    for j in range(k):
        ds = lgb.Dataset(data["X_train"], label=data["Y_train"][:, j], free_raw_data=False)
        model = lgb.train({**LGB_PARAMS_BASE}, ds, num_boost_round=N_ROUNDS)
        models.append(model)
    Y_pred = np.column_stack([m.predict(data["X_test"]) for m in models])
    return Y_pred


def train_multioutput_custom(data, fobj, init_score=None, extra_params=None):
    """Train a single multi-output booster with custom objective via num_class=k.

    LightGBM 4.6 API: pass objective as callable in params.
    Dataset label is dummy (length n); real labels live in fobj closure.
    preds arrive as (n, k) ndarray.

    init_score: optional (n, k) array for initialization (e.g. per-output mean/quantile).
                During training, preds = init_score + tree_values.
                model.predict() returns only tree_values, so we add the per-output
                bias back to test predictions.
    """
    k = data["k"]
    n_train = data["n_train"]

    ds = lgb.Dataset(data["X_train"], label=np.zeros(n_train), free_raw_data=False)
    if init_score is not None:
        ds.set_init_score(init_score.ravel())

    params = {
        **LGB_PARAMS_BASE,
        "num_class": k,
        "objective": fobj,
    }
    if extra_params:
        params.update(extra_params)

    model = lgb.train(params, ds, num_boost_round=N_ROUNDS)

    raw_pred = model.predict(data["X_test"])
    if raw_pred.ndim == 1:
        raw_pred = raw_pred.reshape(-1, k)

    # predict() doesn't include init_score — add per-output bias back
    if init_score is not None:
        bias = init_score[0]  # (k,) — same for all rows
        raw_pred = raw_pred + bias

    return raw_pred


def train_lambdarank_custom(rank_data):
    """Train with custom LambdaRank objective."""
    ds = lgb.Dataset(
        rank_data["X_train"],
        label=rank_data["labels_train"],
        group=rank_data["groups_train"],
        free_raw_data=False,
    )
    params = {
        **LGB_PARAMS_BASE,
        "objective": lambdarank_obj,
    }
    model = lgb.train(params, ds, num_boost_round=200)
    return model.predict(rank_data["X_test"])


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    name: str
    mse: float
    mahal: float
    qloss: float
    corr_mean: float
    corr_per_output: np.ndarray
    elapsed: float


def print_results(results: list[CaseResult], k: int):
    """Print comparison table."""
    hdr_corr = "  ".join([f"r_{j}" for j in range(k)])
    print(
        f"\n{'Case':<30s}  {'MSE':>8s}  {'Mahal':>8s}  {'QLoss':>7s}  "
        f"{'rMean':>6s}  {hdr_corr}  {'Time':>6s}"
    )
    print("-" * (76 + 8 * k))

    baseline_mse = results[0].mse if results else 1.0
    baseline_ql = results[0].qloss if results else 1.0
    for r in results:
        corrs = "  ".join([f"{c:>6.4f}" for c in r.corr_per_output])
        d_mse = (r.mse / baseline_mse - 1) * 100 if baseline_mse > 0 else 0
        d_ql = (r.qloss / baseline_ql - 1) * 100 if baseline_ql > 0 else 0
        if r.name.startswith("0."):
            tag = ""
        elif "Quantile" in r.name:
            tag = f" (ql {d_ql:+.1f}%)"
        else:
            tag = f" (mse {d_mse:+.1f}%)"
        print(
            f"{r.name:<30s}  {r.mse:>8.4f}  {r.mahal:>8.3f}  {r.qloss:>7.4f}  "
            f"{r.corr_mean:>6.4f}  {corrs}  {r.elapsed:>5.1f}s{tag}"
        )


# ---------------------------------------------------------------------------
# Main benchmarks
# ---------------------------------------------------------------------------

def run_multioutput_benchmark(n=5000, d=20, k=3, seed=0):
    """Run Cases 0-5 and print comparison."""
    print(f"=== Multi-Output Benchmark (n={n}, d={d}, k={k}, seed={seed}) ===\n")

    data = make_correlated_regression(n=n, d=d, k=k, seed=seed)
    Y_train = data["Y_train"]
    Sigma = np.cov(Y_train.T)
    P = np.linalg.inv(Sigma)

    # Per-output quantile init: start from the empirical tau-quantile
    tau = 0.7
    q_init = np.quantile(Y_train, tau, axis=0)  # (k,)
    q_init_full = np.broadcast_to(q_init, (data["n_train"], k)).copy()

    cases = [
        ("0. Vanilla (k indep.)",        None,                                          None,        None),
        ("1. Independent MSE",           make_independent_mse_obj(Y_train),             None,        None),
        ("2. Mahalanobis",               make_mahalanobis_obj(Y_train, P),              None,        None),
        ("3. Asymm. Quantile (tau=0.7)", make_asymmetric_quantile_obj(Y_train, tau=tau),q_init_full, {"learning_rate": 0.1}),
        ("4. Shared-Norm (lam=0.1)",     make_shared_norm_obj(Y_train, lam=0.1),        None,        None),
        ("5. Cholesky Multi-Out",        make_cholesky_obj(Y_train, Sigma),             None,        None),
    ]

    results = []
    for name, fobj, init_score, extra_params in cases:
        t0 = time.time()
        if fobj is None:
            Y_pred = train_vanilla_baseline(data)
        else:
            Y_pred = train_multioutput_custom(data, fobj, init_score=init_score, extra_params=extra_params)
        elapsed = time.time() - t0

        metrics = eval_multioutput(data["Y_test"], Y_pred, P)
        results.append(CaseResult(
            name=name,
            mse=metrics["mse_total"],
            mahal=metrics["mahal_dist"],
            qloss=metrics["qloss"],
            corr_mean=metrics["corr_mean"],
            corr_per_output=metrics["corr_per_output"],
            elapsed=elapsed,
        ))
        print(f"  [done] {name} ({elapsed:.1f}s)")

    print_results(results, k)
    return results


def run_ranking_benchmark(seed=0):
    """Run Case 6 (LambdaRank) vs regression baseline and built-in LambdaRank."""
    print(f"\n=== Ranking Benchmark (seed={seed}) ===\n")

    rank_data = make_ranking_data(seed=seed)
    n_items = rank_data["n_items"]

    # Baseline: vanilla regression on relevance labels
    t0 = time.time()
    ds_reg = lgb.Dataset(
        rank_data["X_train"], label=rank_data["labels_train"], free_raw_data=False,
    )
    model_reg = lgb.train({**LGB_PARAMS_BASE}, ds_reg, num_boost_round=200)
    preds_reg = model_reg.predict(rank_data["X_test"])
    t_reg = time.time() - t0
    m_reg = eval_ranking(rank_data["labels_test"], preds_reg, rank_data["groups_test"], n_items)
    print(f"  [done] Vanilla Regression ({t_reg:.1f}s)")

    # Built-in LambdaRank
    t0 = time.time()
    ds_rank = lgb.Dataset(
        rank_data["X_train"], label=rank_data["labels_train"],
        group=rank_data["groups_train"], free_raw_data=False,
    )
    model_builtin = lgb.train(
        {**LGB_PARAMS_BASE, "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5, 10]},
        ds_rank, num_boost_round=200,
    )
    preds_builtin = model_builtin.predict(rank_data["X_test"])
    t_builtin = time.time() - t0
    m_builtin = eval_ranking(rank_data["labels_test"], preds_builtin, rank_data["groups_test"], n_items)
    print(f"  [done] Built-in LambdaRank ({t_builtin:.1f}s)")

    # Custom LambdaRank
    t0 = time.time()
    preds_custom = train_lambdarank_custom(rank_data)
    t_custom = time.time() - t0
    m_custom = eval_ranking(rank_data["labels_test"], preds_custom, rank_data["groups_test"], n_items)
    print(f"  [done] Custom LambdaRank ({t_custom:.1f}s)")

    print(f"\n{'Case':<30s}  {'Spearman':>9s}  {'Std':>7s}  {'Time':>6s}")
    print("-" * 58)
    for name, m, t in [
        ("Vanilla Regression",  m_reg,     t_reg),
        ("Built-in LambdaRank", m_builtin, t_builtin),
        ("6. Custom LambdaRank",m_custom,  t_custom),
    ]:
        print(f"{name:<30s}  {m['spearman_mean']:>9.4f}  {m['spearman_std']:>7.4f}  {t:>5.1f}s")


# ---------------------------------------------------------------------------
# Real data benchmark: erdNov16Prod.csv
# Features: mr[ABCD]* columns (324 alphas)
# Targets: p1d (primary), p3d, p5d (multi-output cases)
# Split: temporal — train on first 80% dates, test on last 20%
# Eval: RMSE, per-date cross-sectional Spearman & Pearson
# ---------------------------------------------------------------------------

ERD_PATH = os.path.expanduser("~/erdNov16Prod.csv")
TARGET_COLS = ["p1d", "p3d", "p5d"]   # multi-output horizons
PRIMARY_TARGET = "p1d"                  # evaluation focus


def load_erd_data(path=ERD_PATH, train_frac=0.8):
    """Load erdNov16Prod.csv with mr* features, temporal train/test split."""
    import pandas as pd

    df = pd.read_csv(path)
    mr_cols = sorted([c for c in df.columns if c.startswith("mr")])
    keep_cols = ["date", "token"] + TARGET_COLS + mr_cols
    df = df[keep_cols].copy()

    # Drop rows with NaN in features or targets
    df = df.dropna(subset=TARGET_COLS).reset_index(drop=True)
    # Fill remaining NaN in mr features with 0 (common for sparse alphas)
    df[mr_cols] = df[mr_cols].fillna(0.0)

    # Temporal split
    dates = sorted(df["date"].unique())
    split_date = dates[int(len(dates) * train_frac)]
    train_mask = df["date"] < split_date
    test_mask = df["date"] >= split_date

    df_train = df[train_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)

    X_train = df_train[mr_cols].values.astype(np.float64)
    X_test = df_test[mr_cols].values.astype(np.float64)
    Y_train = df_train[TARGET_COLS].values.astype(np.float64)
    Y_test = df_test[TARGET_COLS].values.astype(np.float64)

    # Group sizes per date (for ranking)
    train_groups = df_train.groupby("date").size().tolist()
    test_groups = df_test.groupby("date").size().tolist()

    print(f"  Data loaded: {len(mr_cols)} features, {len(dates)} dates "
          f"(train {df_train['date'].nunique()}, test {df_test['date'].nunique()})")
    print(f"  Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")
    print(f"  Split date: {split_date}")
    print(f"  Target corr (train): p1d/p3d={np.corrcoef(Y_train[:,0],Y_train[:,1])[0,1]:.3f}, "
          f"p1d/p5d={np.corrcoef(Y_train[:,0],Y_train[:,2])[0,1]:.3f}")

    return {
        "X_train": X_train, "X_test": X_test,
        "Y_train": Y_train, "Y_test": Y_test,            # (n, 3) multi-horizon
        "y_train": Y_train[:, 0], "y_test": Y_test[:, 0], # p1d only
        "train_groups": train_groups, "test_groups": test_groups,
        "dates_test": df_test["date"].values,
        "tokens_test": df_test["token"].values,
        "n_train": X_train.shape[0], "n_test": X_test.shape[0],
        "d": X_train.shape[1], "k": len(TARGET_COLS),
        "mr_cols": mr_cols,
    }


def eval_erd(y_true, y_pred, dates):
    """Evaluate predictions: RMSE, per-date cross-sectional Spearman & Pearson."""
    from scipy.stats import pearsonr

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Per-date cross-sectional correlations
    unique_dates = np.unique(dates)
    spearman_vals, pearson_vals = [], []
    for d in unique_dates:
        mask = dates == d
        yt, yp = y_true[mask], y_pred[mask]
        if len(yt) < 3:
            continue
        rho_s, _ = spearmanr(yt, yp)
        rho_p, _ = pearsonr(yt, yp)
        if not np.isnan(rho_s):
            spearman_vals.append(rho_s)
        if not np.isnan(rho_p):
            pearson_vals.append(rho_p)

    return {
        "rmse": rmse,
        "spearman_mean": np.mean(spearman_vals) if spearman_vals else np.nan,
        "spearman_std": np.std(spearman_vals) if spearman_vals else np.nan,
        "pearson_mean": np.mean(pearson_vals) if pearson_vals else np.nan,
        "pearson_std": np.std(pearson_vals) if pearson_vals else np.nan,
        "n_dates": len(spearman_vals),
    }


# --- Single-output training helpers ---

def train_single_vanilla(X_train, y_train, X_test, n_rounds=N_ROUNDS):
    """Vanilla LightGBM regression on single target."""
    ds = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    model = lgb.train({**LGB_PARAMS_BASE}, ds, num_boost_round=n_rounds)
    return model.predict(X_test)


def train_single_custom(X_train, y_train, X_test, fobj, init_score=None,
                        extra_params=None, n_rounds=N_ROUNDS):
    """Single-output with custom objective (k=1, no num_class trick needed)."""
    ds = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    if init_score is not None:
        ds.set_init_score(init_score)
    params = {**LGB_PARAMS_BASE, "objective": fobj}
    if extra_params:
        params.update(extra_params)
    model = lgb.train(params, ds, num_boost_round=n_rounds)
    preds = model.predict(X_test)
    if init_score is not None:
        preds = preds + init_score[0]
    return preds


# --- Single-output custom objectives ---

def make_single_mse_obj():
    """Custom MSE for single output (verifies custom obj pipeline)."""
    def obj(preds, dataset):
        labels = dataset.get_label()
        grad = preds - labels
        hess = np.ones_like(grad)
        return grad, hess
    return obj


def make_single_quantile_obj(tau=0.7):
    """Pinball loss for single output."""
    def obj(preds, dataset):
        labels = dataset.get_label()
        r = labels - preds
        grad = np.where(r >= 0, -tau, (1.0 - tau))
        hess = np.ones_like(grad)
        return grad, hess
    return obj


def make_single_huber_obj(delta=0.05):
    """Huber loss — robust to outlier returns (clip at delta).

    Quadratic for |r| <= delta, linear beyond. hess=1.0 uniformly
    (gradient descent) to prevent large Newton steps in the linear region.
    """
    def obj(preds, dataset):
        labels = dataset.get_label()
        r = labels - preds
        abs_r = np.abs(r)
        grad = np.where(abs_r <= delta, -r, -delta * np.sign(r))
        hess = np.ones_like(grad)
        return grad, hess
    return obj


# --- XS-Aware objectives (designed for cross-sectional alpha prediction) ---

def make_xs_aware_mse_obj(y_train, groups, alpha=0.5):
    """Cross-Section Aware MSE: the key objective for alpha prediction.

    Insight: 77% of p1d variance is market-level (all tokens move together).
    Standard MSE wastes capacity predicting market direction instead of
    relative token performance, which is what Spearman/Pearson measure.

    Loss:
        L = α * Σ(y_i - ŷ_i)²  +  (1-α) * Σ_t Σ_i (ỹ_ti - ŷ̃_ti)²

    where ỹ, ŷ̃ are cross-sectionally de-meaned within each date group t.

    Gradient:
        grad_i = α * 2(ŷ_i - y_i)  +  (1-α) * 2(ŷ̃_i - ỹ_i)

    The XS component directly optimizes for cross-sectional correlation:
    - Forces predictions to capture which tokens OUTperform vs UNDERperform
    - Ignores the market-level component that correlation metrics ignore
    - α controls trade-off: α=1 → pure MSE, α=0 → pure XS correlation

    Hessian = 1 (gradient descent): stable on noisy financial data,
    avoids Newton overshoot that hurts vanilla LightGBM.
    """
    y = y_train.copy()
    # Pre-compute group slices for fast per-date centering
    group_slices = []
    idx = 0
    for gs in groups:
        group_slices.append(slice(idx, idx + gs))
        idx += gs

    def obj(preds, dataset):
        # MSE gradient: push predictions toward absolute target
        r = y - preds
        grad_mse = -2.0 * r

        # XS gradient: push predictions toward cross-sectional target
        grad_xs = np.zeros_like(preds)
        for sl in group_slices:
            y_g = y[sl]
            p_g = preds[sl]
            y_c = y_g - y_g.mean()   # centered labels within date
            p_c = p_g - p_g.mean()   # centered preds within date
            grad_xs[sl] = -2.0 * (y_c - p_c)

        grad = alpha * grad_mse + (1.0 - alpha) * grad_xs
        hess = np.ones_like(grad)
        return grad, hess

    return obj


def make_xs_corr_obj(y_train, groups, alpha=0.3, beta=0.5):
    """XS-Aware with explicit Pearson correlation gradient.

    Loss:
        L = α * MSE(y, ŷ)
          + (1-α) * [ β * XS_MSE(ỹ, ŷ̃)  +  (1-β) * (-Pearson_t) ]

    The Pearson gradient per cross-section:
        ∂ρ_t/∂ŷ_i = [(y_i - ȳ_t) - ρ_t * (ŷ_i - ŷ̄_t) * s_y/s_ŷ] / (n_t * s_y)

    Maximizing Pearson = minimizing angle between centered y and centered ŷ.
    Combined with XS_MSE, this optimizes both magnitude and direction.

    The Pearson gradient is rescaled to have the same RMS as the XS-MSE gradient
    to prevent divergence from the 1/s_p singularity early in training.
    """
    y = y_train.copy()
    group_slices = []
    idx = 0
    for gs in groups:
        group_slices.append(slice(idx, idx + gs))
        idx += gs

    def obj(preds, dataset):
        grad_mse = -2.0 * (y - preds)

        grad_xs = np.zeros_like(preds)
        grad_corr = np.zeros_like(preds)
        for sl in group_slices:
            y_g, p_g = y[sl], preds[sl]
            n_g = len(y_g)
            if n_g < 3:
                continue
            y_c = y_g - y_g.mean()
            p_c = p_g - p_g.mean()

            # XS MSE gradient
            grad_xs[sl] = -2.0 * (y_c - p_c)

            # Pearson gradient: ∂(-ρ)/∂ŷ_i
            s_y = np.std(y_c) + 1e-10
            s_p = np.std(p_c) + 1e-10
            rho = np.clip(np.mean(y_c * p_c) / (s_y * s_p), -1, 1)
            # Derivative of -ρ w.r.t. ŷ_i
            raw = -(y_c / (n_g * s_y * s_p) - rho * p_c / (n_g * s_p * s_p))
            grad_corr[sl] = raw

        # Rescale Pearson gradient to match XS-MSE gradient RMS (prevent blowup)
        rms_xs = np.sqrt(np.mean(grad_xs ** 2)) + 1e-10
        rms_corr = np.sqrt(np.mean(grad_corr ** 2)) + 1e-10
        grad_corr = grad_corr * (rms_xs / rms_corr)

        grad_rank = beta * grad_xs + (1.0 - beta) * grad_corr
        grad = alpha * grad_mse + (1.0 - alpha) * grad_rank
        hess = np.ones_like(grad)
        return grad, hess

    return obj


def make_mse_rank_hybrid_obj(y_train, groups, rank_weight=0.3):
    """Hybrid MSE + pairwise ranking gradient.

    Base: gradient-descent MSE (hess=1), which already gives strong Spearman.
    Addon: within each cross-section (date group), add a pairwise ranking
    gradient that pushes higher-y items to have higher predictions.

    The ranking gradient is a simplified LambdaRank: for each pair (i,j)
    where y_i > y_j, push pred_i up and pred_j down proportionally to
    sigmoid(pred_j - pred_i), scaled by |y_i - y_j| as importance weight.

    rank_weight controls blend: 0 = pure MSE, 1 = pure ranking.
    """
    y = y_train.copy()
    group_slices = []
    idx = 0
    for gs in groups:
        group_slices.append(slice(idx, idx + gs))
        idx += gs

    sigma = 1.0

    def obj(preds, dataset):
        # MSE gradient (gradient descent)
        grad_mse = preds - y
        grad_rank = np.zeros_like(preds)

        for sl in group_slices:
            y_g = y[sl]
            p_g = preds[sl]
            n_g = len(y_g)
            if n_g < 3:
                continue

            # Vectorized pairwise: all pairs within group
            y_i, y_j = np.meshgrid(y_g, y_g, indexing="ij")
            p_i, p_j = np.meshgrid(p_g, p_g, indexing="ij")

            mask = y_i > y_j  # only pairs where i should rank above j
            delta_s = p_i - p_j  # score difference

            with np.errstate(over="ignore"):
                sig = 1.0 / (1.0 + np.exp(sigma * delta_s))  # sigmoid

            # Importance weight: proportional to |y_i - y_j|
            importance = np.abs(y_i - y_j)
            lam = -sigma * sig * importance * mask

            # Per-item gradient: sum of pushes from all pairs
            g_rank = lam.sum(axis=1) - lam.sum(axis=0)
            grad_rank[sl] = g_rank

        # Normalize ranking gradient to same scale as MSE gradient
        rms_mse = np.sqrt(np.mean(grad_mse ** 2)) + 1e-10
        rms_rank = np.sqrt(np.mean(grad_rank ** 2)) + 1e-10
        grad_rank = grad_rank * (rms_mse / rms_rank)

        grad = (1.0 - rank_weight) * grad_mse + rank_weight * grad_rank
        hess = np.ones_like(grad)
        return grad, hess

    return obj


def _compute_consensus(y_train, X_train, groups):
    """Compute feature consensus: rank-average across all features, rank-matched to target.

    Returns consensus array (same shape as y_train) and list of group slices.
    """
    from scipy.stats import rankdata

    y = y_train.copy()
    n = len(y)
    consensus = np.zeros(n)
    group_slices = []
    idx = 0
    for gs in groups:
        sl = slice(idx, idx + gs)
        group_slices.append(sl)

        X_g = X_train[sl]
        ranks = np.apply_along_axis(
            lambda x: rankdata(x, method='average'), 0, X_g
        )
        mean_rank = ranks.mean(axis=1)

        y_g = y[sl]
        y_sorted = np.sort(y_g)
        consensus_order = np.argsort(np.argsort(mean_rank))
        consensus[sl] = y_sorted[consensus_order]
        idx += gs

    return consensus, group_slices


def make_feature_consensus_obj(y_train, X_train, groups, consensus_weight=0.3):
    """Feature Agreement MSE: blend target with feature-consensus signal.

    consensus_weight: 0 = pure MSE, 1 = pure consensus.
    """
    consensus, group_slices = _compute_consensus(y_train, X_train, groups)

    def obj(preds, dataset):
        labels = dataset.get_label()
        grad_target = preds - labels
        grad_consensus = preds - consensus
        grad = (1.0 - consensus_weight) * grad_target + consensus_weight * grad_consensus
        hess = np.ones_like(grad)
        return grad, hess

    return obj


def make_consensus_listnet_obj(y_train, X_train, groups, temperature=2.0,
                                consensus_weight=0.6, mse_weight=0.3):
    """Feature Consensus + ListNet: listwise softmax with consensus-blended target.

    Target distribution = softmax(blended / T) where blended = w*consensus + (1-w)*y.
    The consensus denoises the target ranking, ListNet provides smooth ranking loss,
    and an MSE anchor toward the blended target keeps predictions calibrated.

    This combines:
    - Feature Consensus's denoising power (winner from previous benchmark)
    - ListNet's O(n) listwise gradient (vs LambdaRank's O(n²) pairwise)
    - MSE anchor for valid RMSE
    """
    consensus, group_slices = _compute_consensus(y_train, X_train, groups)
    blended = (1.0 - consensus_weight) * y_train + consensus_weight * consensus

    def _softmax(x, T):
        x_s = x / T
        x_s = x_s - x_s.max()
        e = np.exp(x_s)
        return e / (e.sum() + 1e-10)

    def obj(preds, dataset):
        grad_list = np.zeros_like(preds)
        for sl in group_slices:
            b_g = blended[sl]
            p_g = preds[sl]
            if len(b_g) < 3:
                continue
            p_target = _softmax(b_g, temperature)
            p_pred = _softmax(p_g, temperature)
            grad_list[sl] = (p_pred - p_target) / temperature

        # MSE anchor toward blended target
        grad_mse = preds - blended

        # Scale matching
        rms_mse = np.sqrt(np.mean(grad_mse ** 2)) + 1e-10
        rms_list = np.sqrt(np.mean(grad_list ** 2)) + 1e-10
        grad_list = grad_list * (rms_mse / rms_list)

        grad = mse_weight * grad_mse + (1.0 - mse_weight) * grad_list
        hess = np.ones_like(grad)
        return grad, hess

    return obj


def make_consensus_lambdarank_obj(y_train, X_train, groups, mse_weight=0.3,
                                   consensus_weight=0.6):
    """Feature Consensus + LambdaRank: NDCG ranking on consensus ordering + MSE anchor.

    Key difference from plain LambdaRank: pair ordering based on CONSENSUS
    (what features collectively say) rather than raw y (noisy realized return).
    MSE anchor toward consensus keeps predictions calibrated.
    """
    consensus, group_slices = _compute_consensus(y_train, X_train, groups)
    sigma = 1.0

    # Discretize consensus for LambdaRank pair selection
    disc_labels = np.empty(len(y_train))
    for sl in group_slices:
        v = consensus[sl]
        disc_labels[sl] = np.digitize(v, np.percentile(v, [20, 40, 60, 80]))

    def obj(preds, dataset):
        grad_rank = np.zeros_like(preds)
        for sl in group_slices:
            g_lab = disc_labels[sl]
            g_pred = preds[sl]
            g_size = len(g_lab)
            if g_size < 3:
                continue

            sorted_idx = np.argsort(-g_pred)
            ranks = np.empty(g_size, dtype=np.int64)
            ranks[sorted_idx] = np.arange(g_size)

            ideal_order = np.sort(g_lab)[::-1]
            idcg = _compute_dcg(ideal_order)
            if idcg < 1e-12:
                continue

            lab_i, lab_j = np.meshgrid(g_lab, g_lab, indexing="ij")
            pred_i, pred_j = np.meshgrid(g_pred, g_pred, indexing="ij")
            rank_i, rank_j = np.meshgrid(ranks, ranks, indexing="ij")

            mask = lab_i > lab_j
            delta_s = pred_i - pred_j

            with np.errstate(over="ignore"):
                exp_val = np.exp(sigma * delta_s)
            p_lambda = 1.0 / (1.0 + exp_val)

            gain_diff = np.abs(2.0 ** lab_i - 2.0 ** lab_j)
            disc_diff = np.abs(
                1.0 / np.log2(rank_i + 2.0) - 1.0 / np.log2(rank_j + 2.0)
            )
            delta_ndcg = (gain_diff * disc_diff) / idcg

            lam = -sigma * p_lambda * delta_ndcg * mask
            grad_rank[sl] = lam.sum(axis=1) - lam.sum(axis=0)

        # MSE toward consensus (not raw y)
        grad_mse = preds - consensus

        rms_mse = np.sqrt(np.mean(grad_mse ** 2)) + 1e-10
        rms_rank = np.sqrt(np.mean(grad_rank ** 2)) + 1e-10
        grad_rank_scaled = grad_rank * (rms_mse / rms_rank)

        grad = mse_weight * grad_mse + (1.0 - mse_weight) * grad_rank_scaled
        hess = np.ones_like(grad)
        return grad, hess

    return obj


def make_lambdarank_v2_obj(y_train, groups,
                            spread_penalty=0.01,
                            topk_boost=2.0,
                            smoothness_penalty=0.005,
                            mse_anchor=0.0):
    """LambdaRank with multiple penalty/boost terms.

    Penalties on top of standard LambdaRank:
    1. Spread: L2 on (pred - group_mean) — prevents score explosion
    2. Top-K boost: extra weight on top/bottom 30% items — P&L focus
    3. Smoothness: penalize large gaps between adjacent-ranked predictions
    4. MSE anchor: optional MSE toward raw y for point prediction calibration

    These address the main issues with vanilla LambdaRank:
    - Unbounded scores (spread penalty)
    - Equal weight on all positions (top-K boost)
    - Noisy score jumps (smoothness)
    """
    y = y_train.copy()
    sigma = 1.0

    group_slices = []
    disc_labels = np.empty_like(y)
    idx = 0
    for gs in groups:
        sl = slice(idx, idx + gs)
        group_slices.append(sl)
        v = y[sl]
        disc_labels[sl] = np.digitize(v, np.percentile(v, [20, 40, 60, 80]))
        idx += gs

    def obj(preds, dataset):
        grad_rank = np.zeros_like(preds)
        grad_spread = np.zeros_like(preds)
        grad_smooth = np.zeros_like(preds)

        for sl in group_slices:
            g_lab = disc_labels[sl]
            g_pred = preds[sl]
            g_size = len(g_lab)
            if g_size < 3:
                continue

            sorted_idx = np.argsort(-g_pred)
            ranks = np.empty(g_size, dtype=np.int64)
            ranks[sorted_idx] = np.arange(g_size)

            ideal_order = np.sort(g_lab)[::-1]
            idcg = _compute_dcg(ideal_order)
            if idcg < 1e-12:
                continue

            lab_i, lab_j = np.meshgrid(g_lab, g_lab, indexing="ij")
            pred_i, pred_j = np.meshgrid(g_pred, g_pred, indexing="ij")
            rank_i, rank_j = np.meshgrid(ranks, ranks, indexing="ij")

            mask = lab_i > lab_j
            delta_s = pred_i - pred_j

            with np.errstate(over="ignore"):
                exp_val = np.exp(sigma * delta_s)
            p_lambda = 1.0 / (1.0 + exp_val)

            gain_diff = np.abs(2.0 ** lab_i - 2.0 ** lab_j)
            disc_diff = np.abs(
                1.0 / np.log2(rank_i + 2.0) - 1.0 / np.log2(rank_j + 2.0)
            )
            delta_ndcg = (gain_diff * disc_diff) / idcg

            # Top-K boost: extra weight on pairs involving top/bottom 30%
            n_topk = max(1, int(g_size * 0.3))
            topk_mask = (ranks < n_topk) | (ranks >= g_size - n_topk)
            topk_i = topk_mask[np.arange(g_size)[:, None].repeat(g_size, axis=1)]
            topk_j = topk_mask[np.arange(g_size)[None, :].repeat(g_size, axis=0)]
            topk_weight = np.where(topk_i | topk_j, topk_boost, 1.0)

            lam = -sigma * p_lambda * delta_ndcg * mask * topk_weight
            grad_rank[sl] = lam.sum(axis=1) - lam.sum(axis=0)

            # Spread penalty: L2 on deviation from group mean
            g_mean = g_pred.mean()
            grad_spread[sl] = spread_penalty * (g_pred - g_mean)

            # Smoothness: penalize large gaps between adjacent predictions
            order = np.argsort(g_pred)
            pred_sorted = g_pred[order]
            gaps = np.diff(pred_sorted)
            gs_sorted = np.zeros(g_size)
            gs_sorted[:-1] += smoothness_penalty * gaps
            gs_sorted[1:] -= smoothness_penalty * gaps
            inv_order = np.argsort(order)
            grad_smooth[sl] = gs_sorted[inv_order]

        grad = grad_rank + grad_spread + grad_smooth

        # Optional MSE anchor
        if mse_anchor > 0:
            grad_mse = preds - y
            rms_mse = np.sqrt(np.mean(grad_mse ** 2)) + 1e-10
            rms_rank = np.sqrt(np.mean(grad ** 2)) + 1e-10
            grad = grad + mse_anchor * grad_mse * (rms_rank / rms_mse)

        hess = np.ones_like(grad)
        return grad, hess

    return obj


def make_penalized_lambdarank_obj(y_train, groups, mse_weight=0.3, l2_pred=0.001):
    """LambdaRank + MSE anchor + L2 prediction penalty.

    Combines:
    1. NDCG-weighted LambdaRank gradient for ranking quality
    2. MSE gradient toward actual returns for point prediction accuracy
    3. L2 penalty on prediction magnitude for stability

    This produces predictions that are both well-ranked (high Spearman/Pearson)
    and calibrated to actual return scale (valid RMSE).
    """
    y = y_train.copy()
    sigma = 1.0

    # Pre-compute group slices and discretized labels for LambdaRank
    group_slices = []
    disc_labels = np.empty_like(y)
    idx = 0
    for gs in groups:
        sl = slice(idx, idx + gs)
        group_slices.append(sl)
        v = y[sl]
        disc_labels[sl] = np.digitize(v, np.percentile(v, [20, 40, 60, 80]))
        idx += gs

    def obj(preds, dataset):
        # --- LambdaRank component ---
        grad_rank = np.zeros_like(preds)
        for sl in group_slices:
            g_lab = disc_labels[sl]
            g_pred = preds[sl]
            g_size = len(g_lab)
            if g_size < 3:
                continue

            sorted_idx = np.argsort(-g_pred)
            ranks = np.empty(g_size, dtype=np.int64)
            ranks[sorted_idx] = np.arange(g_size)

            ideal_order = np.sort(g_lab)[::-1]
            idcg = _compute_dcg(ideal_order)
            if idcg < 1e-12:
                continue

            lab_i, lab_j = np.meshgrid(g_lab, g_lab, indexing="ij")
            pred_i, pred_j = np.meshgrid(g_pred, g_pred, indexing="ij")
            rank_i, rank_j = np.meshgrid(ranks, ranks, indexing="ij")

            mask = lab_i > lab_j
            delta_s = pred_i - pred_j

            with np.errstate(over="ignore"):
                exp_val = np.exp(sigma * delta_s)
            p_lambda = 1.0 / (1.0 + exp_val)

            gain_diff = np.abs(2.0 ** lab_i - 2.0 ** lab_j)
            disc_diff = np.abs(
                1.0 / np.log2(rank_i + 2.0) - 1.0 / np.log2(rank_j + 2.0)
            )
            delta_ndcg = (gain_diff * disc_diff) / idcg

            lam = -sigma * p_lambda * delta_ndcg * mask
            grad_rank[sl] = lam.sum(axis=1) - lam.sum(axis=0)

        # --- MSE component ---
        grad_mse = preds - y

        # --- L2 prediction penalty ---
        grad_l2 = l2_pred * preds

        # Normalize ranking gradient to match MSE gradient scale
        rms_mse = np.sqrt(np.mean(grad_mse ** 2)) + 1e-10
        rms_rank = np.sqrt(np.mean(grad_rank ** 2)) + 1e-10
        grad_rank_scaled = grad_rank * (rms_mse / rms_rank)

        grad = mse_weight * grad_mse + (1.0 - mse_weight) * grad_rank_scaled + grad_l2
        hess = np.ones_like(grad)
        return grad, hess

    return obj


def make_listnet_obj(y_train, groups, temperature=1.0):
    """Listwise softmax ranking (ListNet).

    Cross-entropy between target and predicted rank probability distributions:
        P(y) = softmax(y/T),  P(ŷ) = softmax(ŷ/T)
        L = -Σ P(y_i) * log P(ŷ_i)
        grad_i = (P_ŷ_i - P_y_i) / T

    Advantages over pairwise LambdaRank: O(n) gradient per group (not O(n²)),
    naturally handles ties, smoother optimization landscape.
    """
    y = y_train.copy()

    group_slices = []
    idx = 0
    for gs in groups:
        group_slices.append(slice(idx, idx + gs))
        idx += gs

    def _softmax(x, T):
        x_s = x / T
        x_s = x_s - x_s.max()
        e = np.exp(x_s)
        return e / (e.sum() + 1e-10)

    def obj(preds, dataset):
        grad = np.zeros_like(preds)

        for sl in group_slices:
            y_g = y[sl]
            p_g = preds[sl]
            if len(y_g) < 3:
                continue
            p_y = _softmax(y_g, temperature)
            p_pred = _softmax(p_g, temperature)
            grad[sl] = (p_pred - p_y) / temperature

        hess = np.ones_like(grad)
        return grad, hess

    return obj


def make_trimmed_mse_obj(y_train, groups, trim_quantile=0.4):
    """Trimmed MSE: only backprop on extreme tokens per date group.

    Only the top and bottom tokens matter for P&L (long/short).
    Middle tokens are "don't care" — skipping them focuses model capacity
    on distinguishing the tails, which is exactly what alpha generation needs.

    trim_quantile: fraction of tokens to keep from each tail (0.3 = top 30% + bottom 30%).
    """
    y = y_train.copy()

    group_slices = []
    idx = 0
    for gs in groups:
        group_slices.append(slice(idx, idx + gs))
        idx += gs

    def obj(preds, dataset):
        labels = dataset.get_label()
        grad = np.zeros_like(preds)

        for sl in group_slices:
            y_g = labels[sl]
            p_g = preds[sl]
            n_g = len(y_g)
            if n_g < 5:
                grad[sl] = p_g - y_g
                continue

            # Select top/bottom by target rank
            ranks = np.argsort(np.argsort(y_g))
            n_keep = max(1, int(n_g * trim_quantile))
            mask = (ranks < n_keep) | (ranks >= n_g - n_keep)

            grad_g = np.zeros(n_g)
            grad_g[mask] = p_g[mask] - y_g[mask]
            grad[sl] = grad_g

        hess = np.ones_like(grad)
        return grad, hess

    return obj


def make_focal_mse_obj(gamma=2.0):
    """Focal MSE: upweight samples with large cross-sectional residuals.

    Inspired by focal loss: modulate gradient by |residual|^gamma.
    Samples where the model is already accurate get lower weight,
    focusing capacity on "hard" tokens (often the extremes that drive P&L).

    Unlike trimmed MSE, this uses soft weighting — all samples contribute
    but hard ones contribute disproportionately more.
    """
    def obj(preds, dataset):
        labels = dataset.get_label()
        r = labels - preds
        abs_r = np.abs(r) + 1e-8
        focal_weight = (abs_r / (np.median(abs_r) + 1e-8)) ** gamma
        focal_weight = np.clip(focal_weight, 0.01, 10.0)
        grad = -focal_weight * r
        hess = np.ones_like(grad)
        return grad, hess

    return obj


# --- LambdaRank for ranking ---

def train_erd_lambdarank(X_train, labels_train, groups_train, X_test, fobj, n_rounds=200):
    """Train ranking model. Labels = discretized p1d relevance grades."""
    ds = lgb.Dataset(X_train, label=labels_train, group=groups_train, free_raw_data=False)
    params = {**LGB_PARAMS_BASE, "objective": fobj}
    model = lgb.train(params, ds, num_boost_round=n_rounds)
    return model.predict(X_test)


def run_erd_benchmark():
    """Run all models on real erdNov16Prod.csv data."""
    print("=" * 80)
    print("=== Real Data Benchmark: erdNov16Prod.csv (mr* → p1d) ===")
    print("=" * 80 + "\n")

    data = load_erd_data()
    X_tr, X_te = data["X_train"], data["X_test"]
    Y_tr, Y_te = data["Y_train"], data["Y_test"]      # (n, 3): p1d, p3d, p5d
    y_tr, y_te = data["y_train"], data["y_test"]        # p1d only
    dates_te = data["dates_test"]
    k = data["k"]
    n_train = data["n_train"]

    # Covariance of multi-horizon targets (for Mahalanobis/Cholesky)
    Sigma = np.cov(Y_tr.T)
    P = np.linalg.inv(Sigma)

    # Quantile init
    tau = 0.7
    q_init_single = np.full(len(y_tr), np.quantile(y_tr, tau))
    q_init_multi = np.broadcast_to(
        np.quantile(Y_tr, tau, axis=0), (n_train, k)
    ).copy()

    # Discretized labels for ranking (quintile grades 0-4)
    def discretize_by_group(vals, groups):
        result = np.empty_like(vals)
        idx = 0
        for gs in groups:
            sl = slice(idx, idx + gs)
            v = vals[sl]
            result[sl] = np.digitize(v, np.percentile(v, [20, 40, 60, 80]))
            idx += gs
        return result.astype(np.float64)

    rank_labels_tr = discretize_by_group(y_tr, data["train_groups"])
    rank_labels_te = discretize_by_group(y_te, data["test_groups"])

    print()
    results = []

    # ------------------------------------------------------------------
    # Case 0: Vanilla LightGBM (baseline)
    # ------------------------------------------------------------------
    t0 = time.time()
    pred0 = train_single_vanilla(X_tr, y_tr, X_te)
    t0e = time.time() - t0
    results.append(("0. Vanilla LGBM", eval_erd(y_te, pred0, dates_te), t0e))
    print(f"  [done] 0. Vanilla LGBM ({t0e:.1f}s)")

    # ------------------------------------------------------------------
    # Case 1: Custom MSE (single output — verifies custom obj pipeline)
    # ------------------------------------------------------------------
    t0 = time.time()
    pred1 = train_single_custom(X_tr, y_tr, X_te, make_single_mse_obj())
    t1e = time.time() - t0
    results.append(("1. Custom MSE", eval_erd(y_te, pred1, dates_te), t1e))
    print(f"  [done] 1. Custom MSE ({t1e:.1f}s)")

    # ------------------------------------------------------------------
    # Case 2: Mahalanobis multi-output (p1d + p3d + p5d), eval on p1d
    # ------------------------------------------------------------------
    t0 = time.time()
    fobj2 = make_mahalanobis_obj(Y_tr, P)
    pred2_multi = train_multioutput_custom(
        {"X_train": X_tr, "X_test": X_te, "Y_train": Y_tr,
         "n_train": n_train, "k": k},
        fobj2,
    )
    pred2 = pred2_multi[:, 0]  # p1d column
    t2e = time.time() - t0
    results.append(("2. Mahalanobis (p1d+p3d+p5d)", eval_erd(y_te, pred2, dates_te), t2e))
    print(f"  [done] 2. Mahalanobis ({t2e:.1f}s)")

    # ------------------------------------------------------------------
    # Case 3: Asymmetric Quantile (tau=0.7)
    # ------------------------------------------------------------------
    t0 = time.time()
    pred3 = train_single_custom(
        X_tr, y_tr, X_te, make_single_quantile_obj(tau=tau),
        init_score=q_init_single,
        extra_params={"learning_rate": 0.02, "min_data_in_leaf": 50, "num_leaves": 15},
    )
    t3e = time.time() - t0
    results.append(("3. Quantile (tau=0.7)", eval_erd(y_te, pred3, dates_te), t3e))
    print(f"  [done] 3. Quantile ({t3e:.1f}s)")

    # ------------------------------------------------------------------
    # Case 4: Shared-Norm multi-output (p1d + p3d + p5d), eval on p1d
    # ------------------------------------------------------------------
    t0 = time.time()
    fobj4 = make_shared_norm_obj(Y_tr, lam=0.05)
    pred4_multi = train_multioutput_custom(
        {"X_train": X_tr, "X_test": X_te, "Y_train": Y_tr,
         "n_train": n_train, "k": k},
        fobj4,
    )
    pred4 = pred4_multi[:, 0]
    t4e = time.time() - t0
    results.append(("4. Shared-Norm (p1d+p3d+p5d)", eval_erd(y_te, pred4, dates_te), t4e))
    print(f"  [done] 4. Shared-Norm ({t4e:.1f}s)")

    # ------------------------------------------------------------------
    # Case 5: Cholesky multi-output (p1d + p3d + p5d), eval on p1d
    # ------------------------------------------------------------------
    t0 = time.time()
    fobj5 = make_cholesky_obj(Y_tr, Sigma)
    pred5_multi = train_multioutput_custom(
        {"X_train": X_tr, "X_test": X_te, "Y_train": Y_tr,
         "n_train": n_train, "k": k},
        fobj5,
    )
    pred5 = pred5_multi[:, 0]
    t5e = time.time() - t0
    results.append(("5. Cholesky (p1d+p3d+p5d)", eval_erd(y_te, pred5, dates_te), t5e))
    print(f"  [done] 5. Cholesky ({t5e:.1f}s)")

    # ------------------------------------------------------------------
    # Case 3b: Huber loss (robust to outlier returns)
    # ------------------------------------------------------------------
    # Huber delta = 1 sigma of p1d — clips outlier returns beyond 1 std
    huber_delta = np.std(y_tr)
    t0 = time.time()
    pred3b = train_single_custom(X_tr, y_tr, X_te, make_single_huber_obj(delta=huber_delta))
    t3be = time.time() - t0
    results.append((f"3b. Huber (d={huber_delta:.3f})", eval_erd(y_te, pred3b, dates_te), t3be))
    print(f"  [done] 3b. Huber (delta={huber_delta:.3f}) ({t3be:.1f}s)")

    # ------------------------------------------------------------------
    # Case 6a: Built-in LambdaRank
    # ------------------------------------------------------------------
    t0 = time.time()
    ds_rank = lgb.Dataset(
        X_tr, label=rank_labels_tr, group=data["train_groups"], free_raw_data=False,
    )
    model_rank = lgb.train(
        {**LGB_PARAMS_BASE, "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5]},
        ds_rank, num_boost_round=200,
    )
    pred6a = model_rank.predict(X_te)
    t6ae = time.time() - t0
    results.append(("6a. Built-in LambdaRank", eval_erd(y_te, pred6a, dates_te), t6ae))
    print(f"  [done] 6a. Built-in LambdaRank ({t6ae:.1f}s)")

    # ------------------------------------------------------------------
    # Case 6b: Custom LambdaRank
    # ------------------------------------------------------------------
    t0 = time.time()
    pred6b = train_erd_lambdarank(
        X_tr, rank_labels_tr, data["train_groups"], X_te, lambdarank_obj,
    )
    t6be = time.time() - t0
    results.append(("6b. Custom LambdaRank", eval_erd(y_te, pred6b, dates_te), t6be))
    print(f"  [done] 6b. Custom LambdaRank ({t6be:.1f}s)")

    # ------------------------------------------------------------------
    # Case 7: XS-Aware MSE (fine alpha sweep)
    # ------------------------------------------------------------------
    for alpha_val in [0.0, 0.05, 0.1, 0.15, 0.2]:
        label = f"7. XS-MSE (α={alpha_val})"
        t0 = time.time()
        fobj7 = make_xs_aware_mse_obj(y_tr, data["train_groups"], alpha=alpha_val)
        pred7 = train_single_custom(X_tr, y_tr, X_te, fobj7)
        t7e = time.time() - t0
        results.append((label, eval_erd(y_te, pred7, dates_te), t7e))
        print(f"  [done] {label} ({t7e:.1f}s)")

    # ------------------------------------------------------------------
    # Case 8: XS-Corr (alpha/beta sweep)
    # ------------------------------------------------------------------
    for alpha_val, beta_val in [(0.3, 0.5), (0.1, 0.3), (0.0, 0.5), (0.0, 0.0)]:
        label = f"8. XS-Corr (α={alpha_val},β={beta_val})"
        t0 = time.time()
        fobj8 = make_xs_corr_obj(y_tr, data["train_groups"], alpha=alpha_val, beta=beta_val)
        pred8 = train_single_custom(X_tr, y_tr, X_te, fobj8)
        t8e = time.time() - t0
        results.append((label, eval_erd(y_te, pred8, dates_te), t8e))
        print(f"  [done] {label} ({t8e:.1f}s)")

    # ------------------------------------------------------------------
    # Case 9: Hybrid MSE-Rank (Custom MSE + mild LambdaRank gradient)
    # Idea: Use gradient-descent MSE as base (which gives good Spearman),
    # then blend in a small LambdaRank gradient for ranking push.
    # ------------------------------------------------------------------
    for rank_weight in [0.1, 0.3, 0.5]:
        label = f"9. MSE+Rank (w={rank_weight})"
        t0 = time.time()
        fobj9 = make_mse_rank_hybrid_obj(y_tr, data["train_groups"], rank_weight=rank_weight)
        pred9 = train_single_custom(X_tr, y_tr, X_te, fobj9)
        t9e = time.time() - t0
        results.append((label, eval_erd(y_te, pred9, dates_te), t9e))
        print(f"  [done] {label} ({t9e:.1f}s)")

    # ------------------------------------------------------------------
    # Case 10: Feature Consensus MSE
    # ------------------------------------------------------------------
    for cw in [0.2, 0.4, 0.6]:
        label = f"10. FeatureConsensus (w={cw})"
        t0 = time.time()
        fobj10 = make_feature_consensus_obj(y_tr, X_tr, data["train_groups"], consensus_weight=cw)
        pred10 = train_single_custom(X_tr, y_tr, X_te, fobj10)
        t10e = time.time() - t0
        results.append((label, eval_erd(y_te, pred10, dates_te), t10e))
        print(f"  [done] {label} ({t10e:.1f}s)")

    # ------------------------------------------------------------------
    # Case 11: Penalized LambdaRank + MSE (ranking + point prediction)
    # ------------------------------------------------------------------
    for mw in [0.2, 0.4, 0.6]:
        label = f"11. PenalizedLRank (mse={mw})"
        t0 = time.time()
        fobj11 = make_penalized_lambdarank_obj(y_tr, data["train_groups"], mse_weight=mw)
        pred11 = train_single_custom(X_tr, y_tr, X_te, fobj11)
        t11e = time.time() - t0
        results.append((label, eval_erd(y_te, pred11, dates_te), t11e))
        print(f"  [done] {label} ({t11e:.1f}s)")

    # ------------------------------------------------------------------
    # Case 12: ListNet (listwise softmax ranking)
    # ------------------------------------------------------------------
    for temp in [0.5, 1.0, 2.0]:
        label = f"12. ListNet (T={temp})"
        t0 = time.time()
        fobj12 = make_listnet_obj(y_tr, data["train_groups"], temperature=temp)
        pred12 = train_single_custom(X_tr, y_tr, X_te, fobj12)
        t12e = time.time() - t0
        results.append((label, eval_erd(y_te, pred12, dates_te), t12e))
        print(f"  [done] {label} ({t12e:.1f}s)")

    # ------------------------------------------------------------------
    # Case 13: Trimmed MSE (focus on extreme tokens)
    # ------------------------------------------------------------------
    for tq in [0.3, 0.4, 0.5]:
        label = f"13. TrimmedMSE (q={tq})"
        t0 = time.time()
        fobj13 = make_trimmed_mse_obj(y_tr, data["train_groups"], trim_quantile=tq)
        pred13 = train_single_custom(X_tr, y_tr, X_te, fobj13)
        t13e = time.time() - t0
        results.append((label, eval_erd(y_te, pred13, dates_te), t13e))
        print(f"  [done] {label} ({t13e:.1f}s)")

    # ------------------------------------------------------------------
    # Case 14: Focal MSE (hard-sample weighting)
    # ------------------------------------------------------------------
    for gamma in [0.5, 1.0, 2.0]:
        label = f"14. FocalMSE (γ={gamma})"
        t0 = time.time()
        fobj14 = make_focal_mse_obj(gamma=gamma)
        pred14 = train_single_custom(X_tr, y_tr, X_te, fobj14)
        t14e = time.time() - t0
        results.append((label, eval_erd(y_te, pred14, dates_te), t14e))
        print(f"  [done] {label} ({t14e:.1f}s)")

    # ------------------------------------------------------------------
    # Case 15: Consensus + ListNet
    # ------------------------------------------------------------------
    for cw, mw in [(0.6, 0.3), (0.6, 0.1), (0.8, 0.2)]:
        label = f"15. Cons+ListNet (cw={cw},mw={mw})"
        t0 = time.time()
        fobj15 = make_consensus_listnet_obj(
            y_tr, X_tr, data["train_groups"],
            temperature=2.0, consensus_weight=cw, mse_weight=mw)
        pred15 = train_single_custom(X_tr, y_tr, X_te, fobj15)
        t15e = time.time() - t0
        results.append((label, eval_erd(y_te, pred15, dates_te), t15e))
        print(f"  [done] {label} ({t15e:.1f}s)")

    # ------------------------------------------------------------------
    # Case 16: Consensus + LambdaRank
    # ------------------------------------------------------------------
    for cw, mw in [(0.6, 0.3), (0.6, 0.1), (0.8, 0.2)]:
        label = f"16. Cons+LRank (cw={cw},mw={mw})"
        t0 = time.time()
        fobj16 = make_consensus_lambdarank_obj(
            y_tr, X_tr, data["train_groups"],
            mse_weight=mw, consensus_weight=cw)
        pred16 = train_single_custom(X_tr, y_tr, X_te, fobj16)
        t16e = time.time() - t0
        results.append((label, eval_erd(y_te, pred16, dates_te), t16e))
        print(f"  [done] {label} ({t16e:.1f}s)")

    # ------------------------------------------------------------------
    # Case 17: Penalized LambdaRank v2 (spread + topK + smoothness)
    # ------------------------------------------------------------------
    for topk, mse_a in [(2.0, 0.0), (3.0, 0.0), (2.0, 0.3), (3.0, 0.3)]:
        label = f"17. LRankV2 (topK={topk},mse={mse_a})"
        t0 = time.time()
        fobj17 = make_lambdarank_v2_obj(
            y_tr, data["train_groups"],
            spread_penalty=0.01, topk_boost=topk,
            smoothness_penalty=0.005, mse_anchor=mse_a)
        pred17 = train_single_custom(X_tr, y_tr, X_te, fobj17)
        t17e = time.time() - t0
        results.append((label, eval_erd(y_te, pred17, dates_te), t17e))
        print(f"  [done] {label} ({t17e:.1f}s)")

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print(f"\n{'Case':<35s}  {'RMSE':>8s}  {'Spearman':>9s}  {'Pearson':>9s}  "
          f"{'Sp.Std':>7s}  {'Pe.Std':>7s}  {'Time':>6s}")
    print("-" * 93)

    baseline = results[0][1]
    for name, m, t in results:
        is_ranker = name.startswith("6")
        d_rmse = (m["rmse"] / baseline["rmse"] - 1) * 100
        d_sp = (m["spearman_mean"] / baseline["spearman_mean"] - 1) * 100 if baseline["spearman_mean"] != 0 else 0
        d_pe = (m["pearson_mean"] / baseline["pearson_mean"] - 1) * 100 if baseline["pearson_mean"] != 0 else 0
        if name.startswith("0."):
            tag = ""
        elif is_ranker:
            tag = f" (sp {d_sp:+.1f}%)"
        else:
            tag = f" (rmse {d_rmse:+.1f}%, sp {d_sp:+.1f}%, pe {d_pe:+.1f}%)"

        rmse_str = "    N/A " if is_ranker else f"{m['rmse']:>8.6f}"
        print(
            f"{name:<35s}  {rmse_str}  {m['spearman_mean']:>9.4f}  "
            f"{m['pearson_mean']:>9.4f}  {m['spearman_std']:>7.4f}  "
            f"{m['pearson_std']:>7.4f}  {t:>5.1f}s{tag}"
        )

    print("\n  Note: RMSE is N/A for ranking models (scores ≠ returns).")
    print(f"  XS correlations are per-date with ~10 tokens — high variance is expected.")

    # ------------------------------------------------------------------
    # Stability check: top candidates across multiple LightGBM seeds
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("=== Stability Check (5 seeds) ===")
    print(f"{'='*80}\n")

    # Pick top candidates from above + new objectives for stability check
    candidates = {
        "0. Vanilla LGBM": lambda: train_single_vanilla(X_tr, y_tr, X_te),
        "1. Custom MSE": lambda: train_single_custom(X_tr, y_tr, X_te, make_single_mse_obj()),
        "7. XS-MSE α=0.0": lambda: train_single_custom(
            X_tr, y_tr, X_te,
            make_xs_aware_mse_obj(y_tr, data["train_groups"], alpha=0.0)),
    }

    # Dynamically pick best from each new case family based on single-run Pearson
    def _best_from_results(prefix):
        matches = [(n, m, t) for n, m, t in results if n.startswith(prefix)]
        if not matches:
            return None, None
        best = max(matches, key=lambda x: x[1]["pearson_mean"])
        return best[0], best

    # Auto-pick best from each case family by single-run Pearson
    family_builders = {
        "10.": lambda name: (
            lambda _cw=float(name.split("w=")[1].rstrip(")")): train_single_custom(
                X_tr, y_tr, X_te,
                make_feature_consensus_obj(y_tr, X_tr, data["train_groups"], consensus_weight=_cw))
        ),
        "11.": lambda name: (
            lambda _mw=float(name.split("mse=")[1].rstrip(")")): train_single_custom(
                X_tr, y_tr, X_te,
                make_penalized_lambdarank_obj(y_tr, data["train_groups"], mse_weight=_mw))
        ),
        "12.": lambda name: (
            lambda _t=float(name.split("T=")[1].rstrip(")")): train_single_custom(
                X_tr, y_tr, X_te,
                make_listnet_obj(y_tr, data["train_groups"], temperature=_t))
        ),
        "15.": lambda name: (
            lambda: train_single_custom(
                X_tr, y_tr, X_te,
                make_consensus_listnet_obj(
                    y_tr, X_tr, data["train_groups"],
                    temperature=2.0,
                    consensus_weight=float(name.split("cw=")[1].split(",")[0]),
                    mse_weight=float(name.split("mw=")[1].rstrip(")"))))
        ),
        "16.": lambda name: (
            lambda: train_single_custom(
                X_tr, y_tr, X_te,
                make_consensus_lambdarank_obj(
                    y_tr, X_tr, data["train_groups"],
                    mse_weight=float(name.split("mw=")[1].rstrip(")")),
                    consensus_weight=float(name.split("cw=")[1].split(",")[0])))
        ),
        "17.": lambda name: (
            lambda: train_single_custom(
                X_tr, y_tr, X_te,
                make_lambdarank_v2_obj(
                    y_tr, data["train_groups"],
                    topk_boost=float(name.split("topK=")[1].split(",")[0]),
                    mse_anchor=float(name.split("mse=")[1].rstrip(")"))))
        ),
    }

    for prefix, builder in family_builders.items():
        best_name, _ = _best_from_results(prefix)
        if best_name is None:
            continue
        candidates[best_name] = builder(best_name)

    seed_params = [
        {"seed": s, "bagging_seed": s, "feature_fraction_seed": s}
        for s in range(5)
    ]

    col_w = 35
    print(f"{'Model':<{col_w}s}  {'RMSE (avg±std)':>18s}  {'Spearman (avg±std)':>20s}  {'Pearson (avg±std)':>20s}")
    print("-" * (col_w + 65))

    for model_name, train_fn in candidates.items():
        rmses, sps, pes = [], [], []
        for sp in seed_params:
            # Inject seed into base params temporarily
            old_vals = {}
            for k, v in sp.items():
                old_vals[k] = LGB_PARAMS_BASE.get(k)
                LGB_PARAMS_BASE[k] = v

            preds = train_fn()

            # Restore
            for k, v in old_vals.items():
                if v is None:
                    LGB_PARAMS_BASE.pop(k, None)
                else:
                    LGB_PARAMS_BASE[k] = v

            m = eval_erd(y_te, preds, dates_te)
            rmses.append(m["rmse"])
            sps.append(m["spearman_mean"])
            pes.append(m["pearson_mean"])

        rmses, sps, pes = np.array(rmses), np.array(sps), np.array(pes)
        print(f"{model_name:<{col_w}s}  {rmses.mean():.6f}±{rmses.std():.6f}  "
              f"{sps.mean():>8.4f}±{sps.std():.4f}     "
              f"{pes.mean():>8.4f}±{pes.std():.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AwareLGBM multi-output benchmark")
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-only", action="store_true", help="Skip real data benchmark")
    parser.add_argument("--real-only", action="store_true", help="Only run real data benchmark")
    args = parser.parse_args()

    if not args.real_only:
        run_multioutput_benchmark(n=args.n_samples, d=args.d, k=args.k, seed=args.seed)
        run_ranking_benchmark(seed=args.seed)

    if not args.synthetic_only:
        if os.path.exists(ERD_PATH):
            run_erd_benchmark()
        else:
            print(f"\nSkipping real data benchmark: {ERD_PATH} not found")
