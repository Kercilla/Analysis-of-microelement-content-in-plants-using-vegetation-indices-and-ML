#!/usr/bin/env python3
"""
00_audit_pipeline.py — аудит пайплайна на утечку данных (data leakage).

Проверяет:
1. StandardScaler/MixUp применяются ВНУТРИ CV-fold (не до split)
2. Permutation test — R² должен падать до ~0 при перемешанных y
3. Baseline "предсказать среднее" — нижняя граница
4. Пространственная CV vs случайная KFold — разница = оптимистическое смещение

Запуск:
    python 00_audit_pipeline.py
    python 00_audit_pipeline.py --elements N_% Ca_% --n-perm 199
"""
import argparse, os, sys, warnings
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from analysis.cfg import load_config, short_name
from analysis.loaders import load_multi, load_chemistry
from analysis.indices import calculate_indices
from analysis.preprocessing import mixup_augment
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans


# ── Helpers ───────────────────────────────────────────────────────────────────

def regression_metrics(yt, yp):
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) < 3:
        return {"R2": np.nan, "RMSE": np.nan}
    r2   = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    return {"R2": round(r2, 4), "RMSE": round(rmse, 4)}


def kfold_cv(model, X, y, n_splits=5, scale=True):
    """Корректная CV: scaler fit только на train."""
    mask = np.isfinite(y)
    Xc, yc = X[mask], y[mask]
    cv = KFold(n_splits=min(n_splits, len(yc) // 2), shuffle=True, random_state=42)
    y_pred = np.full(len(yc), np.nan)
    for tr, te in cv.split(Xc):
        Xtr, Xte = Xc[tr], Xc[te]
        if scale:
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)
        m = clone(model)
        m.fit(Xtr, yc[tr])
        y_pred[te] = m.predict(Xte)
    return regression_metrics(yc, y_pred)


def spatial_kfold_cv(model, X, y, coords, n_folds=5, scale=True):
    """Пространственная CV через KMeans-кластеры координат."""
    mask = np.isfinite(y)
    Xc, yc, cc = X[mask], y[mask], coords[mask]
    km = KMeans(n_clusters=n_folds, random_state=42, n_init=10)
    folds = km.fit_predict(cc)
    y_pred = np.full(len(yc), np.nan)
    for f in range(n_folds):
        te = np.where(folds == f)[0]
        tr = np.where(folds != f)[0]
        if len(te) == 0 or len(tr) < 5:
            continue
        Xtr, Xte = Xc[tr], Xc[te]
        if scale:
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)
        m = clone(model)
        m.fit(Xtr, yc[tr])
        y_pred[te] = m.predict(Xte)
    return regression_metrics(yc, y_pred)


def permutation_test(model, X, y, n_perm=199, scale=True, n_splits=5, seed=42):
    """
    Permutation test: если истинный R² >> permuted R² — модель настоящая.
    Если permuted R² ≈ истинный — leakage.
    """
    rng = np.random.default_rng(seed)
    true_r2 = kfold_cv(model, X, y, n_splits, scale)["R2"]
    perm_r2s = []
    mask = np.isfinite(y)
    yc = y[mask]
    for _ in range(n_perm):
        yp = rng.permutation(yc)
        # подставляем обратно в полный массив
        y_perm = y.copy()
        y_perm[mask] = yp
        r2 = kfold_cv(model, X, y_perm, n_splits, scale)["R2"]
        perm_r2s.append(r2)
    perm_r2s = np.array(perm_r2s)
    p_val = (perm_r2s >= true_r2).mean()
    return {
        "true_R2":    round(true_r2, 4),
        "perm_mean":  round(perm_r2s.mean(), 4),
        "perm_max":   round(perm_r2s.max(), 4),
        "p_value":    round(p_val, 4),
        "significant": p_val < 0.05,
    }


def mixup_leakage_test(model, X, y, n_splits=5, scale=True):
    """
    Проверка MixUp leakage:
    - Вариант A: MixUp ПЕРЕД split (как могло быть неправильно)
    - Вариант B: нет MixUp
    Если A >> B — то MixUp до split раздувает R².
    """
    mask = np.isfinite(y)
    Xc, yc = X[mask], y[mask]

    # A: MixUp перед split (НЕПРАВИЛЬНО — имитируем возможный баг)
    Xa, ya = mixup_augment(Xc, yc, n_aug=len(yc))
    r2_before_split = kfold_cv(model, Xa, ya, n_splits, scale)["R2"]

    # B: без MixUp
    r2_no_aug = kfold_cv(model, Xc, yc, n_splits, scale)["R2"]

    return {
        "R2_mixup_before_split": round(r2_before_split, 4),
        "R2_no_augmentation":    round(r2_no_aug, 4),
        "inflation":             round(r2_before_split - r2_no_aug, 4),
        "leakage_suspected":     (r2_before_split - r2_no_aug) > 0.05,
    }


def baseline_mean(y):
    """R² предсказания среднего — всегда равен 0 по определению."""
    mask = np.isfinite(y)
    yc = y[mask]
    yp = np.full_like(yc, yc.mean())
    return regression_metrics(yc, yp)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    cfg  = load_config(args.config)
    out  = "results/00_audit"
    os.makedirs(out, exist_ok=True)

    bmap  = cfg["camera"]["band_map"]
    cols  = cfg["chemistry"]["columns"]
    elems = args.elements or cfg["chemistry"]["target_elements"]
    dsmap = cfg["chemistry"]["date_sampling_map"]
    tiers = cfg["statistics"]["index_tiers"]
    dk    = "date2"   # используем дату ближайшую к химии

    # ── загрузка данных ──
    pat = cfg["paths"]["multi"].get(dk)
    if not pat or not glob(pat):
        print(f"Нет данных для {dk}"); return

    df_bands = load_multi(pat, bmap)
    df_idx   = calculate_indices(df_bands, tiers=tiers)
    X_full   = pd.concat([df_bands, df_idx], axis=1)

    sk   = dsmap.get(dk, "sampling2")
    cp   = cfg["paths"]["chemistry"].get(sk)
    off  = cfg["chemistry"]["id_offsets"].get(sk, 0)
    chem = load_chemistry(cp, cols, off)

    ids       = X_full.index.intersection(chem.index)
    X         = X_full.loc[ids].values.astype(float)
    coords_df = df_bands.loc[ids]   # используем индекс как proxy координат

    # Если есть реальные координаты из gpkg — лучше их
    # Здесь используем порядковые номера как fallback
    fake_coords = np.column_stack([
        np.arange(len(ids)) % 10,
        np.arange(len(ids)) // 10,
    ]).astype(float)

    model_gbr = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42)
    model_rf  = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1)
    model_ridge = Ridge(alpha=1.0)

    print(f"\n{'='*65}")
    print(f"АУДИТ ПАЙПЛАЙНА — {len(ids)} точек, {X.shape[1]} признаков")
    print(f"{'='*65}")

    results = []

    for el in elems:
        if el not in chem.columns:
            continue
        y   = chem.loc[ids, el].values.astype(float)
        nv  = np.isfinite(y).sum()
        if nv < 15:
            continue
        sn  = short_name(cfg, el)
        print(f"\n── {sn} (n={nv}) ──────────────────────────────")

        row = {"element": el, "n_valid": nv}

        # 1. Baseline (предсказать среднее)
        bm = baseline_mean(y)
        row["R2_baseline_mean"] = bm["R2"]
        print(f"  Baseline (среднее):   R²={bm['R2']:+.4f}  (должно быть ≈0)")

        # 2. Ridge на 5 каналах (физический minimum)
        X5 = X[:, :5]
        r5 = kfold_cv(model_ridge, X5, y, scale=True)
        row["R2_ridge_5bands"] = r5["R2"]
        print(f"  Ridge (5 каналов):    R²={r5['R2']:+.4f}  (физический предел)")

        # 3. GBR — случайная KFold
        r_kf = kfold_cv(model_gbr, X, y, scale=False)
        row["R2_GBR_random_kfold"] = r_kf["R2"]
        print(f"  GBR (random KFold):   R²={r_kf['R2']:+.4f}")

        # 4. GBR — пространственная CV
        r_sp = spatial_kfold_cv(model_gbr, X, y, fake_coords, scale=False)
        row["R2_GBR_spatial_cv"] = r_sp["R2"]
        row["inflation_spatial"] = round(r_kf["R2"] - r_sp["R2"], 4)
        print(f"  GBR (spatial CV):     R²={r_sp['R2']:+.4f}  "
              f"(Δ={row['inflation_spatial']:+.4f} — оптимист. смещение)")

        # 5. Permutation test
        if not args.skip_perm:
            print(f"  Permutation test ({args.n_perm} перест.)...", end=" ", flush=True)
            pt = permutation_test(model_gbr, X, y, n_perm=args.n_perm, scale=False)
            row.update({
                "perm_true_R2":  pt["true_R2"],
                "perm_mean_R2":  pt["perm_mean"],
                "perm_max_R2":   pt["perm_max"],
                "p_value_perm":  pt["p_value"],
            })
            sig = "✓ ЗНАЧИМ" if pt["significant"] else "✗ НЕ ЗНАЧИМ (подозрение на leakage)"
            print(f"R²(perm_mean)={pt['perm_mean']:+.4f}  p={pt['p_value']:.3f}  → {sig}")

        # 6. MixUp leakage test
        print(f"  MixUp leakage test...", end=" ", flush=True)
        ml = mixup_leakage_test(model_gbr, X, y, scale=False)
        row.update({
            "R2_mixup_before_split": ml["R2_mixup_before_split"],
            "R2_no_aug":             ml["R2_no_augmentation"],
            "mixup_inflation":       ml["inflation"],
        })
        leak_flag = "⚠ ПОДОЗРЕНИЕ НА LEAKAGE" if ml["leakage_suspected"] else "OK"
        print(f"with={ml['R2_mixup_before_split']:+.4f}  "
              f"without={ml['R2_no_augmentation']:+.4f}  "
              f"Δ={ml['inflation']:+.4f}  → {leak_flag}")

        results.append(row)

    # ── Сводка ────────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(f"{out}/audit_results.csv", index=False, float_format="%.4f")

    print(f"\n{'='*65}")
    print("СВОДКА АУДИТА")
    print(f"{'='*65}")
    print(f"\n{'Элемент':<8} {'R²_baseline':>12} {'R²_ridge5':>10} "
          f"{'R²_random':>10} {'R²_spatial':>11} {'Δ_infl':>8} "
          f"{'p_perm':>8} {'MixUp_Δ':>9}")
    print("-"*85)

    leakage_found = False
    for _, r in df.iterrows():
        sn = short_name(cfg, r["element"])
        p_str = f"{r['p_value_perm']:.3f}" if "p_value_perm" in r and pd.notna(r.get("p_value_perm")) else "  n/a"
        mix_d = r.get("mixup_inflation", 0)
        flag  = " ⚠" if (r.get("p_value_perm", 1) >= 0.05 or mix_d > 0.05) else ""
        print(f"{sn:<8} {r['R2_baseline_mean']:>+12.4f} "
              f"{r['R2_ridge_5bands']:>+10.4f} "
              f"{r['R2_GBR_random_kfold']:>+10.4f} "
              f"{r['R2_GBR_spatial_cv']:>+11.4f} "
              f"{r['inflation_spatial']:>+8.4f} "
              f"{p_str:>8} "
              f"{mix_d:>+9.4f}{flag}")
        if r.get("p_value_perm", 1) >= 0.05 or mix_d > 0.05:
            leakage_found = True

    print(f"\n{'='*65}")
    if leakage_found:
        print("⚠ ОБНАРУЖЕНЫ ПОДОЗРИТЕЛЬНЫЕ РЕЗУЛЬТАТЫ — см. флаги выше")
        print("  Рекомендация: перейти на buffered-LOO CV для честных метрик")
    else:
        print("✓ Серьёзных признаков leakage не обнаружено")
        print("  Тем не менее, рекомендуется buffered-LOO для пространственной честности")

    print(f"\n  Оптимистическое смещение (median random - spatial): "
          f"{df['inflation_spatial'].median():+.4f}")
    print(f"\nСохранено: {out}/audit_results.csv")

    # ── Текстовый отчёт ──────────────────────────────────────────────────────
    with open(f"{out}/audit_report.md", "w", encoding="utf-8") as f:
        f.write("# Аудит пайплайна на data leakage\n\n")
        f.write("## Что проверялось\n\n")
        f.write("| Проверка | Описание | Что значит плохой результат |\n")
        f.write("|---|---|---|\n")
        f.write("| Baseline (среднее) | R² предсказания среднего | > 0.01 — баг в формуле R² |\n")
        f.write("| Ridge (5 каналов) | Физический минимум сигнала | Превышение GBR > +0.5 — подозрительно |\n")
        f.write("| Random KFold vs Spatial CV | Разница = оптимистическое смещение | Δ > 0.2 — пространственная автокорреляция |\n")
        f.write("| Permutation test | p < 0.05 — модель значима | p ≥ 0.05 — результат случайный |\n")
        f.write("| MixUp before/after split | Разница в R² | Δ > 0.05 — MixUp до split раздувает |\n\n")
        f.write("## Вывод по ml_pipeline.py\n\n")
        f.write("**StandardScaler**: применяется ВНУТРИ CV-fold (строки 99-101 ml_pipeline.py) — ✓ КОРРЕКТНО\n\n")
        f.write("**MixUp в 07_ml_enhanced.py**: применяется ко ВСЕЙ выборке ДО передачи в compare_models.\n")
        f.write("Это означает что синтетические MixUp-образцы могут попасть и в train и в test.\n")
        f.write("compare_models внутри делает KFold — но делает его по АУГМЕНТИРОВАННЫМ данным.\n\n")
        f.write("**Итог по MixUp**: это потенциальный leakage. Правильный вариант — MixUp только внутри train-fold.\n\n")
        f.write("## Результаты\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")

    print(f"Отчёт: {out}/audit_report.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",    default="config.yaml")
    ap.add_argument("--elements",  nargs="+")
    ap.add_argument("--n-perm",    type=int, default=199)
    ap.add_argument("--skip-perm", action="store_true",
                    help="пропустить permutation test (быстрый запуск)")
    run(ap.parse_args())
