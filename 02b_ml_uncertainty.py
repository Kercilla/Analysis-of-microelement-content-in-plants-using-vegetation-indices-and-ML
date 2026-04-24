#!/usr/bin/env python3
"""
02b_ml_uncertainty.py — детальный анализ неопределённости.

    python 02b_ml_uncertainty.py
    python 02b_ml_uncertainty.py --results-dir results/02_honest
"""
import argparse, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from analysis.cfg import load_config, short_name


def run(args):
    cfg     = load_config(args.config)
    # results-dir: CLI → config.paths.output.honest_ml → дефолт
    default_res = cfg.get("paths", {}).get("output", {}).get("honest_ml", "results/02_honest")
    res_dir = Path(args.results_dir or default_res)
    out     = Path(args.output) if args.output else res_dir / "uncertainty"
    out.mkdir(parents=True, exist_ok=True)
    dpi = cfg["plots"]["dpi"]

    # ── 1. Загрузка основных результатов ─────────────────────────────────
    metrics_path = res_dir / "honest_metrics.csv"
    if not metrics_path.exists():
        print(f"Не найдено: {metrics_path}"); return
    df = pd.read_csv(metrics_path)
    print(f"Загружено {len(df)} нутриентов из {metrics_path}")

    # ── 2. Таблица для диссертации ────────────────────────────────────────
    _make_dissertation_table(df, out, cfg)

    # ── 3. Heatmap: все модели × все нутриенты ───────────────────────────
    _plot_model_nutrient_heatmap(df, out, dpi)

    # ── 4. Buffer sweep графики ───────────────────────────────────────────
    sweep_files = list(res_dir.glob("sweep_*.csv"))
    if sweep_files:
        _plot_sweep_curves(sweep_files, out, dpi)

    # ── 5. Сводный scatter plot ───────────────────────────────────────────
    scatter_imgs = list(res_dir.glob("scatter_*.png"))
    if scatter_imgs:
        _make_scatter_grid(scatter_imgs, out, dpi)

    # ── 6. Сравнение random vs buffered ──────────────────────────────────
    cmp_path = res_dir / "comparison_random_vs_buffered.csv"
    if cmp_path.exists():
        _plot_inflation_analysis(pd.read_csv(cmp_path), out, dpi)

    print(f"\ndone → {out}/")


# ── Таблица для диссертации ───────────────────────────────────────────────

def _make_dissertation_table(df, out, cfg):
    """
    Главная таблица результатов в форматах CSV и LaTeX.
    Группирует нутриенты по интерпретируемым категориям.
    """
    df_sorted = df.sort_values("R2", ascending=False).copy()

    # Категории предсказуемости
    def category(r2, p):
        if pd.isna(r2):
            return "—"
        if r2 > 0.5 and p < 0.05:
            return "Хорошая"
        elif r2 > 0.2 and p < 0.05:
            return "Умеренная"
        elif r2 > 0 and p < 0.05:
            return "Слабая"
        elif r2 > 0:
            return "Незначимая (R²>0)"
        else:
            return "Нет сигнала"

    df_sorted["category"] = df_sorted.apply(
        lambda r: category(r["R2"], r.get("p_perm", 1.0)), axis=1)

    # Сохранение CSV
    df_sorted.to_csv(out/"dissertation_ranking.csv",
                     index=False, float_format="%.4f")

    # LaTeX таблица
    _latex_table(df_sorted, out/"dissertation_table.tex")

    # Печать итогов по категориям
    print("\n" + "="*60)
    print("РАНЖИРОВАНИЕ НУТРИЕНТОВ ПО ПРЕДСКАЗУЕМОСТИ")
    print("="*60)
    for cat in ["Хорошая","Умеренная","Слабая","Незначимая (R²>0)","Нет сигнала"]:
        sub = df_sorted[df_sorted["category"]==cat]
        if sub.empty:
            continue
        names = ", ".join(sub["short_name"].values)
        print(f"\n{cat} ({len(sub)}):")
        for _, r in sub.iterrows():
            ci_str = (f"[{r['R2_ci_lo']:.3f}, {r['R2_ci_hi']:.3f}]"
                      if pd.notna(r.get("R2_ci_lo")) else "")
            p_str  = f"p={r['p_perm']:.3f}" if pd.notna(r.get("p_perm")) else ""
            print(f"  {r['short_name']:<6}: R²={r['R2']:+.3f} {ci_str}  "
                  f"{p_str}  [{r['best_model']}]")


def _latex_table(df, path):
    """Генерирует LaTeX-таблицу результатов для диссертации."""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Результаты честной пространственной кросс-валидации "
        r"(buffered-LOO) для 12 нутриентов пшеницы}",
        r"\label{tab:honest_results}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Нутриент} & \textbf{Модель} & \textbf{$R^2$} & "
        r"\textbf{95\% CI} & \textbf{RMSE} & \textbf{RPD} & "
        r"\textbf{$p_\text{perm}$} \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        ci = (f"[{r['R2_ci_lo']:.3f}, {r['R2_ci_hi']:.3f}]"
              if pd.notna(r.get("R2_ci_lo")) else "---")
        p  = f"{r['p_perm']:.3f}" if pd.notna(r.get("p_perm")) else "---"
        sig = r"$^{**}$" if r.get("significant") else ""
        lines.append(
            f"  {r['short_name']} & {r['best_model']} & "
            f"{r['R2']:+.3f}{sig} & {ci} & "
            f"{r['RMSE']:.4f} & {r['RPD']:.2f} & {p} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\footnotesize",
        r"\raggedright",
        r"Примечание: $^{**}$ — значимо на уровне $p < 0.05$ "
        r"(permutation test). R² вычислен методом buffered leave-one-out "
        r"с индивидуальным радиусом буфера (из вариограмм). "
        r"BCa~bootstrap B=2000.",
        r"\end{table}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  saved: {path.name}")


# ── Heatmap модели × нутриенты ────────────────────────────────────────────

def _plot_model_nutrient_heatmap(df, out, dpi):
    """Показывает R² для каждой комбинации модель × нутриент."""
    if "best_model" not in df.columns:
        return
    # Если в df есть только лучшая модель — строим упрощённый bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    df_s = df.sort_values("R2", ascending=True)
    colors = ["#C00000" if r < 0 else
              "#F4A460" if r < 0.2 else
              "#2E75B6" if r < 0.5 else
              "#1A4D8C"
              for r in df_s["R2"]]
    bars = ax.barh(df_s["short_name"], df_s["R2"], color=colors, alpha=0.9)

    # CI
    for j, (_, row) in enumerate(df_s.iterrows()):
        if pd.notna(row.get("R2_ci_lo")) and pd.notna(row.get("R2_ci_hi")):
            ax.plot([row["R2_ci_lo"], row["R2_ci_hi"]], [j, j],
                    "k-", lw=2.5, alpha=0.6)

    # p-value аннотации
    for j, (_, row) in enumerate(df_s.iterrows()):
        p = row.get("p_perm", np.nan)
        if pd.notna(p):
            marker = "***" if p < 0.001 else "**" if p < 0.01 \
                     else "*" if p < 0.05 else "ns"
            x_pos  = max(row["R2"] + 0.02, row.get("R2_ci_hi", row["R2"]) + 0.01)
            ax.text(x_pos, j, marker, va="center", fontsize=8)

    ax.axvline(0, color="gray", lw=1, ls="--")
    ax.set_xlabel("R² (buffered-LOO + BCa 95% CI)", fontsize=10)
    ax.set_title("Предсказуемость нутриентов (цвет: тёмно-синий ≥ 0.5, "
                 "синий 0.2–0.5, оранжевый < 0.2, красный < 0)", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out/"nutrient_ranking_detailed.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: nutrient_ranking_detailed.png")


# ── Buffer sweep curves ───────────────────────────────────────────────────

def _plot_sweep_curves(sweep_files, out, dpi):
    """
    Кривые R²(buffer_radius) для каждого нутриента.
    Показывают как быстро падает R² при увеличении буфера —
    это и есть мера оптимистического смещения.
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    for k, fpath in enumerate(sorted(sweep_files)):
        if k >= len(axes):
            break
        sn  = Path(fpath).stem.replace("sweep_", "")
        df  = pd.read_csv(fpath)
        ax  = axes[k]
        ax.plot(df["buffer_m"], df["R2"], "o-", color="#2E75B6",
                lw=2, ms=5)
        ax.axhline(0, color="red", lw=0.8, ls="--", alpha=0.5)
        ax.fill_between(df["buffer_m"], df["R2"], 0,
                        where=df["R2"] > 0, alpha=0.15, color="#2E75B6")
        ax.set_title(sn, fontsize=9)
        ax.set_xlabel("буфер (м)", fontsize=8)
        ax.set_ylabel("R²", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
        # Аннотация при buffer=0 (leakage) и при честном буфере
        if len(df) >= 2:
            r2_0    = df["R2"].iloc[0]
            r2_last = df["R2"].iloc[-1]
            ax.annotate(f"buf=0: {r2_0:+.2f}", xy=(df["buffer_m"].iloc[0], r2_0),
                        fontsize=6, color="gray")

    for j in range(k+1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Зависимость R² от радиуса буфера\n"
                 "(быстрое падение = сильная пространственная автокорреляция)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out/"sweep_curves.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: sweep_curves.png")


# ── Scatter grid ──────────────────────────────────────────────────────────

def _make_scatter_grid(scatter_imgs, out, dpi):
    """Объединяет все scatter-plots в одну фигуру."""
    from PIL import Image
    n  = len(scatter_imgs)
    nc = 4
    nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4*nc, 4*nr))
    axes = axes.flatten()
    for k, img_path in enumerate(sorted(scatter_imgs)):
        try:
            img = Image.open(img_path)
            axes[k].imshow(np.array(img))
        except Exception:
            pass
        axes[k].axis("off")
    for j in range(k+1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Predicted vs Observed (buffered-LOO)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out/"scatter_all.png", dpi=dpi/2, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: scatter_all.png")


# ── Inflation analysis ────────────────────────────────────────────────────

def _plot_inflation_analysis(df, out, dpi):
    """
    Визуализация разницы random CV vs buffered-LOO.
    Главный методологический аргумент для диссертации.
    """
    df_s = df.sort_values("inflation", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: grouped bar
    x = np.arange(len(df_s))
    axes[0].bar(x-0.2, df_s["R2_random"],   0.35,
                label="Random 5-fold CV", color="#F4A460", alpha=0.85)
    axes[0].bar(x+0.2, df_s["R2_buffered"], 0.35,
                label="Buffered-LOO", color="#2E75B6", alpha=0.85)
    axes[0].axhline(0, color="gray", lw=1, ls="--")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_s["nutrient"], rotation=45,
                             ha="right", fontsize=9)
    axes[0].set_ylabel("R²", fontsize=10)
    axes[0].set_title("Random CV vs Buffered-LOO", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: inflation bars
    colors = ["#C00000" if v > 0.3 else "#F4A460" if v > 0.1 else "#90EE90"
              for v in df_s["inflation"]]
    axes[1].bar(x, df_s["inflation"], color=colors, alpha=0.85)
    axes[1].axhline(0, color="gray", lw=1, ls="--")
    axes[1].axhline(0.2, color="red", lw=1, ls=":", alpha=0.5,
                    label="Порог >0.2 (сильное смещение)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_s["nutrient"], rotation=45,
                             ha="right", fontsize=9)
    axes[1].set_ylabel("Завышение ΔR²", fontsize=10)
    axes[1].set_title("Оптимистическое смещение random CV\n"
                       "(красный = >0.3, оранжевый = 0.1–0.3, зелёный = <0.1)",
                       fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    med_infl = df_s["inflation"].median()
    axes[1].text(0.98, 0.95, f"Медиана Δ = {med_infl:+.3f}",
                 transform=axes[1].transAxes, ha="right", va="top",
                 fontsize=10, color="darkred",
                 bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    plt.tight_layout()
    fig.savefig(out/"inflation_analysis.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: inflation_analysis.png")

    # Текстовый вывод для диссертации
    print(f"\n  Медианное завышение random CV: {med_infl:+.3f}")
    print(f"  Максимальное завышение: "
          f"{df_s['inflation'].max():+.3f} "
          f"({df_s.loc[df_s['inflation'].idxmax(),'nutrient']})")
    n_high = (df_s["inflation"] > 0.2).sum()
    print(f"  Нутриентов с завышением >0.2: {n_high}/{len(df_s)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      default="config.yaml")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--output",      default=None)
    run(ap.parse_args())
