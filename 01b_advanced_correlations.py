#!/usr/bin/env python3
"""
01b_advanced_correlations.py — продвинутые меры зависимости.

Сравнивает для каждой пары (индекс, нутриент):
  Pearson r, Spearman ρ, Kendall τ, dCor, MI (KSG), partial correlation.

Дополнительно: CCA между блоком индексов и блоком нутриентов,
Graphical LASSO для структуры условных зависимостей.

    python 01b_advanced_correlations.py
    python 01b_advanced_correlations.py --date date2 --elements N_% Ca_%
"""
import argparse, os, sys, warnings
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from analysis.cfg import load_config, short_name
from analysis.loaders import load_multi, load_chemistry
from analysis.indices import calculate_indices
from analysis.dependence import (
    full_dependence_profile, graphical_lasso, canonical_correlation
)


def plot_measure_comparison(df_long, measures, out_dir, dpi=150):
    """
    Тепловые карты для каждой меры зависимости рядом.
    df_long: строки = (element, index), столбцы = меры.
    """
    fig, axes = plt.subplots(1, len(measures),
                              figsize=(5 * len(measures), 8))
    if len(measures) == 1:
        axes = [axes]

    for ax, meas in zip(axes, measures):
        # Агрегируем по датам перед pivot (берём макс |значение| по абсолютной величине)
        df_agg = (df_long.groupby(["index", "element"])[meas]
                  .apply(lambda s: s.loc[s.abs().idxmax()] if s.notna().any() else np.nan)
                  .reset_index())
        pivot = df_agg.pivot(index="index", columns="element", values=meas)
        pivot = pivot.reindex(
            pivot.abs().max(axis=1).sort_values(ascending=False).index)
        sns.heatmap(pivot, ax=ax, cmap="RdBu_r", center=0,
                    vmin=-0.6, vmax=0.6,
                    linewidths=0.3, linecolor="white",
                    cbar_kws={"shrink": 0.6})
        ax.set_title(meas, fontsize=11)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=7)

    plt.suptitle("Сравнение мер зависимости", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(Path(out_dir) / "dependence_comparison.png",
                dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: dependence_comparison.png")


def plot_pearson_vs_dcor(df_long, out_dir, dpi=150):
    """Scatter: Pearson r vs dCor — визуализация нелинейных эффектов."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    pairs = [("pearson_r", "dCor"),
             ("pearson_r", "MI"),
             ("spearman_rho", "dCor")]

    for ax, (mx, my) in zip(axes, pairs):
        sub = df_long[[mx, my, "element"]].dropna()
        if sub.empty:
            ax.set_visible(False); continue
        for el, grp in sub.groupby("element"):
            ax.scatter(grp[mx].abs(), grp[my],
                       s=20, alpha=0.5, label=el)
        ax.set_xlabel(f"|{mx}|", fontsize=10)
        ax.set_ylabel(my, fontsize=10)
        ax.set_title(f"|{mx}| vs {my}", fontsize=10)
        # диагональ: если всё линейно, точки лежат на y=x
        lim = max(sub[[mx, my]].abs().max().max(), 0.1)
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4)
        ax.grid(alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:12], labels[:12], loc="lower center",
               ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.05))
    plt.suptitle("Pearson vs нелинейные меры (точки выше диагонали = нелинейность)",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(Path(out_dir) / "pearson_vs_nonlinear.png",
                dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: pearson_vs_nonlinear.png")


def plot_glasso_graph(edges_df, out_dir, dpi=150):
    """Визуализация графа условных зависимостей из Graphical LASSO."""
    try:
        import networkx as nx
    except ImportError:
        print("  networkx не установлен, пропускаем граф")
        return

    G = nx.Graph()
    for _, row in edges_df.iterrows():
        if abs(row["partial_corr"]) > 0.05:
            G.add_edge(row["node_a"], row["node_b"],
                       weight=abs(row["partial_corr"]),
                       sign=np.sign(row["partial_corr"]))

    if G.number_of_edges() == 0:
        print("  нет значимых рёбер в графе")
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=2)
    weights = [G[u][v]["weight"] * 4 for u, v in G.edges()]
    colors  = ["#C00000" if G[u][v]["sign"] > 0 else "#2E75B6"
               for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=300,
                           node_color="lightgray", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    nx.draw_networkx_edges(G, pos, width=weights,
                           edge_color=colors, alpha=0.7, ax=ax)
    ax.set_title("Graphical LASSO: граф условных зависимостей\n"
                 "(красный = положит., синий = отрицат.)", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(Path(out_dir) / "glasso_graph.png",
                dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: glasso_graph.png")


def run(args):
    cfg  = load_config(args.config)
    out  = args.output or "results/01b_advanced"
    os.makedirs(out, exist_ok=True)

    bmap  = cfg["camera"]["band_map"]
    cols  = cfg["chemistry"]["columns"]
    elems = args.elements or cfg["chemistry"]["target_elements"]
    tiers = cfg["statistics"]["index_tiers"]
    dsmap = cfg["chemistry"]["date_sampling_map"]
    dpi   = cfg["plots"]["dpi"]
    dates = args.date or list(cfg["paths"]["multi"].keys())

    all_rows = []

    for dk in dates:
        pat = cfg["paths"]["multi"].get(dk)
        if not pat or not glob(pat):
            print(f"[{dk}] нет данных"); continue

        df_bands = load_multi(pat, bmap)
        df_idx   = calculate_indices(df_bands, tiers=tiers)

        sk  = dsmap.get(dk, "sampling1")
        cp  = cfg["paths"]["chemistry"].get(sk)
        off = cfg["chemistry"]["id_offsets"].get(sk, 0)
        chem = load_chemistry(cp, cols, off)

        ids   = df_idx.index.intersection(chem.index)
        avail = [e for e in elems if e in chem.columns]
        print(f"\n[{dk}] n={len(ids)}  индексов={df_idx.shape[1]}  нутриентов={len(avail)}")

        idx_sub  = df_idx.loc[ids]
        chem_sub = chem.loc[ids, avail]

        # ── Все пары (index, nutrient) ────────────────────────────────────────
        print(f"  Вычисляем меры зависимости ({df_idx.shape[1]} × {len(avail)} пар)...")
        for el in avail:
            y   = chem_sub[el].values.astype(float)
            sn  = short_name(cfg, el)

            # сформируем df для partial corr (контролируем остальные 4 канала)
            band_names = list(df_bands.columns)
            df_pc = pd.concat([df_bands.loc[ids], chem_sub[[el]]], axis=1).dropna()

            for idx_name in df_idx.columns:
                x = idx_sub[idx_name].values.astype(float)

                profile = full_dependence_profile(
                    x, y,
                    df=df_pc if idx_name in df_pc.columns else None,
                    covar_cols=band_names if idx_name in df_pc.columns else None,
                    x_name=idx_name, y_name=el,
                )
                profile["element"] = el
                profile["index"]   = idx_name
                profile["date"]    = dk
                all_rows.append(profile)

        # ── CCA ───────────────────────────────────────────────────────────────
        X_cca = idx_sub.values.astype(float)
        Y_cca = chem_sub.values.astype(float)
        cca_res = canonical_correlation(X_cca, Y_cca, n_components=5)
        if cca_res:
            print(f"  CCA канонические корреляции: {cca_res['canonical_corrs']}")
            pd.DataFrame([cca_res]).to_csv(
                f"{out}/cca_{dk}.csv", index=False)

        # ── Graphical LASSO ───────────────────────────────────────────────────
        # Объединяем нутриенты + топ-20 индексов по Pearson для наглядности
        top_idx = (df_idx.loc[ids].corrwith(chem_sub.mean(axis=1))
                   .abs().nlargest(20).index.tolist())
        glasso_names = avail + top_idx
        Z_gl = pd.concat([
            chem_sub[avail],
            df_idx.loc[ids, top_idx]
        ], axis=1).values.astype(float)

        print(f"  Graphical LASSO ({len(glasso_names)} узлов)...", end=" ", flush=True)
        gl_res = graphical_lasso(Z_gl, feature_names=glasso_names)
        if gl_res and not gl_res["edges"].empty:
            gl_res["edges"].to_csv(f"{out}/glasso_edges_{dk}.csv",
                                    index=False, float_format="%.4f")
            plot_glasso_graph(gl_res["edges"], out, dpi)
            print(f"{len(gl_res['edges'])} рёбер")
        else:
            print("нет результата")

    # ── Сохранение и визуализация ─────────────────────────────────────────────
    if not all_rows:
        print("Нет результатов"); return

    df = pd.DataFrame(all_rows)
    df.to_csv(f"{out}/dependence_matrix.csv", index=False, float_format="%.6f")
    print(f"\nСохранено: {out}/dependence_matrix.csv  ({len(df)} пар)")

    measures = ["pearson_r", "spearman_rho", "kendall_tau", "dCor", "MI"]
    avail_m  = [m for m in measures if m in df.columns
                and df[m].notna().sum() > 0]

    plot_measure_comparison(df, avail_m, out, dpi)
    plot_pearson_vs_dcor(df, out, dpi)

    # ── Сводная таблица: лучший индекс по каждому нутриенту × мере ──────────
    print(f"\n{'='*65}")
    print(f"ЛУЧШИЙ ИНДЕКС ПО КАЖДОМУ НУТРИЕНТУ (по |Pearson r|, дата 2)")
    print(f"{'Нутриент':<8} {'Индекс':<16} {'|r|':>5} {'ρ':>6} "
          f"{'dCor':>6} {'MI':>7} {'p_r':>7}")
    print("-"*60)

    for el in elems:
        sub = df[(df["element"] == el) & (df["date"] == dates[-1])].copy()
        if sub.empty: continue
        sub["abs_r"] = sub["pearson_r"].abs()
        b   = sub.nlargest(1, "abs_r").iloc[0]
        sn  = short_name(cfg, el)
        print(f"{sn:<8} {b['index']:<16} "
              f"{b['abs_r']:>5.3f} {b.get('spearman_rho', np.nan):>6.3f} "
              f"{b.get('dCor', np.nan):>6.3f} {b.get('MI', np.nan):>7.5f} "
              f"{b.get('pearson_p', np.nan):>7.4f}")

    print(f"\ndone → {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   default="config.yaml")
    ap.add_argument("--date",     nargs="+")
    ap.add_argument("--elements", nargs="+")
    ap.add_argument("--output",   default=None)
    run(ap.parse_args())
