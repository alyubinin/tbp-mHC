import pickle
import os 
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

log_dir = "log_out/"
analyze_dir = "analyze/"

plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 18,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.linewidth": 1.2,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.alpha": 0.4,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

BOX_COLOR = "#2E86AB"       
BOX_EDGE_COLOR = "#1A5276"  
MEDIAN_COLOR = "#1A3A4A"    
WHISKER_COLOR = "#333333"   
FLIER_COLOR = "#7B68EE"     



def plot_boxplot_with_sampled_fliers(data, labels, max_fliers=1000, keep_extreme=50):
    boxprops = dict(facecolor=BOX_COLOR, color=BOX_EDGE_COLOR, alpha=0.7, linewidth=1.5)
    medianprops = dict(color=MEDIAN_COLOR, linewidth=1)
    whiskerprops = dict(color=WHISKER_COLOR, linewidth=1.2)
    capprops = dict(color=WHISKER_COLOR, linewidth=1.2)
    
    bp = plt.boxplot(
        data, labels=labels, showfliers=False, patch_artist=True,
        boxprops=boxprops, medianprops=medianprops,
        whiskerprops=whiskerprops, capprops=capprops
    )
    
    bp["boxes"][0].set_label("mHC")
    plt.axhline(y=1, color="gray", linestyle="--", linewidth=1.2, label="mHC-lite")
    
    for i, d in enumerate(data):
        arr = np.array(d)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        fliers = arr[(arr < lower) | (arr > upper)]
        if len(fliers) > max_fliers:
            fliers_sorted = np.sort(fliers)
            low_extreme = fliers_sorted[:keep_extreme]
            high_extreme = fliers_sorted[-keep_extreme:]
            middle = fliers_sorted[keep_extreme:-keep_extreme]
            n_middle = max_fliers - 2 * keep_extreme
            if len(middle) > n_middle and n_middle > 0:
                middle = np.random.choice(middle, n_middle, replace=False)
            fliers = np.concatenate([low_extreme, middle, high_extreme])
        plt.scatter(
            [i + 1] * len(fliers), fliers, 
            marker="o", s=12, alpha=0.45, 
            color=FLIER_COLOR, edgecolors="none"
        )

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--suffix", type=str, default="")
    args = args.parse_args()

    filename = f"infos_{args.suffix}.pkl" if args.suffix else "infos.pkl"
    with open(os.path.join(log_dir, filename), "rb") as f:
        infos = pickle.load(f)
    suffix = "default" if not args.suffix else args.suffix

    all_ranges = []              
    col_sums_by_idx = {}         
    prod_col_sums_by_idx = {}    
    
    for t in infos.keys():
        H_res_list = []

        for name, info in infos[t].items():
            H_res = info["H_res"]         # (b, n, n)
            H_res_bef = info["H_res_bef"] # (b, n, n)
            H_res_list.append(H_res)

            for mat in H_res_bef:
                min_val = mat.min()
                max_val = mat.max()
                all_ranges.append(max_val - min_val)

            col_sums = H_res.sum(axis=1)  # (b, n)
            n = col_sums.shape[1]
            for j in range(n):
                col_sums_by_idx.setdefault(j, []).extend(col_sums[:, j].tolist())

        H_prod = H_res_list[0]  # (b, n, n)
        for H in H_res_list[1:]:
            H_prod = np.matmul(H_prod, H)  # (b, n, n)
        
        prod_col_sums = H_prod.sum(axis=1)  # (b, n)
        n = prod_col_sums.shape[1]
        for j in range(n):
            prod_col_sums_by_idx.setdefault(j, []).extend(prod_col_sums[:, j].tolist())

    fig, ax = plt.subplots(figsize=(4,6))
    n = len(all_ranges)
    n_gt_30 = len([x for x in all_ranges if x > 30])
    print(f"%>30: {n_gt_30}/{n}={n_gt_30 / n}")
    ax.hist(all_ranges, bins=50, edgecolor=BOX_EDGE_COLOR, color=BOX_COLOR, alpha=0.75)
    ax.set_xlabel("$\\log(1/\\nu)$")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x / 1000)}k" if x >= 1000 else f"{int(x)}"))
    fig.savefig(os.path.join(analyze_dir, f"nu_{suffix}.pdf"))
    plt.close(fig)

    n_idx = len(col_sums_by_idx)
    col_data = [col_sums_by_idx[j] for j in range(n_idx)]
    prod_col_data = [prod_col_sums_by_idx[j] for j in range(n_idx)]

    fig = plt.figure(figsize=(4, 4.5))
    plot_boxplot_with_sampled_fliers(col_data, [str(j) for j in range(n_idx)])
    plt.xlabel("Column Index")
    plt.ylabel("Column Sum")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(analyze_dir, f"hres_{suffix}.pdf"))
    plt.close()

    fig = plt.figure(figsize=(4, 4.5))
    plot_boxplot_with_sampled_fliers(prod_col_data, [str(j) for j in range(n_idx)])
    plt.xlabel("Column Index")
    plt.ylabel("Column Sum")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(analyze_dir, f"hres_prod_{suffix}.pdf"))
    plt.close()
