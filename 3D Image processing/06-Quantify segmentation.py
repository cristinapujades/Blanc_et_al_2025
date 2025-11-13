import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
import re
from scipy.stats import shapiro, levene

##############################################################################
base_dir = Path("F:/01-Analyzed/05-Cropped/03-Posterior r6-SC/Ngn1xGlut/")
conditions = ["Ascl1bKO", "Atoh1aKO", "Ptf1aKO", "Neurog1KO", "ScrambledKO"]
control_condition = "ScrambledKO"

# Minimum number of samples to keep per condition
min_samples_per_condition = 10

# We only detect outliers based on these raw metrics:
raw_outlier_metrics = ["C2", "C3", "TotalCount"]

# We'll still report "Ratio" in the final data, but NOT use it for outlier detection
all_metrics = ["C2", "C3", "TotalCount", "Ratio"]

# MAD-based threshold
mad_threshold = 3.5

# Output Excel name
output_excel_name = "analysis_MAD_maxZ_outliers.xlsx"

###############################################################################

def main():
    print("1) Reading TIF data and pivoting...")

    records = []
    conds_to_process = list(set(conditions + [control_condition]))
    
    for cond in conds_to_process:
        cond_path = base_dir / cond
        for tif_file in cond_path.glob("C[23]E*.tif"):
            mask = tifffile.imread(tif_file)
            cell_count = len(np.unique(mask[mask > 0]))

            match = re.search(r"E(\d+)", tif_file.name)
            if not match:
                continue
            exp_id = match.group(1)
            channel = tif_file.name[1]  # '2' or '3'

            records.append({
                "Condition": cond,
                "Experiment": f"E{exp_id}",
                "Channel": f"C{channel}",
                "Count": cell_count
            })

    if not records:
        print("No TIF masks found. Check your data and paths.")
        return

    df = pd.DataFrame(records)
    pivot_df = df.pivot_table(
        index=["Condition", "Experiment"],
        columns="Channel",
        values="Count",
        aggfunc="sum"
    ).fillna(0)

    # Ensure columns are named "C2","C3"
    pivot_df.rename(columns={"C2": "C2", "C3": "C3"}, inplace=True)

    # Compute ratio if both columns exist
    if {"C2","C3"}.issubset(pivot_df.columns):
        pivot_df["Ratio"] = pivot_df["C2"] / (pivot_df["C2"] + pivot_df["C3"])
    # Compute total
    pivot_df["TotalCount"] = pivot_df.get("C2", 0) + pivot_df.get("C3", 0)

    pivot_df = pivot_df.reset_index()

    # Sort by numeric experiment E1, E2, ...
    pivot_df["ExpNum"] = pivot_df["Experiment"].apply(
        lambda x: int(re.search(r"E(\d+)", x).group(1)) if re.search(r"E(\d+)", x) else 999999
    )
    pivot_df.sort_values(["Condition","ExpNum"], inplace=True)
    pivot_df.drop(columns=["ExpNum"], inplace=True)

    # Keep the full data in a separate DataFrame
    full_df = pivot_df.copy()

    ##########################################################################
    # 2) Normality checks (Shapiro) for raw_outlier_metrics, compute robust Z
    #    for each, then define outliers by max(|z_C2|, |z_C3|, |z_TotalCount|)
    ##########################################################################
    print("\n2) Checking normality (Shapiro) for raw metrics, computing robust Z-scores...")

    # We'll add columns for each metric's robust Z
    for metric in raw_outlier_metrics:
        full_df[f"{metric}_robustZ"] = np.nan

    # We'll store Shapiro results for reference
    normality_info = []

    # For each condition, compute median & MAD for each raw metric
    for cond, cond_df in full_df.groupby("Condition"):
        for metric in raw_outlier_metrics:
            if metric not in cond_df.columns:
                continue

            # Only real experiments E\d+
            metric_mask = cond_df["Experiment"].str.match(r"^E\d+$")
            metric_series = cond_df.loc[metric_mask, metric].dropna()
            indices = metric_series.index

            if len(metric_series) < 3:
                # Not enough for robust stats or Shapiro
                for idx in indices:
                    full_df.loc[idx, f"{metric}_robustZ"] = np.nan
                normality_info.append({
                    "Condition": cond,
                    "Metric": metric,
                    "N": len(metric_series),
                    "Shapiro_p": np.nan
                })
                continue

            # Shapiro–Wilk (for reference only)
            stat, pval = shapiro(metric_series)
            normality_info.append({
                "Condition": cond,
                "Metric": metric,
                "N": len(metric_series),
                "Shapiro_p": pval
            })

            # robust Z-scores
            median_val = metric_series.median()
            mad_val = (metric_series - median_val).abs().median()

            if mad_val == 0:
                robust_z = (metric_series - median_val) * 0.0
            else:
                factor = 1.4826
                robust_z = factor * (metric_series - median_val) / mad_val

            for idx in indices:
                full_df.loc[idx, f"{metric}_robustZ"] = robust_z.loc[idx]

    ##########################################################################
    # 3) Use the max(|z_C2|, |z_C3|, |z_TotalCount|) as CombinedZ, remove outliers
    ##########################################################################
    print("\n3) Determining outliers by max absolute robust Z, removing them in one pass...")

    full_df["CombinedZ"] = np.nan

    # Fill CombinedZ for each row
    for idx, row in full_df.iterrows():
        if not re.match(r"^E\d+$", str(row["Experiment"])):
            continue  # skip non-experiment rows

        # Extract robustZ for the raw metrics
        zvals = []
        for metric in raw_outlier_metrics:
            zcol = f"{metric}_robustZ"
            val = row.get(zcol, np.nan)
            if pd.isna(val):
                zvals = []
                break
            zvals.append(abs(val))

        if len(zvals) == 3:
            max_z = max(zvals)
            full_df.loc[idx, "CombinedZ"] = max_z

    # Collect potential outliers
    clean_df = full_df.copy()
    potential_outliers = []

    for cond, cond_df in full_df.groupby("Condition"):
        metric_mask = cond_df["Experiment"].str.match(r"^E\d+$")
        valid_idx = cond_df.loc[metric_mask].index

        for idx in valid_idx:
            cz = cond_df.loc[idx, "CombinedZ"]
            if pd.isna(cz):
                continue
            if cz > mad_threshold:
                potential_outliers.append({
                    "RowIndex": idx,
                    "Condition": cond,
                    "Experiment": cond_df.loc[idx, "Experiment"],
                    "MaxAbsZ": cz
                })

    potential_outliers_df = pd.DataFrame(potential_outliers)
    potential_outliers_df.sort_values(by="MaxAbsZ", ascending=False, inplace=True)

    outliers_list = []
    for _, row in potential_outliers_df.iterrows():
        idx = row["RowIndex"]
        cond = row["Condition"]

        if idx not in clean_df.index:
            continue

        cond_mask = (
            (clean_df["Condition"] == cond) &
            (clean_df["Experiment"].str.match(r"^E\d+$"))
        )
        if cond_mask.sum() > min_samples_per_condition:
            outliers_list.append({
                "Condition": cond,
                "Experiment": row["Experiment"],
                "MaxAbsZ": row["MaxAbsZ"]
            })
            clean_df.drop(index=idx, inplace=True)

    outliers_df = pd.DataFrame(outliers_list).reset_index(drop=True)

    ##########################################################################
    # 4) Levene’s test across conditions for C2, C3, TotalCount (cleaned data)
    ##########################################################################
    print("\n4) Checking homoscedasticity across conditions using Levene’s test...")

    homosc_list = []
    for metric in raw_outlier_metrics:  # or include "Ratio" if you wish
        groups_data = []
        valid_conds = []
        for cond in conditions:
            subdata = clean_df.loc[
                (clean_df["Condition"] == cond) &
                (clean_df["Experiment"].str.match(r"^E\d+$")),
                metric
            ].dropna()
            if len(subdata) >= 3:
                groups_data.append(subdata.values)
                valid_conds.append(cond)

        if len(groups_data) > 1:
            stat, pval = levene(*groups_data, center='median')
            homosc_list.append({
                "Metric": metric,
                "TestedConditions": ", ".join(valid_conds),
                "Levene_Statistic": stat,
                "Levene_p": pval
            })
        else:
            homosc_list.append({
                "Metric": metric,
                "TestedConditions": "N/A",
                "Levene_Statistic": np.nan,
                "Levene_p": np.nan
            })

    homosc_df = pd.DataFrame(homosc_list)

    ##########################################################################
    # 5) Compute summary stats + optional fold changes vs control
    ##########################################################################
    print("\n5) Computing summary stats...")

    summary_list = []
    for cond, grp in clean_df.groupby("Condition"):
        egrp = grp[grp["Experiment"].str.match(r"^E\d+$")]
        for metric in all_metrics:
            if metric in egrp.columns:
                arr = egrp[metric].dropna()
                if len(arr) > 0:
                    summary_list.append({
                        "Condition": cond,
                        "Metric": metric,
                        "N": len(arr),
                        "Mean": arr.mean(),
                        "Std": arr.std(ddof=1)
                    })
                else:
                    summary_list.append({
                        "Condition": cond,
                        "Metric": metric,
                        "N": 0,
                        "Mean": np.nan,
                        "Std": np.nan
                    })
    summary_df = pd.DataFrame(summary_list)

    # Fold changes vs. a "control_condition"
    ctrl_means = {}
    for metric in all_metrics:
        row = summary_df[
            (summary_df["Condition"] == control_condition) &
            (summary_df["Metric"] == metric)
        ]
        if not row.empty:
            ctrl_means[metric] = row["Mean"].values[0]
        else:
            ctrl_means[metric] = np.nan

    clean_df_fc = clean_df.copy()
    for metric in all_metrics:
        fc_col = f"{metric}_FoldChange"
        cm = ctrl_means[metric]
        if pd.isna(cm) or cm == 0:
            clean_df_fc[fc_col] = np.nan
        else:
            clean_df_fc[fc_col] = (clean_df_fc[metric] / cm) - 1.0

    ##########################################################################
    # 6) Output results
    ##########################################################################
    output_path = base_dir / output_excel_name
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Full data (with robustZ and CombinedZ)
        full_df.to_excel(writer, sheet_name="FullData", index=False)

        # Clean data
        clean_df_fc.to_excel(writer, sheet_name="CleanData", index=False)

        # Outliers
        outliers_df.to_excel(writer, sheet_name="Outliers", index=False)

        # Summary
        summary_df.to_excel(writer, sheet_name="SummaryStats", index=False)

        # Normality info
        normality_df = pd.DataFrame(normality_info)
        normality_df.to_excel(writer, sheet_name="Normality_Shapiro", index=False)

        # Homoscedasticity
        homosc_df.to_excel(writer, sheet_name="Homoscedasticity", index=False)

    print(f"\nAnalysis complete! Results saved to {output_path}")

if __name__ == "__main__":
    main()
