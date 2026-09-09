import os
import pandas as pd


# ============================================================
# CR1 reviewer recall summary
#
# Read existing Top-K analysis results
# Generate reviewer-required statistics
#
# Reviewer 2 comment:
# "What is the recall with respect to the full set of true
# interface residues? How many true interface residues were missed?"
# ============================================================


RESULT_DIR = (
    r"C:\Users\Administrator\Desktop\P-P"
    r"\cluster_split_v3_A2"
    r"\partner_only_DSSP"
    r"\11_CR1_case_analysis"
    r"\CR1_case_analysis_old_model_topK"
)


# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

TOPK_FILE = os.path.join(
    RESULT_DIR,
    "CR1_clean_chain_C_TopK_summary.csv"
)


OVERALL_FILE = os.path.join(
    RESULT_DIR,
    "CR1_clean_chain_C_overall_ranking_metrics.csv"
)


TOP8_FILE = os.path.join(
    RESULT_DIR,
    "CR1_clean_chain_C_Top8_Figure7.csv"
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_FILE = os.path.join(
    RESULT_DIR,
    "CR1_reviewer_response_summary.csv"
)


# ============================================================
# Check files
# ============================================================

for f in [
    TOPK_FILE,
    OVERALL_FILE,
    TOP8_FILE
]:
    if not os.path.exists(f):
        raise FileNotFoundError(
            f"\nCannot find:\n{f}"
        )


# ============================================================
# Load
# ============================================================

topk = pd.read_csv(TOPK_FILE)

overall = pd.read_csv(
    OVERALL_FILE
)

top8 = pd.read_csv(
    TOP8_FILE
)


print("=" * 80)
print("CR1 reviewer recall summary")
print("=" * 80)



# ============================================================
# Overall statistics
# ============================================================

total_residues = int(
    overall.loc[
        0,
        "Total_scanned_residues"
    ]
)


true_interface = int(
    overall.loc[
        0,
        "Total_true_interface_residues"
    ]
)


auroc = overall.loc[
    0,
    "AUROC"
]


auprc = overall.loc[
    0,
    "AUPRC"
]


print("\nOverall:")
print("-" * 80)

print(
    f"Total residues: {total_residues}"
)

print(
    f"True interface residues: {true_interface}"
)

print(
    f"AUROC: {auroc:.4f}"
)

print(
    f"AUPRC: {auprc:.4f}"
)



# ============================================================
# Top-K evaluation
# ============================================================

print("\nTop-K evaluation:")
print("-" * 80)


show_columns = [
    "K",
    "TP_at_K",
    "FP_at_K",
    "FN_at_K",
    "Precision_at_K",
    "Recall_at_K",
]


print(
    topk[show_columns]
    .to_string(index=False)
)



# ============================================================
# Reviewer key point
# Use Top-8 because Figure 7
# ============================================================

top8_row = topk[
    topk["K"] == 8
]


if len(top8_row) == 0:
    raise ValueError(
        "Cannot find K=8 result in TopK summary."
    )


top8_row = top8_row.iloc[0]


TP = int(
    top8_row["TP_at_K"]
)

FP = int(
    top8_row["FP_at_K"]
)

FN = int(
    top8_row["FN_at_K"]
)


precision = float(
    top8_row["Precision_at_K"]
)


recall = float(
    top8_row["Recall_at_K"]
)


f1 = (
    2 * precision * recall /
    (precision + recall)
    if precision + recall > 0
    else 0
)



# ============================================================
# Save reviewer summary
# ============================================================


summary = pd.DataFrame(
    [
        {
            "Evaluation": "CR1 Top-8 ranking",

            "Total_residues":
                total_residues,

            "Total_true_interface_residues":
                true_interface,

            "Predicted_topK_residues":
                8,

            "True_positive_TP":
                TP,

            "False_positive_FP":
                FP,

            "Missed_interface_residues_FN":
                FN,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1":
                f1,

            "AUROC":
                auroc,

            "AUPRC":
                auprc
        }
    ]
)


summary.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)



# ============================================================
# Print reviewer response sentence
# ============================================================

print("\nReviewer response key numbers:")
print("-" * 80)


print(
    f"""
For the CR1 case study, the model was evaluated against
{true_interface} experimentally defined interface residues.

Among the top-8 ranked residues,
{TP} residues were correctly identified as interface residues,
while {FP} were false positives.

The remaining {FN} interface residues were missed,
corresponding to a Recall@8 of {recall:.4f}.
"""
)


print("\nSaved:")
print(OUTPUT_FILE)