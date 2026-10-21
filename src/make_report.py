"""Generate the 2-page PDF write-up from results/*. Run AFTER train.py.

    python src/make_report.py --results results --out report/writeup.pdf --names "Name1 (SRN1), Name2 (SRN2)"
"""
import argparse
import json
import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def table(df, widths=None):
    data = [list(df.columns)] + df.values.tolist()
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.5), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6f5")),
                           ("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/writeup.pdf")
    ap.add_argument("--names", default="<Team member names and SRNs>")
    ap.add_argument("--team", default="<Team no.>")
    a = ap.parse_args()
    R = a.results
    ds = json.load(open(f"{R}/data_summary.json"))
    m = pd.read_csv(f"{R}/metrics.csv")
    sub = pd.read_csv(f"{R}/subgroups.csv")
    best = m.sort_values("test_auroc").iloc[-1]
    base = m[m.model.str.startswith("LogReg")].sort_values("test_auroc").iloc[-1]
    gbm = m[m.model.str.startswith("XGB")].iloc[0]
    gap = sub.auroc.max() - sub.auroc.min()
    gbm_name = "XGBoost" if ds["xgboost"] else "gradient boosting"

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.8, leading=11.2, spaceAfter=3)
    h = ParagraphStyle("h", parent=ss["Heading3"], fontSize=10.5, spaceBefore=5, spaceAfter=2)
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=15, spaceAfter=2)
    doc = SimpleDocTemplate(a.out, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.3 * cm, bottomMargin=1.2 * cm)
    P = lambda t, s=body: Paragraph(t, s)

    s = [P("Predicting Cardiovascular Risk using Electronic Health Records", title),
         P(f"UE24CS352A Machine Learning - Mini-Project | Team {a.team} | {a.names}", ss["Normal"]),
         P("1. Problem statement", h),
         P("Cardiovascular disease (CVD) is the leading cause of death worldwide, and identifying at-risk individuals early enables "
           "timely prevention. The task is to predict, from routinely recorded clinical data, whether a patient has cardiovascular "
           "disease (binary classification), and to compare interpretable regularised baselines against a gradient boosting model. "
           "The work follows the approach of Nguyen (CS229, Stanford) on EHR-based CVD risk prediction, adapted to a public dataset "
           "because the original STRIDE EHR data is private."),
         P("2. Dataset", h),
         P(f"We use the public Kaggle <i>Cardiovascular Disease</i> dataset ({ds['n_raw']:,} patient records). Features: age, gender, height, "
           "weight, systolic/diastolic blood pressure, cholesterol and glucose level (1-3), smoking, alcohol and physical activity; target: "
           f"presence of CVD. After removing implausible values (e.g. blood pressure or height/weight outliers, ap_hi &lt;= ap_lo) and "
           f"duplicates, <b>{ds['n_clean']:,}</b> records remain. We add BMI and pulse pressure as engineered features ({ds['n_features']} features in total). "
           f"The positive rate is {ds['positive_rate']*100:.1f}% (CVD incidence is {ds['male_rate']*100:.1f}% in men vs {ds['female_rate']*100:.1f}% in women). "
           "Data is split 80% train / 20% test (stratified); all tuning uses cross-validation on the training portion only."
           + (" <b>NOTE: synthetic data was used to generate this file.</b>" if ds["synthetic"] else "")),
         P("3. Approach", h),
         P(f"<b>Baselines:</b> logistic regression with Ridge (L2), Lasso (L1) and Elastic Net penalties on standardised features; the regularisation "
           f"strength C (and L1 ratio) are tuned by {ds['cv_folds']}-fold cross-validated grid search on AUROC. Regularisation reduces variance and Lasso "
           f"performs feature selection. <b>Main model:</b> {gbm_name} - an additive ensemble of sequentially built decision trees where each tree "
           "corrects the errors of the previous ones, with L1/L2 regularisation on leaf weights. Tuned over tree depth, learning rate, number of trees and "
           "regularisation strengths using the same cross-validation. <b>Metric:</b> AUROC (threshold independent), plus accuracy, sensitivity and specificity at "
           "threshold 0.5. <b>Fairness check:</b> AUROC of the best model within gender and age subgroups."),
         P("4. Implementation overview", h),
         P("Python 3 with pandas, scikit-learn and XGBoost. <font face='Courier'>src/data.py</font> loads/cleans data and builds features; "
           "<font face='Courier'>src/train.py</font> tunes and evaluates all models and writes metrics, ROC curve and subgroup results to "
           "<font face='Courier'>results/</font>; <font face='Courier'>src/make_report.py</font> and <font face='Courier'>src/make_slides.py</font> "
           "build this write-up and the presentation from those results. The full pipeline runs with one command (see README).")]

    mt = m.copy()
    for c in ["cv_auroc", "test_auroc", "accuracy", "sensitivity", "specificity"]:
        mt[c] = mt[c].map(lambda v: f"{v:.3f}")
    mt.columns = ["Model", "CV AUROC", "Test AUROC", "Accuracy", "Sensitivity", "Specificity"]
    st = sub.copy(); st["auroc"] = st.auroc.map(lambda v: f"{v:.3f}"); st = st[["subgroup", "n", "auroc"]]
    st.columns = ["Subgroup", "n (test)", "AUROC"]
    s += [P("5. Results", h), table(mt, [4.3 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm]), Spacer(1, 4)]
    imgs = []
    if os.path.exists(f"{R}/roc.png"):
        imgs.append(Image(f"{R}/roc.png", 6.4 * cm, 5.8 * cm))
    if os.path.exists(f"{R}/feature_importance.png"):
        imgs.append(Image(f"{R}/feature_importance.png", 7 * cm, 5.1 * cm))
    if imgs:
        s.append(Table([imgs], hAlign="LEFT"))
    s += [Spacer(1, 3), P(f"Subgroup AUROC for the best model ({best.model}):"), table(st, [3.5 * cm, 2.3 * cm, 2.3 * cm]),
          P("6. Conclusions", h),
          P(f"The best model was <b>{best.model}</b> with test AUROC {best.test_auroc:.3f} (best regularised logistic regression: {base.test_auroc:.3f}; "
            f"{gbm_name}: {gbm.test_auroc:.3f}). Train/CV and test AUROC are close, indicating little over-fitting. "
            f"Subgroup AUROC varies by {gap:.3f} across gender and age groups, so performance is not uniform and the model should be audited before "
            "clinical use. Limitations: the dataset is a single cross-sectional snapshot with self-reported lifestyle fields and no ethnicity "
            "information, so race-based fairness analysis from the original paper could not be repeated; the target is prevalent CVD rather than "
            "future incident events. Future work: probability calibration, longitudinal features, class-imbalance handling and fairness-aware training.")]
    doc.build(s)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
