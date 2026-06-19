# -*- coding: utf-8 -*-
"""Frontiers Original-Research-Manuskript (TRIPOD+AI) aus der kanonischen Analyse."""
from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Pt, RGBColor, Cm

AN=Path(r"D:\Ausgangsdaten\KISIK Projekt\Eigene Auswertung"); CAN=AN/"canonical"
OUT=AN/"KISIK_Frontiers_Manuskript_v2.docx"
FONT="Times New Roman"; ACC=RGBColor(0x1F,0x4E,0x79); HFILL="1F4E79"; ZEB="EAF1F8"
doc=Document()
st=doc.styles["Normal"]; st.font.name=FONT; st.font.size=Pt(12)
st.paragraph_format.line_spacing_rule=WD_LINE_SPACING.ONE_POINT_FIVE; st.paragraph_format.space_after=Pt(6)
s=doc.sections[0]; s.page_width=Cm(21); s.page_height=Cm(29.7)
s.left_margin=s.right_margin=Cm(2.4); s.top_margin=s.bottom_margin=Cm(2.4)
CW=21-4.8

def run(p,t,bold=False,italic=False,size=None,color=None):
    r=p.add_run(t); r.font.name=FONT; r.font.bold=bold; r.font.italic=italic
    if size is not None: r.font.size=size
    if color is not None: r.font.color.rgb=color
    return r
def title(t):
    p=doc.add_paragraph(); p.paragraph_format.space_after=Pt(10); run(p,t,bold=True,size=Pt(15),color=ACC)
def h1(t):
    p=doc.add_paragraph(); p.paragraph_format.space_before=Pt(12); p.paragraph_format.space_after=Pt(3); run(p,t,bold=True,size=Pt(13),color=ACC)
def h2(t):
    p=doc.add_paragraph(); p.paragraph_format.space_before=Pt(8); p.paragraph_format.space_after=Pt(2); run(p,t,bold=True,italic=True,size=Pt(12))
def body(t):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY; run(p,t,size=Pt(12)); return p
def labeled(lab,t):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY; run(p,lab+" ",bold=True); run(p,t); return p
def small(t):
    p=doc.add_paragraph(); run(p,t,size=Pt(9.5),italic=True,color=RGBColor(0x55,0x55,0x55)); return p
def cap(n,t):
    p=doc.add_paragraph(); p.paragraph_format.space_before=Pt(10); p.paragraph_format.space_after=Pt(3)
    run(p,f"Table {n}. ",bold=True,size=Pt(10.5)); run(p,t,size=Pt(10.5))
def figcap(n,t):
    p=doc.add_paragraph(); p.paragraph_format.space_after=Pt(8)
    run(p,f"Figure {n}. ",bold=True,size=Pt(9.5)); run(p,t,size=Pt(9.5))
def bg(cell,hexc):
    tcPr=cell._tc.get_or_add_tcPr(); sh=OxmlElement("w:shd")
    sh.set(qn("w:val"),"clear"); sh.set(qn("w:color"),"auto"); sh.set(qn("w:fill"),hexc); tcPr.append(sh)
def settext(cell,t,bold=False,color=None,size=9.5,align="left"):
    cell.text=""; p=cell.paragraphs[0]
    p.alignment={"left":WD_ALIGN_PARAGRAPH.LEFT,"center":WD_ALIGN_PARAGRAPH.CENTER}[align]
    p.paragraph_format.space_after=Pt(1); p.paragraph_format.space_before=Pt(1)
    r=p.add_run(t); r.font.name=FONT; r.font.size=Pt(size); r.font.bold=bold
    if color: r.font.color.rgb=color
    cell.vertical_alignment=WD_ALIGN_VERTICAL.CENTER
def table(rows,widths,numcols_from=1):
    t=doc.add_table(rows=len(rows),cols=len(rows[0])); t.style="Table Grid"; t.alignment=WD_TABLE_ALIGNMENT.CENTER
    t.autofit=False
    for ri,row in enumerate(rows):
        for ci,val in enumerate(row):
            c=t.cell(ri,ci); c.width=Cm(widths[ci])
            if ri==0: settext(c,val,bold=True,color=RGBColor(0xFF,0xFF,0xFF),align="center")
            else:
                settext(c,val,align="center" if ci>=numcols_from else "left")
        if ri==0:
            for c in t.rows[0].cells: bg(c,HFILL)
        elif ri%2==0:
            for c in t.rows[ri].cells: bg(c,ZEB)
    return t
def figure(path,width_cm=15.5):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(8); p.paragraph_format.space_after=Pt(2)
    p.add_run().add_picture(str(path),width=Cm(width_cm))

# ============================ TITLE / ABSTRACT ============================
title("Early prediction of intensive-care length of stay from the first 24 hours: "
      "a leakage-controlled model-development study with prospective benchmarking against senior-physician judgement")
small("Original Research prepared for the Frontiers Research Topic “MedicineAI: Advancing the Synergy of "
      "Medicine and AI — From Data to Clinical Impact.” Reported in accordance with the TRIPOD+AI statement "
      "(Collins et al., 2024). Author names, affiliations, ORCID and corresponding author to be inserted.")
p=doc.add_paragraph(); run(p,"Author One¹, Author Two¹, Author Three², Senior Author¹*",size=Pt(11))
p=doc.add_paragraph(); run(p,"¹ Department of Anaesthesiology and Intensive Care Medicine, [Institution], [City], [Country]\n"
      "² [Second affiliation]\n* Correspondence: [name, e-mail]",size=Pt(10.5))

h1("Abstract")
labeled("Background:","Accurate early prediction of intensive care unit (ICU) length of stay (LOS) could support "
    "capacity planning and patient flow. Reported machine-learning (ML) performance is often optimistic because of "
    "information leakage, and models are rarely benchmarked prospectively against the clinicians they are meant to support.")
labeled("Methods:","Using routine data from a single tertiary ICU service, we developed models to predict ICU LOS "
    "(in days) from information available within the first 24 hours after admission only. The retrospective development "
    "cohort comprised 17,032 ICU stays (12,414 patients); the split into training and test data and all cross-validation "
    "folds were grouped by patient. Four candidate models — ridge regression, random forest, extremely randomised trees "
    "(extra trees) and gradient boosting (XGBoost) — were trained on a log1p-transformed target with identical "
    "preprocessing; hyperparameters were tuned by 4-fold patient-grouped cross-validation. The single model with the "
    "lowest cross-validated mean absolute error (MAE) was pre-specified as the final model. The final model was then "
    "evaluated, unchanged, against prospectively documented senior-physician LOS estimates in an independent cohort "
    "(360 matched stays).")
labeled("Results:","A feature set that aggregated measurements over the whole stay produced optimistic apparent "
    "performance (R² ≈ 0.61); restricting predictors to a strict 24-hour window removed this leakage and revealed a "
    "substantially harder task. On the patient-grouped hold-out test set the four models performed near-identically "
    "(R² 0.23–0.32; MAE 2.75–2.94 days); extra trees was selected (test MAE 2.76 days, R² 0.31). The strongest "
    "predictors were early intensive-care complex-treatment and monitoring procedure codes. In the prospective "
    "comparison the senior physician outperformed the model (MAE 2.60 vs 3.63 days; R² 0.25 vs −0.02), with the gap "
    "concentrated in long stays.")
labeled("Conclusion:","Under strict leakage control and prospective, clinician-benchmarked validation, 24-hour ML "
    "models did not match experienced senior-physician judgement for ICU LOS prediction. Transparent reporting of the "
    "prediction time point and clinician benchmarking are essential before such models are considered for deployment.")
p=doc.add_paragraph(); run(p,"Keywords: ",bold=True); run(p,"intensive care unit; length of stay; clinical prediction model; "
    "machine learning; data leakage; external validation; TRIPOD+AI; clinical decision support")

# ============================ 1 INTRODUCTION ============================
h1("1  Introduction")
body("Intensive care is among the most resource-intensive parts of hospital care, and ICU length of stay (LOS) is a "
    "principal driver of bed occupancy, staffing and cost. A reliable early estimate of LOS could support bed "
    "management, step-down planning and communication with patients and relatives. Established severity scores such as "
    "APACHE II, SAPS II and SOFA were designed mainly to predict mortality and explain only a limited share of LOS "
    "variation [1–3].")
body("Machine learning (ML) has been applied widely to ICU LOS prediction [4–6], frequently with encouraging reported "
    "performance. Two recurring problems undermine clinical credibility. First, information leakage — the inadvertent "
    "use of information unavailable at the intended prediction time — inflates apparent performance and is a leading "
    "cause of over-optimism and irreproducibility in ML-for-health research [7]. For a temporal outcome such as LOS, "
    "features aggregated over the whole admission are especially hazardous, because their values are mechanically "
    "related to how long the patient stays. Second, models are almost always evaluated against historical labels and "
    "rarely against the clinicians whose judgement they are meant to augment. Reporting and validation standards, "
    "including the AI-specific TRIPOD+AI statement, therefore emphasise transparent handling of the prediction time "
    "point and rigorous, ideally prospective, evaluation [8,9].")
body("We developed a clinically realistic ML model for ICU LOS that uses only information available within the first "
    "24 hours after admission, explicitly quantified the effect of leakage from whole-stay feature aggregation, "
    "pre-specified a single final model through patient-grouped cross-validation, and benchmarked that model "
    "prospectively against documented senior-physician estimates. The guiding question was the one that matters for "
    "clinical impact: at the point of decision, does the model beat the doctor?")

# ============================ 2 METHODS ============================
h1("2  Materials and Methods")
h2("2.1  Study design, setting and data source")
body("Single-centre study using de-identified routine data from the clinical data repository (KISIK) of a tertiary "
    "care centre. ICU stay records, ICD-10 diagnoses, laboratory results, vital signs, OPS procedure codes and "
    "vascular-access devices were linked by case identifier. The study was conducted per the Declaration of Helsinki; "
    "ethics approval and the waiver of informed consent for analysis of de-identified routine data are to be inserted "
    "(see Ethics statement). Reporting follows TRIPOD+AI [9].")
h2("2.2  Cohorts (retrospective development and prospective evaluation)")
body("Two clearly separated cohorts were used (Table 1). The retrospective development cohort comprised 17,032 ICU "
    "stays from 12,414 patients (13,275 hospital encounters) on 17 ward/care-unit combinations of the participating "
    "ICU service, restricted to stays longer than 24 hours. The independent prospective cohort was drawn from a later "
    "time period (daily live-system snapshots); of these, 360 stays had a senior-physician LOS estimate documented "
    "during routine care and formed the matched evaluation set. The prospective cohort was used only for evaluation, "
    "never for model fitting or selection.")
h2("2.3  Outcome and units")
body("The outcome was ICU LOS in days. In the source data the stay duration is recorded in hours (icu_duration_h); the "
    "modelling target is therefore icu_duration_h ÷ 24 (days). The senior-physician estimates were recorded directly in "
    "days. All reported metrics are on the day scale. Because LOS is strictly positive and strongly right-skewed "
    "(median 2.86 days, 90th percentile 12.1 days, maximum 76.9 days), models were fitted on a log1p-transformed "
    "target. log1p denotes log(1 + LOS) — the “1p” means “plus one”. This transform was chosen over a plain log "
    "because it is numerically well-behaved near the lower bound and avoids the singularity of log at zero, while its "
    "inverse, expm1 (exp(x) − 1), guarantees non-negative day-scale predictions; predictions were back-transformed "
    "with expm1 before any metric was computed. The log1p/expm1 transform was applied identically to all four models "
    "via a single target-transformation wrapper.")
h2("2.4  Predictors, the 24-hour window and leakage control")
body("All predictors were derived strictly from the first 24 hours after ICU admission (admission timestamp to "
    "admission + 24 h). This is the central leakage safeguard: no measurement, procedure or device recorded after the "
    "prediction window can enter the model. From 104 candidate features we used the 84 that were available as genuine "
    "first-24-hour variables in the dataset; 20 candidates were excluded because no leakage-free 24-hour version was "
    "available under the exact feature name. Importantly, an earlier version of the pipeline had silently substituted "
    "whole-stay aggregates for 15 of these (e.g. a whole-stay laboratory summary in place of the 24-hour value); this "
    "constitutes leakage and was removed. The 84 retained predictors span six domains (Table 2): early laboratory "
    "values (30), admission diagnoses (24), procedures (13), vascular access (8), vital signs (6) and "
    "demographic/admission context (3).")
h2("2.5  Handling of repeated stays and multiple procedure records")
body("The analysis unit was the ICU stay. 3,156 patients contributed more than one ICU stay. To prevent optimistic "
    "bias, the train/test split and every cross-validation fold were grouped by patient identifier, so that no patient "
    "appeared simultaneously in training and evaluation data. Where a stay had several procedure or operation records "
    "within the window, these were de-duplicated and aggregated to presence indicators and counts (and, for "
    "perioperative timing analyses, summed durations and the maximum ASA class). Senior-physician estimates were "
    "provided once per stay, so no within-stay aggregation was required for the benchmark.")
h2("2.6  Missing data and preprocessing")
body("A single preprocessing pipeline was applied inside cross-validation (fitted on training folds only, to avoid "
    "leakage): median imputation of numeric predictors, most-frequent imputation and one-hot encoding of the single "
    "categorical predictor (care-unit type), and standardisation of numeric predictors for ridge regression only "
    "(tree ensembles are scale-invariant). In the prospective cohort, early features not materialised in the "
    "live-snapshot data were imputed with the training medians — a deliberate reflection of real-world deployment.")
h2("2.7  Model development, hyperparameter tuning and model selection")
body("Four candidate models were compared consistently throughout: ridge regression (a regularised linear baseline), "
    "random forest, extra trees and XGBoost (gradient boosting). Hyperparameters were tuned on the training set only "
    "by 4-fold patient-grouped cross-validation (GroupKFold on patient identifier), optimising the negative mean "
    "absolute error on the day scale; tree ensembles used randomised search (12 candidates each) and ridge a grid "
    "search. Search spaces: ridge alpha ∈ {0.1, 0.3, 1, 3, 10, 30, 100}; for random forest and extra trees "
    "n_estimators ∈ {300, 500}, max_depth ∈ {None, 12, 20}, min_samples_leaf ∈ {2, 5, 10}, max_features ∈ {sqrt, 0.5}; "
    "for XGBoost n_estimators ∈ {300, 500, 800}, max_depth ∈ {4, 6, 8}, learning_rate ∈ {0.03, 0.05, 0.1}, subsample ∈ "
    "{0.7, 0.9}, colsample_bytree ∈ {0.7, 0.9}, min_child_weight ∈ {1, 3, 5}, reg_lambda ∈ {1, 2, 5}. The selected "
    "hyperparameters are reported in Table 3. The single model with the lowest cross-validated MAE was pre-specified "
    "as the final model; it was refitted on the full training set and evaluated once on the hold-out test set and once "
    "on the prospective cohort. The complete final pipeline is: ColumnTransformer (imputation, one-hot encoding, "
    "optional scaling) → estimator, all wrapped in a target-transformation step applying log1p on fitting and expm1 on "
    "prediction. Every number reported in this paper was produced by this one pipeline.")
h2("2.8  Performance metrics")
body("All metrics were computed on the day scale after back-transformation, with each stay contributing exactly once. "
    "For N stays with observed LOS yᵢ and predicted LOS ŷᵢ (both in days): MAE = (1/N) Σ|yᵢ − ŷᵢ|; median absolute "
    "error = median(|yᵢ − ŷᵢ|); RMSE = √[(1/N) Σ(yᵢ − ŷᵢ)²]; mean bias = (1/N) Σ(ŷᵢ − yᵢ); and R² = 1 − Σ(yᵢ − ŷᵢ)² / "
    "Σ(yᵢ − ȳ)². The MAE is thus the average absolute deviation, in days, between observed and predicted ICU LOS over "
    "the evaluated stays.")
h2("2.9  Interpretability")
body("Predictor importance for the final model was assessed by permutation importance on the hold-out test set "
    "(10 repeats), defined as the increase in MAE (days) when a predictor’s values are randomly permuted; larger "
    "values indicate greater reliance on that predictor.")
h2("2.10  Software and reproducibility")
body("Analyses used Python 3.12 (scikit-learn, XGBoost, DuckDB, SciPy). A fixed random seed (42) was used for the "
    "split, cross-validation and model fitting. The full analysis and figure code is openly available (see Data "
    "availability statement).")

# ============================ 3 RESULTS ============================
h1("3  Results")
h2("3.1  Cohorts")
body("Cohort characteristics are summarised in Table 1. The retrospective cohort contained 17,032 stays (12,414 "
    "patients), split into 13,603 training and 3,429 hold-out test stays with no patient shared between them. The "
    "prospective matched cohort contained 360 stays with a documented senior-physician estimate.")
h2("3.2  Effect of leakage control")
body("Replacing the 15 inadvertently substituted whole-stay features with their genuine 24-hour counterparts (or "
    "removing them where unavailable) reduced apparent retrospective performance markedly: a feature set containing "
    "whole-stay aggregates reached R² ≈ 0.61, whereas the strictly leakage-free 24-hour feature set yielded R² ≈ 0.31 "
    "for the same model family. All results below use the leakage-free feature set.")
h2("3.3  Model development and selection")
body("In patient-grouped cross-validation the three tree ensembles were near-identical and clearly better than the "
    "linear baseline (Table 3). On the hold-out test set the models again performed within a narrow band (MAE "
    "2.75–2.94 days; R² 0.23–0.32). Extra trees had the lowest cross-validated MAE and was pre-specified as the final "
    "model (test MAE 2.76 days, median absolute error 1.22 days, RMSE 5.34 days, R² 0.31, mean bias −1.08 days); random "
    "forest and XGBoost were statistically indistinguishable from it, and ridge regression was consistently worst. All "
    "models showed a modest negative bias, i.e. a tendency to under-predict.")
h2("3.4  Most important predictors")
body("Permutation importance for the final model (Table 5, Figure 2) was dominated by early procedure codes: the "
    "intensive-care complex-treatment code in the first 24 hours was by far the strongest predictor (permuting it "
    "increased MAE by ≈ 0.96 days), followed by its extended/prolonged tiers, the care-unit type, early monitoring "
    "codes and the total number of procedures. Ventilator-dependence and a small number of diagnoses (viral pneumonia, "
    "hydrocephalus) and patient age contributed modestly.")
h2("3.5  Prospective comparison with the senior physician")
body("In the matched prospective cohort the senior physician outperformed the final model on every overall metric "
    "(Table 4): MAE 2.60 vs 3.63 days, median absolute error 0.93 vs 2.23 days, R² 0.25 vs −0.02. All four ML models "
    "behaved similarly (MAE 3.63–3.90 days, R² ≈ 0), i.e. no better than predicting the cohort mean. The marked drop "
    "from retrospective to prospective performance is consistent with distribution shift between the development and "
    "prospective periods and with the reduced availability of early features in the live data.")
h2("3.6  Calibration across the LOS range")
body("Observed-versus-predicted density plots (Figures 3–5, shown to 20 days for legibility) make the behaviour "
    "explicit. Both the model and the physician are well aligned with the identity line for short stays, but "
    "predictions flatten for longer stays: the model in particular regresses long-staying patients towards the "
    "population mean and systematically under-predicts beyond roughly one week, whereas the physician tracks longer "
    "stays more closely. The clinically important long-stay range is therefore where the model is weakest.")

# ---- Figures ----
figure(CAN/"fig_model_comparison.png",15.0)
figcap(1,"Mean absolute error (MAE, days) of the four candidate models in the retrospective hold-out test set and in "
    "the prospective cohort; the dashed line marks the senior-physician MAE. All four models are reported consistently.")
figure(CAN/"fig_importance.png",15.0)
figcap(2,"Permutation feature importance of the final model (extra trees) on the hold-out test set: increase in MAE "
    "(days) when each predictor is permuted (mean ± SD over 10 repeats).")
figure(CAN/"fig_hexbin_retro_ExtraTrees.png",11.5)
figcap(3,"Retrospective hold-out: observed versus predicted ICU LOS (final model, extra trees). Axes truncated at 20 "
    "days for legibility; metrics computed on all test stays.")
figure(CAN/"fig_hexbin_pros_ExtraTrees.png",11.5)
figcap(4,"Prospective cohort: observed versus predicted ICU LOS (final model). Axes truncated at 20 days.")
figure(CAN/"fig_hexbin_pros_oberarzt.png",11.5)
figcap(5,"Prospective cohort: observed ICU LOS versus the senior-physician estimate. Axes truncated at 20 days.")

# ============================ 4 DISCUSSION ============================
h1("4  Discussion")
body("In a large single-centre cohort we developed an ICU LOS model from the first 24 hours after admission, "
    "controlled explicitly for leakage, pre-specified a single final model through patient-grouped cross-validation, "
    "and benchmarked it prospectively against documented senior-physician estimates. Three findings stand out. First, "
    "leakage from whole-stay feature aggregation materially inflated apparent performance (R² ≈ 0.61 → 0.31 once "
    "removed). Second, under leakage control the four model families performed near-identically; the choice between "
    "them is therefore immaterial, and the gradient-boosting-versus-linear distinction emphasised in some reports is "
    "not supported here. Third, and most importantly, the experienced senior physician outperformed the model "
    "prospectively, with the gap concentrated among long-staying patients.")
body("The leakage result generalises beyond this dataset: because LOS is a temporal outcome, any whole-stay summary "
    "encodes part of the answer, and such features are easy to introduce inadvertently from wide, pre-aggregated "
    "tables. This supports recent warnings on leakage and reproducibility [7] and the TRIPOD+AI emphasis on a clearly "
    "defined prediction time point [9]. The central message concerns the clinician comparison: judged against "
    "historical labels the model looks reasonable, but judged against the people who make the decision it falls short, "
    "precisely in the long-stay cases that matter most for capacity planning. Experienced physicians integrate "
    "contextual knowledge — surgical trajectory, anticipated complications, organisational factors — not captured in "
    "structured 24-hour data.")
h2("4.1  Limitations")
body("This is a single-centre study; absolute performance may not transfer. The prospective comparison rested on 360 "
    "matched stays, and the physicians’ estimates may have drawn on information accruing beyond 24 hours. The "
    "retrospective-to-prospective performance drop indicates distribution shift whose drivers (case mix, coding, "
    "feature availability) were not fully characterised. We restricted predictors to the first 24 hours; richer "
    "longitudinal inputs, dynamic re-prediction, or tail-oriented approaches (quantile, Tweedie/Gamma, discrete-time "
    "hazard models) may narrow the long-stay gap and are a priority for future work.")
h1("5  Conclusion")
body("Under strict leakage control and prospective, clinician-benchmarked validation, a machine-learning model built "
    "from the first 24 hours of ICU data did not match experienced senior-physician judgement for length-of-stay "
    "prediction, with the clinician’s advantage concentrated in long stays. Realising clinical impact will depend less "
    "on incremental gains against historical labels and more on rigorous, transparent validation and honest comparison "
    "with the clinicians the technology is meant to serve.")

# ============================ STATEMENTS / REFS ============================
h1("Statements")
labeled("Data availability statement.","De-identified routine clinical data are not publicly available owing to "
    "data-protection regulations; requests may be directed to the corresponding author subject to institutional "
    "approval. The complete analysis and figure code is openly available at "
    "https://github.com/Scarlet1991/KISIK-Dashboard.")
labeled("Ethics statement.","The study analysed de-identified routine clinical data; the protocol and waiver of "
    "informed consent were reviewed by [ethics committee], approval number [to be inserted].")
labeled("Author contributions.","[To be completed.] All authors approved the final manuscript.")
labeled("Funding.","[To be completed; if none, state so.]")
labeled("Conflict of interest.","The authors declare no competing interests. [Amend as appropriate.]")

h1("References")
refs=[
 "Knaus WA, Draper EA, Wagner DP, Zimmerman JE. APACHE II: a severity of disease classification system. Crit Care Med. 1985;13(10):818–829.",
 "Le Gall JR, Lemeshow S, Saulnier F. A new Simplified Acute Physiology Score (SAPS II). JAMA. 1993;270(24):2957–2963.",
 "Vincent JL, Moreno R, Takala J, et al. The SOFA score to describe organ dysfunction/failure. Intensive Care Med. 1996;22(7):707–710.",
 "Verburg IWM, Atashi A, Eslami S, et al. Which models can I use to predict adult ICU length of stay? A systematic review. Crit Care Med. 2017;45(2):e222–e231.",
 "Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible critical care database. Sci Data. 2016;3:160035.",
 "Pollard TJ, Johnson AEW, Raffa JD, et al. The eICU Collaborative Research Database. Sci Data. 2018;5:180178.",
 "Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. Patterns. 2023;4(9):100804.",
 "Collins GS, Reitsma JB, Altman DG, Moons KGM. Transparent reporting of a multivariable prediction model (TRIPOD). Ann Intern Med. 2015;162(1):55–63.",
 "Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine-learning methods. BMJ. 2024;385:e078378.",
 "Breiman L. Random forests. Mach Learn. 2001;45(1):5–32.",
 "Geurts P, Ernst D, Wehenkel L. Extremely randomized trees. Mach Learn. 2006;63(1):3–42.",
 "Chen T, Guestrin C. XGBoost: a scalable tree boosting system. In: Proc. 22nd ACM SIGKDD; 2016. p. 785–794.",
]
for i,r in enumerate(refs,1):
    p=doc.add_paragraph(); p.paragraph_format.left_indent=Cm(0.8); p.paragraph_format.first_line_indent=Cm(-0.8); p.paragraph_format.space_after=Pt(3)
    run(p,f"[{i}]  {r}",size=Pt(10.5))
small("Reference details should be verified and topic-specific references added before submission. In-text markers "
    "[1]–[9] correspond to this list.")

# ============================ TABLES (appended after text) ============================
doc.add_paragraph().add_run("").add_break()
h1("Tables")

cap(1,"Cohort definition and characteristics (retrospective development vs prospective evaluation).")
table([
 ["Characteristic","Retrospective (development)","Prospective (evaluation)"],
 ["ICU stays (n)","17,032","360 (matched to a senior estimate)"],
 ["Patients (n)","12,414","—"],
 ["Hospital encounters (n)","13,275","—"],
 ["Patients with >1 ICU stay","3,156","—"],
 ["Median ICU LOS (days)","2.86","—"],
 ["90th percentile LOS (days)","12.1","—"],
 ["Maximum LOS (days)","76.9","—"],
 ["Train / test split","13,603 / 3,429 (patient-grouped)","evaluation only"],
 ["Role","model development & selection","external prospective test"],
],[5.2,5.5,5.5],numcols_from=1)
small("LOS, length of stay. The prospective cohort was never used for fitting or model selection.")

cap(2,"Predictor domains (84 leakage-free features, all from the first 24 hours).")
table([
 ["Domain","Features (n)","Examples"],
 ["Laboratory (first 24 h)","30","potassium, sodium, glucose, lactate, bilirubin (first/mean/min/max/last/count)"],
 ["Admission diagnoses (ICD-10)","24","sepsis, intracranial haemorrhage, coronary disease, ventilator dependence"],
 ["Procedures (OPS, first 24 h)","13","intensive-care complex treatment, monitoring, ventilation, transfusion"],
 ["Vascular access (first 24 h)","8","arterial line, central venous catheter, urinary catheter"],
 ["Vital signs (first 24 h)","6","peripheral oxygen saturation (SpO₂)"],
 ["Demographics / admission","3","age, ICU stay number, care-unit type"],
 ["Total","84",""],
],[5.0,2.2,8.6],numcols_from=1)
small("20 of 104 candidate features were excluded (no leakage-free 24-hour version under the exact name), including 15 "
    "that an earlier pipeline had mapped to whole-stay aggregates (the removed leakage).")

cap(3,"Model development: tuned hyperparameters, cross-validated and hold-out performance (all four candidate models).")
table([
 ["Model","Selected hyperparameters","CV MAE (d)","Test MAE (d)","Median AE (d)","RMSE (d)","R²","Bias (d)"],
 ["Ridge","alpha = 0.1 (standardised inputs)","2.855","2.936","1.402","5.621","0.234","−1.208"],
 ["Random forest","500 trees, max_depth 20, leaf 2, max_feat 0.5","2.659","2.752","1.239","5.324","0.313","−1.109"],
 ["Extra trees (final)","500 trees, max_depth 20, leaf 2, max_feat 0.5","2.656","2.758","1.218","5.343","0.308","−1.078"],
 ["XGBoost","500 trees, depth 8, lr 0.05, subsample 0.9, colsample 0.9, λ 5","2.671","2.768","1.289","5.310","0.316","−1.010"],
],[2.5,5.0,1.7,1.7,1.7,1.4,1.1,1.3],numcols_from=2)
small("All models use a log1p target and identical preprocessing; ridge additionally standardises inputs. CV = 4-fold "
    "patient-grouped cross-validation on the training set (negative MAE optimised). Final model pre-specified by lowest "
    "CV MAE (extra trees); random forest and XGBoost are statistically indistinguishable.")

cap(4,"Prospective comparison: all models versus the senior physician (matched cohort, n = 360).")
table([
 ["Method","MAE (d)","Median AE (d)","RMSE (d)","R²","Bias (d)"],
 ["Senior physician","2.60","0.93","5.25","0.25","−0.62"],
 ["Extra trees (final ML model)","3.63","2.23","6.12","−0.02","−1.12"],
 ["Random forest","3.72","2.43","6.09","−0.01","−0.75"],
 ["XGBoost","3.78","2.45","6.10","−0.01","−0.64"],
 ["Ridge","3.90","2.77","6.07","−0.00","−0.42"],
],[5.6,2.2,2.4,2.0,1.6,2.0],numcols_from=1)
small("Negative R² indicates worse agreement than predicting the cohort mean. Same metric definitions as Table 3.")

cap(5,"Most important predictors of the final model (permutation importance, hold-out test set).")
table([
 ["Rank","Predictor","Δ MAE when permuted (days)"],
 ["1","Intensive-care complex treatment, base tier (OPS 8-98f.0)","0.96"],
 ["2","Intensive-care complex treatment, extended (OPS 8-98f.10)","0.21"],
 ["3","ICU care-unit type","0.14"],
 ["4","Extended haemodynamic monitoring (OPS 8-931)","0.14"],
 ["5","Cardiac monitoring (OPS 8-924)","0.07"],
 ["6","Number of procedures in first 24 h","0.07"],
 ["7","Intensive-care complex treatment, prolonged (OPS 8-98f.11)","0.06"],
 ["8","Ventilator dependence (ICD-10 Z99.1)","0.03"],
 ["9","ICU stay number (per patient)","0.03"],
 ["10","Age","0.02"],
],[1.4,10.6,4.2],numcols_from=2)
small("Δ MAE = increase in mean absolute error (days) when the predictor is randomly permuted (mean over 10 repeats); "
    "larger = more important.")

doc.save(str(OUT))
print("Gespeichert:",OUT)
print("Groesse:",round(OUT.stat().st_size/1024,1),"KB")
