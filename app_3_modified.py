"""
Streamlit app: classify samples described by image features stored in Excel files.

Upload one Excel file per class, choose a classifier, and the app searches over
feature combinations to find the best-performing subset using cross-validation.
Cluster plots are always available (raw 2D/3D scatter with a decision-boundary
overlay when 2-3 features are used, PCA projection otherwise), and results can
be exported as a one-page PDF report.

Run with:
    streamlit run app.py
"""

import io
import itertools
import time

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from joblib import Parallel, delayed

from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable,
)


# --------------------------------------------------------------------------- #
# Palette / branding tokens (kept in one place so the whole app stays coherent)
# --------------------------------------------------------------------------- #

INK = "#1B2A2E"
INK_SOFT = "#5C6F73"
PRIMARY = "#163B44"
ACCENT = "#1E7F76"
ACCENT_SOFT = "#DCEFEC"
BORDER = "#DCE6E6"
SURFACE = "#FFFFFF"
BG = "#F5F8F8"
ALERT = "#B0413E"
CLASS_COLORS = ["#1E7F76", "#B0413E", "#8A6D3B", "#3B5B8A", "#6B4C8A", "#3B8A6D"]

# Presentation branding — replace these three values when the hospital name/logo is available.
HOSPITAL_NAME = ""
DEPARTMENT_NAME = "Medical Imaging Analysis"
APP_NAME = "Clinical Image Feature Analysis"


# --------------------------------------------------------------------------- #
# Classifiers
# --------------------------------------------------------------------------- #

CLASSIFIERS = {
    "MDC (Nearest Centroid)": lambda **kw: NearestCentroid(),
    "KNN": lambda k=3, **kw: KNeighborsClassifier(n_neighbors=k),
    "LDA": lambda **kw: LinearDiscriminantAnalysis(),
    "Logistic Regression": lambda **kw: LogisticRegression(class_weight="balanced", max_iter=1000),
    "Bayesian (Gaussian NB)": lambda **kw: GaussianNB(),
    "Perceptron": lambda **kw: Perceptron(tol=1e-3, random_state=0),
    "MLP": lambda n_feats=2, **kw: MLPClassifier(
        solver="lbfgs", alpha=1e-5, max_iter=1000,
        hidden_layer_sizes=(max(n_feats, 1) * 2, 2), random_state=1,
    ),
    "SVM (NuSVC)": lambda **kw: NuSVC(kernel="rbf", degree=3, class_weight="balanced", probability=True),
    "Random Forest": lambda **kw: RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1),
    "CART (Decision Tree)": lambda **kw: DecisionTreeClassifier(class_weight="balanced"),
}


def build_model(name, n_feats, knn_k):
    return CLASSIFIERS[name](k=knn_k, n_feats=n_feats)


# --------------------------------------------------------------------------- #
# Data loading (cached so re-running with the same files/settings is instant)
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def read_excel_features(file_bytes):
    df = pd.read_excel(io.BytesIO(file_bytes))
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.to_numpy(), list(numeric_df.columns)


@st.cache_data(show_spinner=False)
def load_classes(files_bytes, class_labels):
    parsed = [read_excel_features(b) for b in files_bytes]
    feature_names = parsed[0][1]
    for _, fNames in parsed[1:]:
        feature_names = [n for n in feature_names if n in fNames]
    if not feature_names:
        raise ValueError("Uploaded files have no common numeric feature columns.")

    aligned = []
    for X_c, fNames in parsed:
        idx = [fNames.index(n) for n in feature_names]
        aligned.append(X_c[:, idx])

    X = np.concatenate(aligned, axis=0)
    y = np.concatenate([np.full(a.shape[0], i, dtype=int) for i, a in enumerate(aligned)])
    return X, y, feature_names


# --------------------------------------------------------------------------- #
# Feature reduction (significance-test ranking, adapted from HOMEWORK_04's
# signTestRanking: rank every feature by how well it separates the classes,
# keep only the significant ones, and cap the total so the downstream
# combination search stays tractable.)
# --------------------------------------------------------------------------- #

def rank_features_by_significance(X, y, test="ttest", p_threshold=0.05, max_features=20, min_features=2):
    """Rank features by univariate significance test p-value (ascending) and
    keep the significant ones (p <= p_threshold), capped at max_features.
    Falls back to the top `min_features` by p-value if too few survive the
    threshold, so the pipeline always has something to search over.
    Uses an independent t-test / Mann-Whitney U for 2 classes, and one-way
    ANOVA / Kruskal-Wallis for 3+ classes (the homework code only handled the
    2-class case; this generalizes it)."""
    from scipy import stats

    classes = np.unique(y)
    n_feats = X.shape[1]
    pvals = np.ones(n_feats)

    for j in range(n_feats):
        groups = [X[y == c, j] for c in classes]
        try:
            if len(classes) == 2:
                if test == "ttest":
                    _, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                else:
                    _, p = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            else:
                if test == "ttest":
                    _, p = stats.f_oneway(*groups)
                else:
                    _, p = stats.kruskal(*groups)
            pvals[j] = p if not np.isnan(p) else 1.0
        except Exception:
            pvals[j] = 1.0

    order = np.argsort(pvals)  # most significant first
    significant = [i for i in order if pvals[i] <= p_threshold]

    if len(significant) < min_features:
        keep = order[:max(min_features, 1)]
        used_fallback = True
    else:
        keep = np.asarray(significant[:max_features])
        used_fallback = False

    return keep, pvals, used_fallback


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def cv_accuracy(X_sub, y, model_name, knn_k, cv_method, n_folds):
    """Cross-validated predictions for one feature subset.
    cv_method: 'loocv' (exact, slower) or 'kfold' (fast, stratified)."""
    n_feats = X_sub.shape[1]
    model = build_model(model_name, n_feats, knn_k)
    if cv_method == "loocv":
        splitter = LeaveOneOut()
    else:
        splitter = StratifiedKFold(n_splits=min(n_folds, np.min(np.bincount(y))), shuffle=True, random_state=0)
    y_pred = cross_val_predict(model, X_sub, y, cv=splitter, n_jobs=1)
    accuracy = np.mean(y_pred == y)
    fitted_model = build_model(model_name, n_feats, knn_k).fit(X_sub, y)
    return accuracy, y_pred, fitted_model


def _evaluate_combo(combo, X, y, feature_names, model_name, knn_k, cv_method, n_folds):
    X_sub = X[:, combo]
    try:
        acc, y_pred, model = cv_accuracy(X_sub, y, model_name, knn_k, cv_method, n_folds)
    except Exception:
        return None
    return {
        "features": combo,
        "feature_names": [feature_names[i] for i in combo],
        "accuracy": acc,
        "y_pred": y_pred,
        "model": model,
    }


def search_feature_combinations(X, y, feature_names, n_feats_combo, model_name,
                                  knn_k, max_combos, cv_method, n_folds, n_jobs=-1):
    all_combos = list(itertools.combinations(range(X.shape[1]), n_feats_combo))
    truncated = len(all_combos) > max_combos
    if truncated:
        all_combos = all_combos[:max_combos]

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_evaluate_combo)(combo, X, y, feature_names, model_name, knn_k, cv_method, n_folds)
        for combo in all_combos
    )
    results = [r for r in results if r is not None]
    results.sort(key=lambda r: r["accuracy"], reverse=True)
    return results, truncated, len(all_combos)


# --------------------------------------------------------------------------- #
# Cluster plots
# --------------------------------------------------------------------------- #

PLOTLY_LAYOUT = dict(
    font=dict(family="IBM Plex Sans, sans-serif", color=INK, size=13),
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    title_font=dict(family="Source Serif 4, serif", size=18, color=PRIMARY),
    legend_title_text="",
    margin=dict(t=60, l=10, r=10, b=10),
)


def plot_2d_scatter(X_sub, y, class_labels, feature_names, max_acc, classifier_name):
    """2D scatter plot following HOMEWORK_04's scatterDiagrams2d."""
    fig = go.Figure()
    if len(class_labels) >= 1:
        mask = y == 0
        fig.add_trace(go.Scatter(x=X_sub[mask, 0], y=X_sub[mask, 1], mode="markers",
                                 name=class_labels[0], marker=dict(symbol="circle", size=10, color="#0000FF")))
    if len(class_labels) >= 2:
        mask = y == 1
        fig.add_trace(go.Scatter(x=X_sub[mask, 0], y=X_sub[mask, 1], mode="markers",
                                 name=class_labels[1], marker=dict(symbol="x", size=10, color="#FF0000", line=dict(width=2))))
    extra_symbols = ["diamond", "square", "triangle-up", "triangle-down"]
    extra_colors = ["#8A6D3B", "#3B5B8A", "#6B4C8A", "#3B8A6D"]
    for i in range(2, len(class_labels)):
        mask = y == i
        fig.add_trace(go.Scatter(x=X_sub[mask, 0], y=X_sub[mask, 1], mode="markers",
                                 name=class_labels[i], marker=dict(symbol=extra_symbols[(i-2)%len(extra_symbols)], size=10, color=extra_colors[(i-2)%len(extra_colors)])))
    title = (f"Scatter diagram of : {feature_names[0]} Vs {feature_names[1]}<br>"
             f"classifier: {classifier_name} accuracy: {max_acc:.2f}%")
    fig.update_layout(title=title, xaxis_title=feature_names[0], yaxis_title=feature_names[1], **PLOTLY_LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor=BORDER)
    fig.update_yaxes(showgrid=True, gridcolor=BORDER)
    return fig


def plot_3d_scatter(X_sub, y, class_labels, feature_names, max_acc, classifier_name):
    """3D scatter plot following HOMEWORK_04's scatterDiagrams3d."""
    fig = go.Figure()
    if len(class_labels) >= 1:
        mask = y == 0
        fig.add_trace(go.Scatter3d(x=X_sub[mask, 0], y=X_sub[mask, 1], z=X_sub[mask, 2], mode="markers",
                                   name=class_labels[0], marker=dict(symbol="x", size=6, color="#FF0000")))
    if len(class_labels) >= 2:
        mask = y == 1
        fig.add_trace(go.Scatter3d(x=X_sub[mask, 0], y=X_sub[mask, 1], z=X_sub[mask, 2], mode="markers",
                                   name=class_labels[1], marker=dict(symbol="circle", size=6, color="#0000FF")))
    extra_symbols = ["diamond", "square", "cross", "diamond-open"]
    extra_colors = ["#8A6D3B", "#3B5B8A", "#6B4C8A", "#3B8A6D"]
    for i in range(2, len(class_labels)):
        mask = y == i
        fig.add_trace(go.Scatter3d(x=X_sub[mask, 0], y=X_sub[mask, 1], z=X_sub[mask, 2], mode="markers",
                                   name=class_labels[i], marker=dict(symbol=extra_symbols[(i-2)%len(extra_symbols)], size=6, color=extra_colors[(i-2)%len(extra_colors)])))
    title = (f"{feature_names[0]} Vs {feature_names[1]} Vs {feature_names[2]}<br>"
             f"classifier: {classifier_name} accuracy: {max_acc:.2f}%")
    fig.update_layout(title=title,
                      scene=dict(xaxis_title=feature_names[0], yaxis_title=feature_names[1], zaxis_title=feature_names[2]),
                      scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)), **PLOTLY_LAYOUT)
    return fig


def render_cluster_section(X_sub, y, model, model_name, knn_k, class_labels, feature_names, accuracy_pct):
    """Render the direct 2D/3D scatter plots used in HOMEWORK_04.

    No decision-boundary overlay or PCA projection is added to the result plot.
    """
    n_feats = X_sub.shape[1]
    if n_feats == 2:
        fig = plot_2d_scatter(X_sub, y, class_labels, feature_names, accuracy_pct, model_name)
        caption = f"2D scatter of the selected features: {feature_names[0]} vs {feature_names[1]}, following HOMEWORK_04."
        info = {"kind":"2d", "X":X_sub, "y":y, "class_labels":class_labels, "names":feature_names,
                "title":f"Scatter diagram of : {feature_names[0]} Vs {feature_names[1]}",
                "accuracy":accuracy_pct, "classifier_name":model_name}
    elif n_feats == 3:
        fig = plot_3d_scatter(X_sub, y, class_labels, feature_names, accuracy_pct, model_name)
        caption = f"3D scatter of the selected features: {', '.join(feature_names)}, following HOMEWORK_04."
        info = {"kind":"3d", "X":X_sub, "y":y, "class_labels":class_labels, "names":feature_names,
                "title":f"{feature_names[0]} Vs {feature_names[1]} Vs {feature_names[2]}",
                "accuracy":accuracy_pct, "classifier_name":model_name}
    else:
        st.warning(f"HOMEWORK_04 scatter diagrams are defined for exactly 2 or 3 features. The current result uses {n_feats} features. Select 2 features for 2D or 3 features for 3D.")
        return f"No HOMEWORK_04 scatter plot: {n_feats} features were selected.", None
    st.plotly_chart(fig, use_container_width=True)
    st.caption(caption)
    return caption, info

def plot_roc(X_sub, y, model, title):
    if not hasattr(model, "predict_proba"):
        return None, None
    fpr, tpr, _ = roc_curve(y, model.predict_proba(X_sub)[:, 1])
    auc = roc_auc_score(y, model.predict_proba(X_sub)[:, 1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC", line=dict(color=ACCENT, width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                              line=dict(dash="dash", color=INK_SOFT, width=1)))
    fig.update_layout(title=f"{title} (AUC = {auc:.2f})",
                       xaxis_title="False positive rate", yaxis_title="True positive rate", **PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER)
    return fig, auc


def plot_confusion_matrix(cm, class_labels):
    fig = px.imshow(cm, x=class_labels, y=class_labels, text_auto=True,
                     color_continuous_scale=[[0, SURFACE], [1, ACCENT]],
                     labels=dict(x="Predicted", y="True", color="Count"))
    fig.update_layout(title="Confusion matrix", **PLOTLY_LAYOUT)
    fig.update_coloraxes(showscale=False)
    return fig



# --------------------------------------------------------------------------- #
# Static (matplotlib) figure builders — used only for the PDF export, since
# they have no headless-browser dependency and work anywhere Python runs.
# --------------------------------------------------------------------------- #

MPL_COLORS = CLASS_COLORS


def mpl_cluster_figure(info):
    """Static HOMEWORK_04-style scatter figure for the PDF export."""
    kind=info["kind"]; X=info["X"]; y=info["y"]; class_labels=info["class_labels"]; names=info["names"]
    accuracy=info.get("accuracy",0); classifier_name=info.get("classifier_name","")
    if kind == "2d":
        fig, ax = plt.subplots(figsize=(6.6,4.2))
        if len(class_labels)>=1:
            mask=y==0; ax.plot(X[mask,0],X[mask,1],"o",color="b",label=class_labels[0])
        if len(class_labels)>=2:
            mask=y==1; ax.plot(X[mask,0],X[mask,1],"x",color="r",label=class_labels[1])
        extra_markers=["D","s","^","v"]; extra_colors=["#8A6D3B","#3B5B8A","#6B4C8A","#3B8A6D"]
        for i in range(2,len(class_labels)):
            mask=y==i; ax.plot(X[mask,0],X[mask,1],marker=extra_markers[(i-2)%4],linestyle="None",color=extra_colors[(i-2)%4],label=class_labels[i])
        ax.set_xlabel(names[0]); ax.set_ylabel(names[1])
        ax.set_title(f"Scatter diagram of : {names[0]} Vs {names[1]}\nclassifier: {classifier_name} accuracy: {accuracy:.2f}%")
        ax.grid(alpha=0.25)
    else:
        fig=plt.figure(figsize=(6.6,4.6)); ax=fig.add_subplot(projection="3d")
        if len(class_labels)>=1:
            mask=y==0; ax.scatter(X[mask,0],X[mask,1],X[mask,2],color="r",marker="x",s=50,alpha=1,label=class_labels[0])
        if len(class_labels)>=2:
            mask=y==1; ax.scatter(X[mask,0],X[mask,1],X[mask,2],color="b",marker="o",s=50,alpha=1,label=class_labels[1])
        extra_markers=["D","s","^","v"]; extra_colors=["#8A6D3B","#3B5B8A","#6B4C8A","#3B8A6D"]
        for i in range(2,len(class_labels)):
            mask=y==i; ax.scatter(X[mask,0],X[mask,1],X[mask,2],color=extra_colors[(i-2)%4],marker=extra_markers[(i-2)%4],s=50,alpha=1,label=class_labels[i])
        ax.set_xlabel(names[0]); ax.set_ylabel(names[1]); ax.set_zlabel(names[2])
        ax.set_title(f"{names[0]} Vs {names[1]} Vs {names[2]}\nclassifier: {classifier_name} accuracy: {accuracy:.2f}%")
    ax.legend(loc="best",fontsize=8,frameon=True); fig.tight_layout(); return fig

def mpl_roc_figure(X_sub, y, model, title):
    if not hasattr(model, "predict_proba"):
        return None
    fpr, tpr, _ = roc_curve(y, model.predict_proba(X_sub)[:, 1])
    auc = roc_auc_score(y, model.predict_proba(X_sub)[:, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(fpr, tpr, color=ACCENT, linewidth=2.4, label="ROC")
    ax.plot([0, 1], [0, 1], color=INK_SOFT, linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{title} (AUC = {auc:.2f})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# --------------------------------------------------------------------------- #
# PDF report export
# --------------------------------------------------------------------------- #

def build_pdf_report(model_name, cv_method, n_folds, class_labels, counts,
                      feature_names_used, accuracy, cm, cluster_info, cluster_caption,
                      roc_data=None):
    """roc_data, if provided, is (X_sub, y, model, title) used to draw the ROC curve."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                             leftMargin=0.7 * inch, rightMargin=0.7 * inch)
    styles = getSampleStyleSheet()
    ink = rl_colors.HexColor(INK)
    primary = rl_colors.HexColor(PRIMARY)
    accent = rl_colors.HexColor(ACCENT)
    ink_soft = rl_colors.HexColor(INK_SOFT)
    border = rl_colors.HexColor(BORDER)
    accent_soft = rl_colors.HexColor(ACCENT_SOFT)

    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], textColor=primary,
                                  fontName="Helvetica-Bold", fontSize=22, spaceAfter=2)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], textColor=ink_soft, fontSize=10, spaceAfter=14)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], textColor=primary, fontSize=13, spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle("Body", parent=styles["Normal"], textColor=ink, fontSize=10, leading=14)
    metric_style = ParagraphStyle("Metric", parent=styles["Normal"], textColor=accent, fontSize=22,
                                   fontName="Helvetica-Bold", leading=26, spaceAfter=12)
    caption_style = ParagraphStyle("Caption", parent=styles["Normal"], textColor=ink_soft, fontSize=9, leading=12, spaceAfter=8)
    disclaimer_style = ParagraphStyle("Disclaimer", parent=styles["Normal"], textColor=ink, fontSize=8.5, leading=12,
                                       backColor=rl_colors.HexColor("#FBEFEE"), borderPadding=8)

    method_desc = "Leave-one-out cross-validation" if cv_method == "loocv" else f"{n_folds}-fold stratified cross-validation"

    story = [
        Paragraph("Image Feature Classification Report", title_style),
        Paragraph("Generated by the Image Feature Classifier tool", subtitle_style),
        HRFlowable(width="100%", thickness=1, color=border),

        Paragraph("Dataset", h2_style),
        Table(
            [["Class", "Samples"]] + [[c, str(n)] for c, n in counts.items()],
            colWidths=[3 * inch, 1.5 * inch],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), primary),
                ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, border),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]),
        ),

        Paragraph("Method", h2_style),
        Paragraph(
            f"<b>Classifier:</b> {model_name}<br/><b>Validation:</b> {method_desc}<br/>"
            f"<b>Features used ({len(feature_names_used)}):</b> {', '.join(feature_names_used)}",
            body_style,
        ),

        Paragraph("Result", h2_style),
        Paragraph(f"{accuracy*100:.2f}% cross-validated accuracy", metric_style),
        Table(
            [[""] + list(class_labels)] + [[class_labels[i]] + [str(v) for v in row] for i, row in enumerate(cm)],
            colWidths=[1.3 * inch] + [1.1 * inch] * len(class_labels),
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), accent_soft),
                ("BACKGROUND", (0, 1), (0, -1), accent_soft),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, border),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]),
        ),

        Paragraph("Cluster visualization", h2_style),
        Image(io.BytesIO(fig_to_png_bytes(mpl_cluster_figure(cluster_info))), width=6.2 * inch, height=4.0 * inch),
        Paragraph(cluster_caption, caption_style),
    ]

    if roc_data is not None:
        roc_fig = mpl_roc_figure(*roc_data)
        if roc_fig is not None:
            story += [
                Paragraph("ROC curve", h2_style),
                Image(io.BytesIO(fig_to_png_bytes(roc_fig)), width=4.2 * inch, height=3.8 * inch),
            ]

    story += [
        Spacer(1, 14),
        Paragraph(
            "This tool is intended for research and exploratory analysis only. It has not been "
            "validated as a diagnostic device and should not be used to guide patient care.",
            disclaimer_style,
        ),
    ]

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

st.set_page_config(page_title=APP_NAME, page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root {{
    --ink: {INK};
    --muted: {INK_SOFT};
    --navy: {PRIMARY};
    --teal: {ACCENT};
    --teal-soft: {ACCENT_SOFT};
    --border: {BORDER};
    --surface: {SURFACE};
    --bg: {BG};
}}

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
}}

.stApp {{ background: var(--bg); }}

.block-container {{
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}}

section[data-testid="stSidebar"] {{
    background: #F9FBFB;
    border-right: 1px solid var(--border);
}}

section[data-testid="stSidebar"] > div {{ padding-top: 1.2rem; }}

.brand-bar {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 24px;
    padding: 26px 30px;
    border: 1px solid #DCE8E9;
    border-radius: 18px;
    background: linear-gradient(135deg, #F8FCFC 0%, #EEF7F7 100%);
    box-shadow: 0 6px 24px rgba(22,59,68,.05);
    margin-bottom: 24px;
}}

.brand-kicker {{
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 7px;
}}

.brand-title {{
    font-family: 'Manrope', sans-serif;
    font-size: 31px;
    line-height: 1.14;
    font-weight: 800;
    color: var(--navy);
    margin: 0;
}}

.brand-subtitle {{
    font-size: 14px;
    color: var(--muted);
    margin-top: 8px;
}}

.brand-badge {{
    white-space: nowrap;
    padding: 9px 13px;
    border-radius: 999px;
    border: 1px solid #CFE2E0;
    background: white;
    color: var(--navy);
    font-size: 12px;
    font-weight: 700;
}}

.section-label {{
    font-family: 'Manrope', sans-serif;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 6px 0 9px;
}}

.summary-card {{
    background: white;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 17px 19px;
    box-shadow: 0 4px 18px rgba(22,59,68,.04);
}}

.summary-label {{
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 5px;
}}

.summary-value {{
    font-family: 'Manrope', sans-serif;
    font-size: 25px;
    font-weight: 800;
    color: var(--navy);
}}

.result-hero {{
    background: linear-gradient(135deg, #163B44 0%, #1E7F76 100%);
    color: white;
    border-radius: 18px;
    padding: 22px 25px;
    margin: 10px 0 20px;
    box-shadow: 0 8px 30px rgba(22,59,68,.12);
}}

.result-hero .eyebrow {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    opacity: .78;
    margin-bottom: 5px;
}}

.result-hero .headline {{
    font-family: 'Manrope', sans-serif;
    font-size: 30px;
    line-height: 1.15;
    font-weight: 800;
    margin: 0 0 7px;
}}

.result-hero .detail {{
    font-size: 14px;
    opacity: .9;
}}

.clinical-note {{
    border-left: 4px solid #7B9094;
    background: #F4F7F7;
    padding: 11px 15px;
    border-radius: 0 10px 10px 0;
    color: #405357;
    font-size: 12px;
    line-height: 1.55;
    margin-top: 18px;
}}

.sidebar-title {{
    font-family: 'Manrope', sans-serif;
    font-size: 19px;
    font-weight: 800;
    color: var(--navy);
    margin-bottom: 2px;
}}

.step-heading {{
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .07em;
    text-transform: uppercase;
    color: var(--teal);
    margin: 16px 0 8px;
}}

div[data-testid="stMetric"] {{
    background: white;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 12px 14px;
    box-shadow: 0 3px 14px rgba(22,59,68,.035);
}}

div[data-testid="stMetricLabel"] {{ color: var(--muted); font-weight: 600; }}
div[data-testid="stMetricValue"] {{ color: var(--navy); font-family: 'Manrope', sans-serif; }}

button[kind="primary"] {{
    border-radius: 11px !important;
    font-weight: 700 !important;
}}

.stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px 10px 0 0;
    padding-left: 14px; padding-right: 14px;
}}

.disclaimer {{
    border-left: 4px solid #8A9A9D;
    background: #F2F5F5;
    padding: 11px 14px;
    border-radius: 0 9px 9px 0;
    color: #44575B;
    font-size: 12px;
    line-height: 1.55;
    margin-top: 22px;
}}

footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="brand-bar">
  <div>
    <div class="brand-kicker">{HOSPITAL_NAME}</div>
    <div class="brand-title">{APP_NAME}</div>
    <div class="brand-subtitle">{DEPARTMENT_NAME} · Quantitative analysis of extracted image features</div>
  </div>
  <div class="brand-badge">Research &amp; Clinical Presentation Prototype</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="clinical-note"><b>Presentation note:</b> This prototype is designed to support discussion of feature-level classification results. It does not replace clinical judgment and is not intended for direct diagnosis or treatment decisions.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-title">Analysis setup</div>', unsafe_allow_html=True)
    st.caption("Configure the dataset and validation strategy before running the analysis.")
    st.markdown('<div class="step-heading">01 · Dataset</div>', unsafe_allow_html=True)
    n_classes = st.number_input("Number of classes", min_value=2, max_value=6, value=2, step=1)

    uploaded_files, class_labels = [], []
    for i in range(int(n_classes)):
        c1, c2 = st.columns([2, 1])
        with c1:
            f = st.file_uploader(f"Feature file · class {i + 1}", type=["xlsx", "xls"], key=f"file_{i}")
        with c2:
            label = st.text_input(f"Class label {i + 1}", value=f"Class {i + 1}", key=f"label_{i}")
        uploaded_files.append(f)
        class_labels.append(label)

    st.markdown('<div class="step-heading">02 · Analysis model</div>', unsafe_allow_html=True)
    model_name = st.selectbox("Analysis model", list(CLASSIFIERS.keys()))
    knn_k = st.slider("K (neighbors)", 1, 15, 3) if model_name == "KNN" else 3

    st.markdown('<div class="step-heading">03 · Validation</div>', unsafe_allow_html=True)
    cv_method = st.radio(
        "Method", ["kfold", "loocv"], format_func=lambda x: "Stratified k-fold" if x == "kfold" else "Leave-one-out",
    )
    n_folds = st.slider("Number of folds", 3, 10, 5) if cv_method == "kfold" else 5

    st.markdown('<div class="step-heading">04 · Feature screening</div>', unsafe_allow_html=True)
    st.caption("Screens features by statistical separation before evaluating candidate combinations.")
    enable_reduction = st.checkbox("Enable significance-based feature reduction", value=True)
    p_threshold = st.slider("Significance threshold (p-value)", 0.01, 0.20, 0.05, 0.01) if enable_reduction else 0.05
    max_features_kept = st.number_input("Max features to keep", min_value=2, max_value=100, value=20) if enable_reduction else None

    st.markdown('<div class="step-heading">05 · Feature search</div>', unsafe_allow_html=True)
    st.caption("Evaluates candidate feature subsets and reports the highest cross-validated result.")
    n_feats_combo = st.number_input("Features per combination", min_value=2, max_value=3, value=2)
    max_combos = st.number_input("Max combinations to try", min_value=10, max_value=20000, value=500, step=10)
    st.caption("Parallelized search is used to keep exploratory analysis responsive.")

    run = st.button("Run analysis", type="primary", use_container_width=True)

if run:
    if not all(uploaded_files) or len(set(class_labels)) != len(class_labels):
        st.error("Please upload a file for every class (and use distinct labels).")
        st.stop()

    files_bytes = tuple(f.getvalue() for f in uploaded_files)
    labels_tuple = tuple(class_labels)

    with st.spinner("Loading data..."):
        try:
            X, y, feature_names = load_classes(files_bytes, labels_tuple)
        except Exception as e:
            st.error(f"Could not load data: {e}")
            st.stop()

    n_features_before_reduction = len(feature_names)
    reduction_pvals = None
    reduction_fallback = False
    if enable_reduction:
        keep_idx, reduction_pvals, reduction_fallback = rank_features_by_significance(
            X, y, test="ttest", p_threshold=p_threshold, max_features=int(max_features_kept),
        )
        X = X[:, keep_idx]
        feature_names = [feature_names[i] for i in keep_idx]
        reduction_pvals = reduction_pvals[keep_idx]

    tab_overview, tab_results, tab_method = st.tabs(["Clinical overview", "Analysis results", "Methodology"])

    with tab_overview:
        st.subheader("Study overview")
        counts = {class_labels[c]: int(np.sum(y == c)) for c in np.unique(y)}
        cols = st.columns(len(counts) + 1)
        cols[0].metric("Total samples", len(y))
        for i, (label, n) in enumerate(counts.items()):
            cols[i + 1].metric(label, n)
        st.write(pd.DataFrame({"class": list(counts.keys()), "n samples": list(counts.values())}))
        st.caption(f"{n_features_before_reduction} common numeric features detected across the uploaded datasets.")

        if enable_reduction:
            st.subheader("Feature reduction")
            if reduction_fallback:
                st.warning(
                    f"No feature reached p ≤ {p_threshold:.2f}, so the {len(feature_names)} "
                    "most significant features were kept anyway (fallback) to keep the search usable."
                )
            else:
                st.success(
                    f"Kept {len(feature_names)} of {n_features_before_reduction} features "
                    f"significant at p ≤ {p_threshold:.2f} (t-test / ANOVA), capped at {int(max_features_kept)}."
                )
            st.dataframe(
                pd.DataFrame({"feature": feature_names, "p-value": np.round(reduction_pvals, 5)})
                .sort_values("p-value").reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )

    with tab_results:
        total_combos = len(list(itertools.combinations(range(len(feature_names)), int(n_feats_combo))))
        t0 = time.time()
        with st.spinner(f"Searching up to {min(total_combos, max_combos)} of {total_combos} combinations in parallel..."):
            results, truncated, n_tried = search_feature_combinations(
                X, y, feature_names, int(n_feats_combo), model_name, knn_k,
                int(max_combos), cv_method, n_folds,
            )
        elapsed = time.time() - t0

        if not results:
            st.error("No valid results — check your data and classifier choice.")
            st.stop()

        if truncated:
            st.warning(f"{total_combos} combinations exist; search was capped at {n_tried} for speed. "
                       "Raise 'Max combinations' in the sidebar for a more exhaustive search.")

        best = results[0]
        r1, r2, r3 = st.columns(3)
        st.markdown(
            f"""<div class="result-hero">
              <div class="eyebrow">Best-performing configuration</div>
              <div class="headline">{best['accuracy'] * 100:.2f}% cross-validated accuracy</div>
              <div class="detail">Model: {model_name} &nbsp;·&nbsp; Features: {', '.join(best['feature_names'])}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        r1, r2, r3 = st.columns(3)
        r1.metric("Cross-validated accuracy", f"{best['accuracy'] * 100:.2f}%")
        r2.metric("Candidate subsets", n_tried)
        r3.metric("Analysis time", f"{elapsed:.2f}s")

        cm = confusion_matrix(y, best["y_pred"])
        st.plotly_chart(plot_confusion_matrix(cm, class_labels), use_container_width=True)

        st.subheader(f"Top candidate feature combinations")
        top_table = pd.DataFrame([
            {"Feature subset": ", ".join(r["feature_names"]), "Cross-validated accuracy (%)": round(r["accuracy"] * 100, 2)}
            for r in results[:10]
        ])
        st.dataframe(top_table, use_container_width=True, hide_index=True)

        X_sub = X[:, best["features"]]
        st.subheader("Feature separation")
        cluster_caption, cluster_info = render_cluster_section(
            X_sub, y, best["model"], model_name, knn_k, class_labels, best["feature_names"], best["accuracy"] * 100,
        )

        roc_data = None
        if len(class_labels) == 2:
            roc_title = f"{model_name} — accuracy {best['accuracy']*100:.2f}%"
            roc_fig, auc = plot_roc(X_sub, y, best["model"], roc_title)
            if roc_fig:
                st.subheader("Discrimination performance")
                st.plotly_chart(roc_fig, use_container_width=True)
                roc_data = (X_sub, y, best["model"], roc_title)
            else:
                st.info(f"{model_name} does not support probability estimates, so no ROC curve is shown.")

        pdf_bytes = build_pdf_report(model_name, cv_method, n_folds, class_labels, counts,
                                      best["feature_names"], best["accuracy"], cm, cluster_info, cluster_caption, roc_data)
        st.download_button("Export presentation report (PDF)", pdf_bytes, file_name="clinical_image_analysis_report.pdf", mime="application/pdf")

    with tab_method:
        st.subheader("How to interpret the analysis")
        st.markdown(f"""
        - **Cross-validation, not training accuracy.** Every reported accuracy is computed on samples the
          model did not see during training for that fold, using either stratified k-fold or exact
          leave-one-out cross-validation.
        - **Stratified k-fold** splits the data into *k* balanced folds and rotates which fold is held out;
          it's fast and gives a stable estimate for most sample sizes.
        - **Leave-one-out** trains on all samples but one and tests on the held-out sample, repeated for
          every sample; it's exact but slower, and best reserved for small datasets.
        - **Feature screening (optional, on by default)** ranks every feature by a univariate
          significance test — an independent t-test for 2 classes, one-way ANOVA for 3+ — and keeps
          only the ones with p-value at or below the chosen threshold, capped at the chosen maximum.
          This trims noisy/uninformative features before the combination search, which both speeds up
          the search and cuts down on combinations that overfit by chance. If nothing clears the
          threshold, the most significant features are kept anyway so the search still has something
          to work with.
        - **Feature search** evaluates every (or a capped subset of) combination of the chosen size on the screened feature set and reports the highest cross-validated accuracy.
          This is an exploratory search, not a statistical guarantee — with many combinations tried,
          some will score well by chance alone.
        - **Feature-separation plots** display the selected raw features directly using the 2D or 3D scatter convention from HOMEWORK_04. They are intended to make group separation visually interpretable; they are not themselves diagnostic plots.
        """)
        st.markdown(
            '<div class="disclaimer">This tool is intended for research and exploratory analysis only. '
            'It has not been validated as a diagnostic device and should not be used to guide patient care.</div>',
            unsafe_allow_html=True,
        )
else:
    st.info("Upload the relevant feature files in the sidebar and press **Run analysis** to begin.")
