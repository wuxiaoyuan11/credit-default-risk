from __future__ import annotations

from pathlib import Path
import importlib.util
import os
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "cs-training.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURE_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "reports" / "models"
SUMMARY_PATH = ROOT / "reports" / "model_summary.txt"

TARGET = "SeriousDlqin2yrs"
RANDOM_STATE = 42
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))


def check_required_dependencies() -> None:
    required = [
        "joblib",
        "matplotlib",
        "numpy",
        "pandas",
        "sklearn",
    ]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            "Missing Python dependencies: "
            f"{missing_list}.\n\n"
            "Install them with:\n"
            "  python3 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  pip install -r requirements.txt\n"
        )


def import_required_dependencies() -> None:
    global ColumnTransformer
    global HistGradientBoostingClassifier
    global LogisticRegression
    global Pipeline
    global RandomForestClassifier
    global RocCurveDisplay
    global SimpleImputer
    global StandardScaler
    global brier_score_loss
    global classification_report
    global confusion_matrix
    global joblib
    global np
    global pd
    global plt
    global precision_recall_curve
    global permutation_importance
    global roc_auc_score
    global roc_curve
    global train_test_split

    import joblib
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        RocCurveDisplay,
        brier_score_loss,
        classification_report,
        confusion_matrix,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import StandardScaler

    plt.style.use("seaborn-v0_8-whitegrid")


def ensure_dirs() -> None:
    for path in [PROCESSED_DIR, FIGURE_DIR, MODEL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    if not RAW_DATA.exists():
        raise FileNotFoundError(
            f"Missing {RAW_DATA}. Download cs-training.csv from Kaggle Give Me Some Credit "
            "and place it in data/raw/."
        )
    df = pd.read_csv(RAW_DATA)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "MonthlyIncome" in df.columns:
        df["monthly_income_missing"] = df["MonthlyIncome"].isna().astype(int)
        income_for_bins = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
        df["income_band"] = pd.cut(
            income_for_bins,
            bins=[-np.inf, 2500, 5000, 10000, 20000, np.inf],
            labels=[1, 2, 3, 4, 5],
        ).astype(float)
    if "NumberOfDependents" in df.columns:
        df["dependents_missing"] = df["NumberOfDependents"].isna().astype(int)
    if {"DebtRatio", "MonthlyIncome"}.issubset(df.columns):
        df["estimated_monthly_debt"] = df["DebtRatio"] * df["MonthlyIncome"].fillna(
            df["MonthlyIncome"].median()
        )
        df["payment_burden_proxy"] = df["DebtRatio"]
        df["debt_to_income_ratio"] = df["DebtRatio"]
    late_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    existing_late_cols = [col for col in late_cols if col in df.columns]
    if existing_late_cols:
        df["total_past_due_events"] = df[existing_late_cols].sum(axis=1)
        df["has_past_due"] = (df["total_past_due_events"] > 0).astype(int)
    if "RevolvingUtilizationOfUnsecuredLines" in df.columns:
        df["credit_utilization"] = df["RevolvingUtilizationOfUnsecuredLines"]
        df["high_credit_utilization"] = (
            df["RevolvingUtilizationOfUnsecuredLines"] > 0.8
        ).astype(int)
    if "age" in df.columns:
        df["age_band"] = pd.cut(
            df["age"],
            bins=[0, 30, 45, 60, 75, np.inf],
            labels=[1, 2, 3, 4, 5],
        ).astype(float)
    return df


def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    feature_cols = [col for col in df.columns if col != TARGET]
    for col in feature_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)
    return df


def make_eda(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    df[TARGET].value_counts().sort_index().plot(kind="bar")
    plt.xlabel(TARGET)
    plt.ylabel("Count")
    plt.title("Default Distribution")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "01_default_distribution.png", dpi=160)
    plt.close()

    sample_df = df.sample(n=min(len(df), 20000), random_state=RANDOM_STATE)
    for index, col in enumerate(
        [
            "age",
            "RevolvingUtilizationOfUnsecuredLines",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfTimes90DaysLate",
        ],
        start=2,
    ):
        if col not in df.columns:
            continue
        plt.figure(figsize=(7, 4))
        groups = [
            sample_df.loc[sample_df[TARGET] == value, col].dropna()
            for value in sorted(sample_df[TARGET].dropna().unique())
        ]
        plt.boxplot(groups, labels=[str(value) for value in sorted(sample_df[TARGET].dropna().unique())])
        plt.xlabel(TARGET)
        plt.ylabel(col)
        plt.title(f"{col} vs Default")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"{index:02d}_{col}_vs_default.png", dpi=160)
        plt.close()

    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "07_correlation_heatmap.png", dpi=160)
    plt.close()


def ks_score(y_true: pd.Series, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= 0.2).astype(int)
    return {
        "model": name,
        "auc": roc_auc_score(y_test, y_score),
        "ks": ks_score(y_test, y_score),
        "confusion_matrix_threshold_0_2": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report_threshold_0_2": classification_report(
            y_test, y_pred, digits=4
        ),
    }


def save_curves(best_name: str, best_model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_score = best_model.predict_proba(X_test)[:, 1]

    RocCurveDisplay.from_predictions(y_test, y_score)
    plt.title(f"ROC Curve - {best_name}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "08_roc_curve.png", dpi=160)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {best_name}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "09_precision_recall_curve.png", dpi=160)
    plt.close()


def save_feature_importance(best_name: str, best_model, feature_names: list[str]) -> None:
    estimator = best_model.named_steps["model"] if isinstance(best_model, Pipeline) else best_model
    if hasattr(estimator, "feature_importances_"):
        importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": estimator.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
    else:
        return

    importance.to_csv(ROOT / "reports" / "feature_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    top = importance.head(12).sort_values("importance")
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.title(f"Top Feature Importance - {best_name}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "10_feature_importance.png", dpi=160)
    plt.close()


def save_permutation_importance(
    best_name: str,
    best_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    sample_size = min(5000, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=RANDOM_STATE)
    y_sample = y_test.loc[X_sample.index]
    result = permutation_importance(
        best_model,
        X_sample,
        y_sample,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(ROOT / "reports" / "permutation_importance.csv", index=False)

    top = importance.head(12).sort_values("importance_mean")
    plt.figure(figsize=(8, 5))
    plt.barh(top["feature"], top["importance_mean"])
    plt.xlabel("AUC decrease after permutation")
    plt.title(f"Permutation Importance - {best_name}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "11_permutation_importance.png", dpi=160)
    plt.close()


def save_business_decision_report(
    best_name: str,
    best_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    scores = best_model.predict_proba(X_test)[:, 1]
    decision_rows = []
    for threshold in [0.3, 0.5, 0.7]:
        approved = scores < threshold
        rejected = ~approved
        approved_count = int(approved.sum())
        rejected_count = int(rejected.sum())
        approved_default_rate = float(y_test[approved].mean()) if approved_count else 0.0
        rejected_default_rate = float(y_test[rejected].mean()) if rejected_count else 0.0
        bad_capture_rate = (
            float(y_test[rejected].sum() / y_test.sum()) if y_test.sum() else 0.0
        )
        decision_rows.append(
            {
                "approval_threshold": threshold,
                "approval_rate": float(approved.mean()),
                "rejection_rate": float(rejected.mean()),
                "approved_default_rate": approved_default_rate,
                "rejected_default_rate": rejected_default_rate,
                "bad_capture_rate": bad_capture_rate,
            }
        )

    decision_df = pd.DataFrame(decision_rows)
    decision_df.to_csv(ROOT / "reports" / "threshold_business_decisions.csv", index=False)

    scored = X_test.copy()
    scored["actual_default"] = y_test
    scored["risk_score"] = scores
    scored["risk_decile"] = pd.qcut(scored["risk_score"], 10, labels=False, duplicates="drop") + 1
    scored.to_csv(ROOT / "reports" / "test_scores.csv", index=False)

    segment_rows = []
    segment_specs = {
        "has_past_due": "has_past_due",
        "high_credit_utilization": "high_credit_utilization",
        "income_band": "income_band",
        "age_band": "age_band",
    }
    for segment_name, col in segment_specs.items():
        if col not in scored.columns:
            continue
        grouped = (
            scored.groupby(col, dropna=False)
            .agg(
                customers=("actual_default", "size"),
                actual_default_rate=("actual_default", "mean"),
                avg_risk_score=("risk_score", "mean"),
            )
            .reset_index()
        )
        grouped.insert(0, "segment", segment_name)
        grouped = grouped.rename(columns={col: "segment_value"})
        segment_rows.append(grouped)
    segment_df = pd.concat(segment_rows, ignore_index=True)
    segment_df.to_csv(ROOT / "reports" / "risk_segments.csv", index=False)

    top_features = pd.read_csv(ROOT / "reports" / "permutation_importance.csv").head(8)
    lines = [
        "# Business Decision Report",
        "",
        f"Best model: `{best_name}`",
        "",
        "## Threshold trade-off",
        "",
        decision_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        "Interpretation: a lower approval threshold rejects more applicants and captures more future defaulters, but it also reduces approval volume. A higher threshold approves more customers, but accepts more default risk.",
        "",
        "## High-risk segments",
        "",
        "Segments with higher predicted and observed risk are mainly driven by past-due history, high credit utilization, low or missing income, and heavy debt burden proxies.",
        "",
        "## Early warning signals",
        "",
    ]
    for feature in top_features["feature"].tolist():
        lines.append(f"- {feature}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "The Give Me Some Credit dataset does not include loan amount, loan purpose, or external credit bureau score fields, so loan-to-income and external score aggregation are documented as recommended extensions rather than implemented features.",
        ]
    )
    (ROOT / "reports" / "business_decision_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def save_decile_lift_gains(
    best_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    scores = best_model.predict_proba(X_test)[:, 1]
    scored = pd.DataFrame(
        {
            "actual_default": y_test.to_numpy(),
            "risk_score": scores,
        },
        index=X_test.index,
    ).sort_values("risk_score", ascending=False)
    scored["risk_decile"] = pd.qcut(
        scored["risk_score"].rank(method="first", ascending=False),
        10,
        labels=range(1, 11),
    ).astype(int)

    total_bads = scored["actual_default"].sum()
    total_customers = len(scored)
    base_default_rate = scored["actual_default"].mean()
    decile = (
        scored.groupby("risk_decile")
        .agg(
            customers=("actual_default", "size"),
            bads=("actual_default", "sum"),
            avg_risk_score=("risk_score", "mean"),
            min_risk_score=("risk_score", "min"),
            max_risk_score=("risk_score", "max"),
            default_rate=("actual_default", "mean"),
        )
        .reset_index()
        .sort_values("risk_decile")
    )
    decile["goods"] = decile["customers"] - decile["bads"]
    decile["customer_share"] = decile["customers"] / total_customers
    decile["bad_share"] = decile["bads"] / total_bads
    decile["cumulative_customers"] = decile["customers"].cumsum()
    decile["cumulative_bads"] = decile["bads"].cumsum()
    decile["cumulative_customer_share"] = decile["cumulative_customers"] / total_customers
    decile["cumulative_bad_capture_rate"] = decile["cumulative_bads"] / total_bads
    decile["lift"] = decile["default_rate"] / base_default_rate
    decile["cumulative_lift"] = (
        decile["cumulative_bad_capture_rate"] / decile["cumulative_customer_share"]
    )
    decile.to_csv(ROOT / "reports" / "decile_lift_gains.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(decile["risk_decile"], decile["default_rate"], marker="o")
    plt.axhline(base_default_rate, color="gray", linestyle="--", label="Portfolio default rate")
    plt.gca().invert_xaxis()
    plt.xlabel("Risk decile (1 = highest risk)")
    plt.ylabel("Observed default rate")
    plt.title("Default Rate by Risk Decile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "12_default_rate_by_decile.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(
        decile["cumulative_customer_share"],
        decile["cumulative_bad_capture_rate"],
        marker="o",
        label="Model",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("Cumulative customer share")
    plt.ylabel("Cumulative bad capture rate")
    plt.title("Cumulative Gains Chart")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "13_cumulative_gains.png", dpi=160)
    plt.close()


def _bin_feature_for_woe(feature: pd.Series) -> pd.Series:
    feature = feature.copy()
    non_missing = feature.dropna()
    if non_missing.nunique() <= 10:
        return feature.astype("object").where(feature.notna(), "Missing")
    try:
        binned = pd.qcut(feature, q=5, duplicates="drop")
    except ValueError:
        binned = pd.cut(feature, bins=5, duplicates="drop")
    return binned.astype("object").where(feature.notna(), "Missing")


def save_woe_iv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    rows = []
    eps = 0.5
    total_bad = float((y_train == 1).sum())
    total_good = float((y_train == 0).sum())

    for feature in X_train.columns:
        binned = _bin_feature_for_woe(X_train[feature])
        temp = pd.DataFrame({"bin": binned, "target": y_train})
        grouped = (
            temp.groupby("bin", dropna=False, observed=False)
            .agg(total=("target", "size"), bad=("target", "sum"))
            .reset_index()
        )
        grouped["good"] = grouped["total"] - grouped["bad"]
        grouped["bad_dist"] = (grouped["bad"] + eps) / (total_bad + eps * len(grouped))
        grouped["good_dist"] = (grouped["good"] + eps) / (total_good + eps * len(grouped))
        grouped["woe"] = np.log(grouped["good_dist"] / grouped["bad_dist"])
        grouped["iv_component"] = (grouped["good_dist"] - grouped["bad_dist"]) * grouped["woe"]
        grouped["event_rate"] = grouped["bad"] / grouped["total"]
        grouped.insert(0, "feature", feature)
        rows.append(grouped)

    woe_bins = pd.concat(rows, ignore_index=True)
    woe_bins["bin"] = woe_bins["bin"].astype(str)
    woe_bins.to_csv(ROOT / "reports" / "woe_bins.csv", index=False)

    iv_summary = (
        woe_bins.groupby("feature", as_index=False)
        .agg(iv=("iv_component", "sum"))
        .sort_values("iv", ascending=False)
    )
    iv_summary["predictive_power"] = pd.cut(
        iv_summary["iv"],
        bins=[-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
        labels=["Not useful", "Weak", "Medium", "Strong", "Suspiciously strong"],
    )
    iv_summary.to_csv(ROOT / "reports" / "iv_summary.csv", index=False)

    top = iv_summary.head(12).sort_values("iv")
    plt.figure(figsize=(8, 5))
    plt.barh(top["feature"], top["iv"])
    plt.xlabel("Information Value")
    plt.title("Top Variables by IV")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "14_information_value.png", dpi=160)
    plt.close()


def _select_scorecard_features(iv_summary: pd.DataFrame, max_features: int = 8) -> list[str]:
    selected = []
    blocked = set()
    duplicate_groups = [
        {
            "total_past_due_events",
            "has_past_due",
            "NumberOfTimes90DaysLate",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
        },
        {
            "credit_utilization",
            "RevolvingUtilizationOfUnsecuredLines",
            "high_credit_utilization",
        },
        {"age", "age_band"},
        {"DebtRatio", "debt_to_income_ratio", "payment_burden_proxy"},
    ]
    for feature in iv_summary["feature"]:
        if feature in blocked:
            continue
        selected.append(feature)
        for group in duplicate_groups:
            if feature in group:
                blocked.update(group - {feature})
        if len(selected) >= max_features:
            break
    return selected


def _fit_woe_maps(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected_features: list[str],
) -> tuple[dict, pd.DataFrame]:
    maps = {}
    rows = []
    eps = 0.5
    total_bad = float((y_train == 1).sum())
    total_good = float((y_train == 0).sum())

    for feature in selected_features:
        binned = _bin_feature_for_woe(X_train[feature])
        temp = pd.DataFrame({"bin": binned, "target": y_train})
        grouped = (
            temp.groupby("bin", dropna=False, observed=False)
            .agg(total=("target", "size"), bad=("target", "sum"))
            .reset_index()
        )
        grouped["good"] = grouped["total"] - grouped["bad"]
        grouped["bad_dist"] = (grouped["bad"] + eps) / (total_bad + eps * len(grouped))
        grouped["good_dist"] = (grouped["good"] + eps) / (total_good + eps * len(grouped))
        grouped["woe"] = np.log(grouped["good_dist"] / grouped["bad_dist"])
        grouped["event_rate"] = grouped["bad"] / grouped["total"]
        grouped.insert(0, "feature", feature)
        rows.append(grouped.copy())
        maps[feature] = {
            "woe_by_bin": dict(zip(grouped["bin"].astype(str), grouped["woe"])),
            "fallback": 0.0,
        }

    return maps, pd.concat(rows, ignore_index=True)


def _transform_woe(
    X: pd.DataFrame,
    selected_features: list[str],
    woe_maps: dict,
) -> pd.DataFrame:
    transformed = pd.DataFrame(index=X.index)
    for feature in selected_features:
        binned = _bin_feature_for_woe(X[feature]).astype(str)
        transformed[f"{feature}_woe"] = binned.map(woe_maps[feature]["woe_by_bin"]).fillna(
            woe_maps[feature]["fallback"]
        )
    return transformed


def save_scorecard_benchmark(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    champion_name: str,
    champion_metrics: pd.DataFrame,
) -> None:
    iv_summary = pd.read_csv(ROOT / "reports" / "iv_summary.csv")
    selected_features = _select_scorecard_features(iv_summary, max_features=8)
    woe_maps, selected_woe_bins = _fit_woe_maps(X_train, y_train, selected_features)
    X_train_woe = _transform_woe(X_train, selected_features, woe_maps)
    X_test_woe = _transform_woe(X_test, selected_features, woe_maps)

    scorecard = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    scorecard.fit(X_train_woe, y_train)
    score = scorecard.predict_proba(X_test_woe)[:, 1]
    auc = roc_auc_score(y_test, score)
    ks = ks_score(y_test, score)

    coefficients = pd.DataFrame(
        {
            "feature": selected_features,
            "woe_feature": X_train_woe.columns,
            "coefficient": scorecard.coef_[0],
        }
    ).sort_values("coefficient", ascending=False)

    selected_woe_bins["bin"] = selected_woe_bins["bin"].astype(str)
    selected_woe_bins.to_csv(ROOT / "reports" / "scorecard_woe_bins.csv", index=False)
    coefficients.to_csv(ROOT / "reports" / "scorecard_coefficients.csv", index=False)
    pd.DataFrame({"selected_feature": selected_features}).to_csv(
        ROOT / "reports" / "scorecard_selected_features.csv",
        index=False,
    )

    base_score = 600
    pdo = 50
    factor = pdo / np.log(2)
    base_pd = float(y_train.mean())
    base_logit = np.log(base_pd / (1 - base_pd))
    score_offset = base_score - factor * (scorecard.intercept_[0] - base_logit)

    points_rows = []
    for feature in selected_features:
        coef = float(coefficients.loc[coefficients["feature"] == feature, "coefficient"].iloc[0])
        feature_bins = selected_woe_bins[selected_woe_bins["feature"] == feature].copy()
        feature_bins["points"] = -factor * coef * feature_bins["woe"]
        points_rows.append(
            feature_bins[
                [
                    "feature",
                    "bin",
                    "total",
                    "bad",
                    "good",
                    "event_rate",
                    "woe",
                    "points",
                ]
            ]
        )
    scorecard_points = pd.concat(points_rows, ignore_index=True)
    scorecard_points.to_csv(ROOT / "reports" / "scorecard_points.csv", index=False)

    train_score = score_offset + X_train_woe.dot(
        coefficients.set_index("woe_feature").loc[X_train_woe.columns, "coefficient"].mul(-factor)
    )
    test_score = score_offset + X_test_woe.dot(
        coefficients.set_index("woe_feature").loc[X_test_woe.columns, "coefficient"].mul(-factor)
    )
    score_output = pd.DataFrame(
        {
            "actual_default": y_test.to_numpy(),
            "predicted_pd": score,
            "credit_score": test_score.to_numpy(),
        },
        index=X_test.index,
    )
    score_output.to_csv(ROOT / "reports" / "scorecard_test_scores.csv", index=False)

    score_params = pd.DataFrame(
        [
            {
                "base_score": base_score,
                "pdo": pdo,
                "base_pd": base_pd,
                "base_logit": base_logit,
                "factor": factor,
                "score_offset": score_offset,
                "train_min_score": float(train_score.min()),
                "train_max_score": float(train_score.max()),
                "test_min_score": float(test_score.min()),
                "test_max_score": float(test_score.max()),
            }
        ]
    )
    score_params.to_csv(ROOT / "reports" / "scorecard_scaling_params.csv", index=False)

    champion = champion_metrics.loc[champion_metrics["model"] == champion_name].iloc[0]
    comparison = pd.DataFrame(
        [
            {
                "model": "woe_logistic_scorecard",
                "role": "Traditional interpretable benchmark",
                "auc": auc,
                "ks": ks,
            },
            {
                "model": champion_name,
                "role": "Machine learning champion",
                "auc": champion["auc"],
                "ks": champion["ks"],
            },
        ]
    )
    comparison["auc_gap_vs_champion"] = comparison["auc"] - float(champion["auc"])
    comparison["ks_gap_vs_champion"] = comparison["ks"] - float(champion["ks"])
    comparison.to_csv(ROOT / "reports" / "scorecard_vs_ml_comparison.csv", index=False)

    scorecard_decile = pd.DataFrame({"actual_default": y_test.to_numpy(), "risk_score": score})
    scorecard_decile = scorecard_decile.sort_values("risk_score", ascending=False)
    scorecard_decile["risk_decile"] = pd.qcut(
        scorecard_decile["risk_score"].rank(method="first", ascending=False),
        10,
        labels=range(1, 11),
    ).astype(int)
    total_bads = scorecard_decile["actual_default"].sum()
    decile = (
        scorecard_decile.groupby("risk_decile")
        .agg(
            customers=("actual_default", "size"),
            bads=("actual_default", "sum"),
            avg_risk_score=("risk_score", "mean"),
            default_rate=("actual_default", "mean"),
        )
        .reset_index()
        .sort_values("risk_decile")
    )
    decile["cumulative_bad_capture_rate"] = decile["bads"].cumsum() / total_bads
    decile.to_csv(ROOT / "reports" / "scorecard_decile.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.hist(score_output.loc[score_output["actual_default"] == 0, "credit_score"], bins=30, alpha=0.7, label="Non-default")
    plt.hist(score_output.loc[score_output["actual_default"] == 1, "credit_score"], bins=30, alpha=0.7, label="Default")
    plt.xlabel("Scorecard credit score")
    plt.ylabel("Borrowers")
    plt.title("Scorecard Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "18_scorecard_score_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.barh(coefficients["feature"], coefficients["coefficient"])
    plt.xlabel("Logistic coefficient on WOE-transformed feature")
    plt.title("WOE Logistic Scorecard Coefficients")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "17_scorecard_coefficients.png", dpi=160)
    plt.close()

    report_lines = [
        "# WOE Logistic Scorecard Benchmark",
        "",
        "## Objective",
        "",
        "Build a compact traditional credit scorecard benchmark using IV-selected variables and WOE transformations, then compare it with the machine learning champion model.",
        "",
        "## Selected Variables",
        "",
    ]
    report_lines.extend([f"- {feature}" for feature in selected_features])
    report_lines.extend(
        [
            "",
            "## Model Comparison",
            "",
            comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
            "",
            "## Interpretation",
            "",
            "The WOE logistic scorecard is more transparent and easier to explain to credit policy stakeholders. The machine learning champion can capture more complex non-linear relationships and interactions. The comparison shows the trade-off between interpretability and predictive performance.",
            "",
            "## Output Files",
            "",
            "- reports/scorecard_selected_features.csv",
            "- reports/scorecard_woe_bins.csv",
            "- reports/scorecard_coefficients.csv",
            "- reports/scorecard_points.csv",
            "- reports/scorecard_scaling_params.csv",
            "- reports/scorecard_test_scores.csv",
            "- reports/scorecard_vs_ml_comparison.csv",
            "- reports/scorecard_decile.csv",
            "- reports/figures/17_scorecard_coefficients.png",
            "- reports/figures/18_scorecard_score_distribution.png",
        ]
    )
    (ROOT / "reports" / "scorecard_benchmark_report.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )


def save_calibration_analysis(
    best_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    scores = best_model.predict_proba(X_test)[:, 1]
    calibration = pd.DataFrame({"actual_default": y_test.to_numpy(), "risk_score": scores})
    calibration["score_bin"] = pd.qcut(
        calibration["risk_score"].rank(method="first"),
        10,
        labels=range(1, 11),
    ).astype(int)
    calibration_table = (
        calibration.groupby("score_bin")
        .agg(
            customers=("actual_default", "size"),
            avg_predicted_pd=("risk_score", "mean"),
            observed_default_rate=("actual_default", "mean"),
        )
        .reset_index()
    )
    calibration_table["prediction_error"] = (
        calibration_table["avg_predicted_pd"]
        - calibration_table["observed_default_rate"]
    )
    brier = brier_score_loss(y_test, scores)
    calibration_table["brier_score"] = brier
    calibration_table.to_csv(ROOT / "reports" / "calibration_table.csv", index=False)

    plt.figure(figsize=(6, 5))
    plt.plot(
        calibration_table["avg_predicted_pd"],
        calibration_table["observed_default_rate"],
        marker="o",
        label="Model bins",
    )
    limit = max(
        calibration_table["avg_predicted_pd"].max(),
        calibration_table["observed_default_rate"].max(),
    )
    plt.plot([0, limit], [0, limit], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Average predicted probability")
    plt.ylabel("Observed default rate")
    plt.title(f"Calibration Plot (Brier={brier:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "15_calibration_plot.png", dpi=160)
    plt.close()


def save_shap_analysis(
    best_name: str,
    best_model,
    X_test: pd.DataFrame,
) -> None:
    shap_note_path = ROOT / "reports" / "shap_status.txt"
    if importlib.util.find_spec("shap") is None:
        shap_note_path.write_text(
            "SHAP is not installed. Install it with `pip install shap` and rerun the pipeline.\n",
            encoding="utf-8",
        )
        return

    try:
        import shap

        sample = X_test.sample(n=min(1000, len(X_test)), random_state=RANDOM_STATE)
        preprocessor = best_model.named_steps["preprocess"]
        estimator = best_model.named_steps["model"]
        transformed = preprocessor.transform(sample)
        if not isinstance(transformed, np.ndarray):
            transformed = transformed.toarray()
        transformed = pd.DataFrame(transformed, columns=sample.columns, index=sample.index)

        explainer = shap.Explainer(estimator, transformed)
        shap_values = explainer(transformed, check_additivity=False)
        shap_importance = pd.DataFrame(
            {
                "feature": transformed.columns,
                "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        shap_importance.to_csv(ROOT / "reports" / "shap_importance.csv", index=False)

        plt.figure()
        shap.summary_plot(shap_values, transformed, show=False, max_display=12)
        plt.title(f"SHAP Summary - {best_name}")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "16_shap_summary.png", dpi=160, bbox_inches="tight")
        plt.close()
        shap_note_path.write_text(
            "SHAP analysis completed successfully.\n",
            encoding="utf-8",
        )
    except Exception as exc:
        shap_note_path.write_text(
            f"SHAP analysis was skipped because of {exc.__class__.__name__}: {exc}\n",
            encoding="utf-8",
        )


def save_risk_model_report(best_name: str) -> None:
    metrics = pd.read_csv(ROOT / "reports" / "model_metrics.csv")
    thresholds = pd.read_csv(ROOT / "reports" / "threshold_business_decisions.csv")
    decile = pd.read_csv(ROOT / "reports" / "decile_lift_gains.csv")
    iv = pd.read_csv(ROOT / "reports" / "iv_summary.csv")
    calibration = pd.read_csv(ROOT / "reports" / "calibration_table.csv")
    scorecard_comparison_path = ROOT / "reports" / "scorecard_vs_ml_comparison.csv"
    scorecard_features_path = ROOT / "reports" / "scorecard_selected_features.csv"
    scorecard_comparison = (
        pd.read_csv(scorecard_comparison_path) if scorecard_comparison_path.exists() else pd.DataFrame()
    )
    scorecard_features = (
        pd.read_csv(scorecard_features_path) if scorecard_features_path.exists() else pd.DataFrame()
    )
    shap_status = (ROOT / "reports" / "shap_status.txt").read_text(encoding="utf-8").strip()
    shap_path = ROOT / "reports" / "shap_importance.csv"
    shap_importance = pd.read_csv(shap_path) if shap_path.exists() else pd.DataFrame()

    top_decile = decile.iloc[0]
    top_two_deciles = decile.iloc[1]
    top_three_deciles = decile.iloc[2]
    brier = calibration["brier_score"].iloc[0]

    lines = [
        "# Credit Risk Model Report",
        "",
        "## Model Performance",
        "",
        f"Best model: `{best_name}`",
        "",
        metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        "## Decile / Lift / Gains",
        "",
        f"- Top 10% highest-risk borrowers have an observed default rate of {top_decile['default_rate']:.2%}.",
        f"- Top 10% highest-risk borrowers capture {top_decile['cumulative_bad_capture_rate']:.2%} of all observed defaulters.",
        f"- Top 20% highest-risk borrowers capture {top_two_deciles['cumulative_bad_capture_rate']:.2%} of all observed defaulters.",
        f"- Top 30% highest-risk borrowers capture {top_three_deciles['cumulative_bad_capture_rate']:.2%} of all observed defaulters.",
        f"- Top-decile lift is {top_decile['lift']:.2f}x over the portfolio default rate.",
        "",
        "Business use: decile analysis shows whether the model can rank-order risk. This is useful for manual review queues, risk-based pricing, and credit line management.",
        "",
        "## WOE / IV Screening",
        "",
        iv.head(10).to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        "Business use: IV identifies variables with strong discriminatory power. Very high IV values should be reviewed for redundancy or leakage before production use.",
        "",
        "## WOE Logistic Scorecard Benchmark",
        "",
    ]
    if not scorecard_comparison.empty:
        lines.extend(
            [
                "Selected scorecard variables:",
                "",
            ]
        )
        if not scorecard_features.empty:
            lines.extend([f"- {feature}" for feature in scorecard_features["selected_feature"]])
        lines.extend(
            [
                "",
                scorecard_comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
                "",
                "Business use: the WOE logistic scorecard acts as a traditional, transparent benchmark. It is easier to explain to credit policy stakeholders, while the machine learning champion provides stronger flexibility for non-linear risk patterns.",
                "",
            ]
        )
    else:
        lines.extend(["Scorecard benchmark was not generated.", ""])
    lines.extend(
        [
            "## Calibration",
            "",
            f"Brier score: {brier:.4f}",
            "",
            "Business use: calibration checks whether predicted default probabilities are close to observed default rates. This matters when model scores are used directly as expected loss inputs.",
            "",
            "## SHAP Explainability",
            "",
            shap_status,
            "",
        ]
    )
    if not shap_importance.empty:
        lines.extend(
            [
                "Top SHAP drivers:",
                "",
                shap_importance.head(10).to_string(index=False, float_format=lambda value: f"{value:.4f}"),
                "",
            ]
        )
    lines.extend(
        [
            "Business use: SHAP explains which variables push model risk predictions higher or lower. This helps with model governance, stakeholder communication, and adverse-action style reasoning.",
            "",
            "## Approval Threshold Trade-off",
            "",
            thresholds.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
            "",
            "Business use: threshold analysis translates risk scores into approval, rejection, or manual review decisions.",
        ]
    )
    (ROOT / "reports" / "risk_model_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def train() -> None:
    check_required_dependencies()
    import_required_dependencies()
    ensure_dirs()
    df = load_data()
    df = add_features(df)
    df = clip_outliers(df)
    make_eda(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    numeric_features = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ]
    )

    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            (
                                "num",
                                SimpleImputer(strategy="median"),
                                numeric_features,
                            )
                        ]
                    ),
                ),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=120,
                        learning_rate=0.08,
                        max_leaf_nodes=31,
                        l2_regularization=0.01,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            (
                                "num",
                                SimpleImputer(strategy="median"),
                                numeric_features,
                            )
                        ]
                    ),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=80,
                        max_depth=6,
                        min_samples_leaf=100,
                        class_weight="balanced",
                        n_jobs=2,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier

        positive_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        models["xgboost"] = Pipeline(
            steps=[
                (
                    "preprocess",
                    ColumnTransformer(
                        transformers=[
                            (
                                "num",
                                SimpleImputer(strategy="median"),
                                numeric_features,
                            )
                        ]
                    ),
                ),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=120,
                        max_depth=3,
                        learning_rate=0.08,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        eval_metric="logloss",
                        scale_pos_weight=positive_weight,
                        n_jobs=2,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    except Exception as exc:
        print(f"Skipping xgboost: {exc.__class__.__name__}: {exc}")

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        results.append(evaluate_model(name, model, X_test, y_test))

    results_df = pd.DataFrame(
        [{"model": r["model"], "auc": r["auc"], "ks": r["ks"]} for r in results]
    ).sort_values("auc", ascending=False)
    results_df.to_csv(ROOT / "reports" / "model_metrics.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_model = models[best_name]
    save_curves(best_name, best_model, X_test, y_test)
    save_feature_importance(best_name, best_model, numeric_features)
    save_permutation_importance(best_name, best_model, X_test, y_test)
    save_business_decision_report(best_name, best_model, X_test, y_test)
    save_decile_lift_gains(best_model, X_test, y_test)
    save_woe_iv(X_train, y_train)
    save_scorecard_benchmark(X_train, X_test, y_train, y_test, best_name, results_df)
    save_calibration_analysis(best_model, X_test, y_test)
    save_shap_analysis(best_name, best_model, X_test)
    save_risk_model_report(best_name)

    processed = df.copy()
    processed.to_csv(PROCESSED_DIR / "clean_modeling_data.csv", index=False)

    lines = [
        "Credit Default Prediction - Model Summary",
        "=" * 48,
        "",
        f"Rows: {len(df):,}",
        f"Features: {len(numeric_features)}",
        f"Default rate: {y.mean():.4%}",
        "",
        "Model metrics:",
        results_df.to_string(index=False),
        "",
        "Detailed threshold=0.2 reports:",
    ]
    for result in results:
        lines.extend(
            [
                "",
                f"[{result['model']}]",
                f"AUC: {result['auc']:.4f}",
                f"KS: {result['ks']:.4f}",
                f"Confusion matrix: {result['confusion_matrix_threshold_0_2']}",
                result["classification_report_threshold_0_2"],
            ]
        )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(results_df)
    print(f"\nBest model: {best_name}")
    print(f"Saved summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    train()
