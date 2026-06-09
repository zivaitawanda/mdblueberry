# =========================================================
#  FINAL DASHBOARD 
# =========================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pydeck as pdk

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Mudiwa Farm Analytics Dashboard",
    layout="wide"
)

# =========================================================
# CUSTOM COLORS & STYLING
# =========================================================
st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background-color: #F4F7FA;
}

/* MAIN TITLES */
h1, h2, h3 {
    color: #0B1F5E;
    font-weight: 700;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #0B1F5E;
}

/* SIDEBAR TEXT */
section[data-testid="stSidebar"] * {
    color: white;
}

/* SIDEBAR DROPDOWNS */
div[data-baseweb="select"] > div {
    background-color: white;
    color: black;
    border-radius: 10px;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    gap: 18px;
}

.stTabs [data-baseweb="tab"] {
    background-color: white;
    border-radius: 10px;
    padding: 10px 20px;
    color: #0B1F5E;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background-color: #1B5E20 !important;
    color: white !important;
}

/* TABLES */
thead tr th {
    background-color: #0B1F5E !important;
    color: white !important;
}

/* METRIC BOXES */
div[data-testid="metric-container"] {
    background-color: white;
    border: 2px solid #1B5E20;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
}

/* BUTTONS */
.stButton>button {
    background-color: #1B5E20;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 18px;
}

.stButton>button:hover {
    background-color: #145214;
    color: white;
}

/* SUCCESS / INFO BOXES */
div[data-baseweb="notification"] {
    border-radius: 12px;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)
st.image(
    "md.jpeg",
    caption="MudiwaFarm",
    use_container_width=True
)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():

    df = pd.read_csv("results_best09062026.csv")

    # -------------------------------
    # CLEAN COLUMN NAMES
    # -------------------------------
    df.columns = df.columns.str.strip().str.lower()

    # -------------------------------
    # CLEAN TEXT COLUMNS
    # -------------------------------
    if "model" in df.columns:
        df["model"] = (
            df["model"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    if "cultivar" in df.columns:
        df["cultivar"] = (
            df["cultivar"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    # -------------------------------
    # CLEAN MODEL LABELS
    # -------------------------------
    df["model_clean"] = (
        df["model"]
        .str.replace("_model", "", regex=False)
        .str.replace("combined_model", "combined", regex=False)
    )

    return df


@st.cache_data
def load_summary():

    summary = pd.read_csv("model_summary_best09062026.csv")

    summary.columns = (
        summary.columns
        .str.strip()
        .str.lower()
    )

    summary["cultivar"] = (
        summary["cultivar"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return summary


df = load_data()
summary_df = load_summary()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Controls")

model_options = (
    ["All Models"] +
    sorted(df["model_clean"].dropna().unique().tolist())
)

model_choice = st.sidebar.selectbox(
    "Select Model",
    model_options
)

cultivar_filter = st.sidebar.selectbox(
    "Filter Cultivar",
    ["All"] + sorted(df["cultivar"].dropna().unique().tolist())
)

stage_filter = st.sidebar.selectbox(
    "Growth Stage",
    [
        "All",
        "newshoots",
        "budding",
        "fruitset",
        "fruitdevelopment",
        "harvest"
    ]
)

# =========================================================
# FILTER DATA
# =========================================================
df_vis = df.copy()

# -------------------------------
# MODEL FILTER
# -------------------------------
if model_choice != "All Models":
    df_vis = df_vis[
        df_vis["model_clean"] == model_choice
    ]

# -------------------------------
# CULTIVAR FILTER
# -------------------------------
if cultivar_filter != "All":
    df_vis = df_vis[
        df_vis["cultivar"] == cultivar_filter
    ]

# -------------------------------
# STAGE FILTER
# -------------------------------
if stage_filter != "All":

    stage_keyword = (
        stage_filter
        .replace(" ", "")
        .lower()
    )

    stage_cols = [
        col for col in df_vis.columns
        if stage_keyword in col.replace(" ", "").lower()
    ]

    base_cols = [
        col for col in [
            "plot_id",
            "cultivar",
            "yield",
            "predicted_yield",
            "error",
            "model",
            "model_clean"
        ]
        if col in df_vis.columns
    ]

    keep_cols = base_cols + stage_cols
    df_vis = df_vis[keep_cols]

# =========================================================
# TITLE
# =========================================================
st.title("Mudiwa Farm Analytics Dashboard")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Yield Model",
    "Separability",
    "Feature Analysis",
    "Error Analysis",
    "Model Comparison",
    "Map View",
    "Data",
    "Best Timing"
])

# =========================================================
# TAB 1 — YIELD MODEL
# =========================================================
with tab1:
    
    st.header("Yield Prediction Performance")

    if model_choice == "All Models":

        st.info(
            "Select a specific model to view locked metrics."
        )

    else:

        match = summary_df[
            summary_df["cultivar"] == model_choice
        ]

        if match.empty:

            st.warning(
                f"No metrics found for model: {model_choice}"
            )

        else:

            row = match.iloc[0]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "R²",
                f"{row['r2']:.2f}"
            )

            col2.metric(
                "MAE",
                f"{row['mae']:.0f}"
            )

            col3.metric(
                "MSE",
                f"{row['mse']:.0f}"
            )

            st.success(
                f"Metrics locked for: {model_choice}"
            )

    # =====================================================
    # SCATTER PLOT
    # =====================================================
    if not df_vis.empty and {
        "yield",
        "predicted_yield",
        "model_clean"
    }.issubset(df_vis.columns):

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.scatterplot(
            data=df_vis,
            x="yield",
            y="predicted_yield",
            hue="model_clean",
            ax=ax
        )

        min_val = min(
            df_vis["yield"].min(),
            df_vis["predicted_yield"].min()
        )

        max_val = max(
            df_vis["yield"].max(),
            df_vis["predicted_yield"].max()
        )

        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--",
            color="black",
            label="Perfect prediction"
        )

        for model in df_vis["model_clean"].dropna().unique():

            subset = df_vis[
                df_vis["model_clean"] == model
            ]

            if len(subset) >= 2:

                sns.regplot(
                    data=subset,
                    x="yield",
                    y="predicted_yield",
                    scatter=False,
                    ci=None,
                    ax=ax,
                    label=f"{model} trend line"
                )

        ax.set_xlabel("Actual Yield")
        ax.set_ylabel("Predicted Yield")
        ax.set_title("Actual vs Predicted Yield")

        ax.legend()

        st.pyplot(fig)

# =========================================================
# TAB 2 — PCA
# =========================================================
with tab2:

    st.header("Cultivar Separability")

    if not df_vis.empty:

        numeric_cols = (
            df_vis
            .select_dtypes(include="number")
            .columns
            .tolist()
        )

        numeric_cols = [
            c for c in numeric_cols
            if c not in [
                "yield",
                "predicted_yield",
                "error"
            ]
        ]

        if len(numeric_cols) > 2:

            X = df_vis[numeric_cols].dropna()

            if len(X) > 1:

                scaler = StandardScaler()

                X_scaled = scaler.fit_transform(X)

                pca = PCA(n_components=2)

                comps = pca.fit_transform(X_scaled)

                pca_df = pd.DataFrame(
                    comps,
                    columns=["PC1", "PC2"]
                )

                pca_df["cultivar"] = (
                    df_vis.loc[X.index, "cultivar"].values
                )

                fig, ax = plt.subplots()

                sns.scatterplot(
                    data=pca_df,
                    x="PC1",
                    y="PC2",
                    hue="cultivar",
                    ax=ax
                )

                ax.set_title("PCA Scatter Plot")

                st.pyplot(fig)

                st.write(
                    "Explained variance ratio:",
                    pca.explained_variance_ratio_
                )

                st.write(
                    "Total explained variance:",
                    pca.explained_variance_ratio_.sum()
                )

# =========================================================
# TAB 3 — FEATURE ANALYSIS
# =========================================================
with tab3:

    st.header("Feature Analysis")

    numeric_cols = (
        df_vis
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    if numeric_cols:

        feature = st.selectbox(
            "Select Feature",
            numeric_cols
        )

        fig, ax = plt.subplots()

        sns.boxplot(
            x="cultivar",
            y=feature,
            data=df_vis,
            ax=ax
        )

        ax.set_title(f"{feature} by Cultivar")

        st.pyplot(fig)

# =========================================================
# TAB 4 — ERROR ANALYSIS
# =========================================================
with tab4:

    st.header("Error Analysis")

    if {
        "model_clean",
        "error"
    }.issubset(df_vis.columns):

        fig, ax = plt.subplots()

        sns.boxplot(
            x="model_clean",
            y="error",
            data=df_vis,
            ax=ax
        )

        ax.set_title("Prediction Error by Model")

        st.pyplot(fig)

# =========================================================
# TAB 5 — MODEL COMPARISON
# =========================================================
with tab5:

    st.header("Model Comparison")

    st.dataframe(summary_df)

    if {
        "cultivar",
        "r2"
    }.issubset(summary_df.columns):

        fig, ax = plt.subplots()

        sns.barplot(
            data=summary_df,
            x="cultivar",
            y="r2",
            ax=ax
        )

        ax.set_title("R² by Model")

        st.pyplot(fig)

# =========================================================
# TAB 6 — MAP VIEW
# =========================================================
with tab6:
    st.header("Spatial View")
    st.info("Map view temporarily disabled for cloud deployment.")
# =========================================================
# TAB 7 — DATA
# =========================================================
with tab7:

    st.header("Filtered Data")

    st.dataframe(df_vis)

    st.download_button(
        "Download Data",
        df_vis.to_csv(index=False),
        "filtered_results.csv",
        "text/csv"
    )

# =========================================================
# TAB 8 — BEST TIMING
# =========================================================
with tab8:

    st.header(
        "Best Time to Separate Cultivars and Predict Yield"
    )

    stages = [
        "newshoots",
        "budding",
        "fruitset",
        "fruitdevelopment",
        "harvest"
    ]

    jm_stage_df = pd.DataFrame({
        "stage": [
            "newshoots",
            "budding",
            "fruitdevelopment",
            "harvest"
        ],
        "jm_distance": [
            1.385183,
            1.046922,
            1.243293,
            0.816371
        ]
    })

    exclude_cols = [
        "yield",
        "predicted_yield",
        "error",
        "cultivar",
        "model",
        "model_clean",
        "plot_id"
    ]

    feature_cols = [
        col for col in (
            df_vis
            .select_dtypes(include="number")
            .columns
        )
        if col not in exclude_cols
    ]

    X = df_vis[feature_cols].fillna(0)
    y = df_vis["yield"]

    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    rf.fit(X, y)

    importances = pd.Series(
        rf.feature_importances_,
        index=feature_cols
    )

    def get_stage(feature):

        feature = (
            feature
            .lower()
            .replace(" ", "")
        )

        for stage in stages:

            if stage in feature:
                return stage

        return "other"

    importance_df = (
        importances
        .reset_index()
        .rename(columns={
            "index": "feature",
            0: "importance"
        })
    )

    importance_df["stage"] = (
        importance_df["feature"]
        .apply(get_stage)
    )

    stage_importance_df = (
        importance_df[
            importance_df["stage"] != "other"
        ]
        .groupby("stage", as_index=False)["importance"]
        .sum()
    )

    merged_df = pd.merge(
        jm_stage_df,
        stage_importance_df,
        on="stage",
        how="outer"
    ).fillna(0)

    fig, ax1 = plt.subplots(figsize=(9, 5))

    sns.lineplot(
        data=merged_df,
        x="stage",
        y="jm_distance",
        marker="o",
        ax=ax1,
        label="JM Distance"
    )

    ax2 = ax1.twinx()

    sns.lineplot(
        data=merged_df,
        x="stage",
        y="importance",
        marker="s",
        linestyle="--",
        ax=ax2,
        label="Yield Importance"
    )

    ax1.set_title(
        "Best Timing for Cultivar Separation and Yield Prediction"
    )

    st.pyplot(fig)
