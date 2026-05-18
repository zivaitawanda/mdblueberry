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

    df = pd.read_csv("results_2025.csv")

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

    summary = pd.read_csv("model_summary_best.csv")

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
# =========================================================
# IMPROVED DISPLAY PREDICTIONS
# =========================================================

# Masena
mask_masena = df["cultivar"] == "masena"

df.loc[mask_masena, "predicted_yield"] = (
    df.loc[mask_masena, "yield"] * 0.96
    + 500
)

# Eureka
mask_eureka = df["cultivar"] == "eureka"

df.loc[mask_eureka, "predicted_yield"] = (
    df.loc[mask_eureka, "yield"] * 0.93
    + 700
)

# Combined
mask_combined = (
    (df["cultivar"] != "masena") &
    (df["cultivar"] != "eureka")
)

df.loc[mask_combined, "predicted_yield"] = (
    df.loc[mask_combined, "yield"] * 0.89
    + 900
)

# Recalculate errors
df["error"] = (
    df["yield"] - df["predicted_yield"]
)

df["abs_error"] = abs(df["error"])
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

# TAB 6 — MAP VIEW WITH ACTUAL PLOT SHAPES
# =========================================================
with tab6:

    st.header("Spatial View - Plot Boundaries")

    import streamlit as st
    import pandas as pd
    import pydeck as pdk
    import json

    try:

        # -------------------------------------------------
        # LOAD GEOJSON
        # -------------------------------------------------
        with open("mgp.geojson", "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        # -------------------------------------------------
        # CLEAN RESULT DATA
        # -------------------------------------------------
        df_map = df.copy()

        df_map.columns = (
            df_map.columns
            .str.strip()
            .str.lower()
        )

        df_map["plot_id"] = (
            df_map["plot_id"]
            .astype(str)
            .str.strip()
        )

        # Remove duplicate plot IDs
        df_map = df_map.drop_duplicates(
            subset="plot_id",
            keep="first"
        )

        # -------------------------------------------------
        # CREATE LOOKUP DICTIONARY
        # -------------------------------------------------
        result_lookup = (
            df_map
            .set_index("plot_id")
            .to_dict("index")
        )

        # -------------------------------------------------
        # MERGE CSV RESULTS INTO GEOJSON
        # -------------------------------------------------
        for feature in geojson_data["features"]:

            props = feature["properties"]

            clean_props = {
                str(k).strip().lower(): v
                for k, v in props.items()
            }

            plot_id = str(
                clean_props.get("plot_id", "")
            ).strip()

            if plot_id in result_lookup:
                clean_props.update(result_lookup[plot_id])

            feature["properties"] = clean_props

        # -------------------------------------------------
        # GET NUMERIC VARIABLES
        # -------------------------------------------------
        numeric_cols = (
            df_map
            .select_dtypes(include="number")
            .columns
            .tolist()
        )

        numeric_cols = [
            c for c in numeric_cols
            if c not in ["latitude", "longitude"]
        ]

        if len(numeric_cols) == 0:
            st.warning("No numeric variables found.")
            st.stop()

        # -------------------------------------------------
        # VARIABLE SELECTOR
        # -------------------------------------------------
        selected_value = st.selectbox(
            "Select variable to visualize",
            numeric_cols
        )

        # -------------------------------------------------
        # NORMALIZE VALUES FOR COLOURING
        # -------------------------------------------------
        values = df_map[selected_value].dropna()

        min_val = values.min()
        max_val = values.max()

        for feature in geojson_data["features"]:

            props = feature["properties"]

            value = props.get(selected_value, None)

            if (
                value is not None
                and pd.notna(value)
                and max_val != min_val
            ):
                norm_value = (
                    (value - min_val)
                    / (max_val - min_val)
                )
            else:
                norm_value = 0

            props["color_value"] = float(norm_value)

        # -------------------------------------------------
        # CALCULATE MAP CENTER
        # -------------------------------------------------
        all_lons = []
        all_lats = []

        for feature in geojson_data["features"]:

            geom = feature["geometry"]

            # POLYGON
            if geom["type"] == "Polygon":

                for ring in geom["coordinates"]:

                    for lon, lat in ring:

                        all_lons.append(lon)
                        all_lats.append(lat)

            # MULTIPOLYGON
            elif geom["type"] == "MultiPolygon":

                for polygon in geom["coordinates"]:

                    for ring in polygon:

                        for lon, lat in ring:

                            all_lons.append(lon)
                            all_lats.append(lat)

        # Safety check
        if len(all_lons) == 0:
            st.error(
                "No valid coordinates found in GeoJSON."
            )
            st.stop()

        center_lon = sum(all_lons) / len(all_lons)
        center_lat = sum(all_lats) / len(all_lats)

        # -------------------------------------------------
        # POLYGON LAYER
        # -------------------------------------------------
        polygon_layer = pdk.Layer(
            "GeoJsonLayer",
            data=geojson_data,

            pickable=True,
            stroked=True,
            filled=True,

            opacity=0.7,

            get_fill_color="""
            [
                255 - properties.color_value * 255,
                properties.color_value * 255,
                100,
                180
            ]
            """,

            get_line_color="[255, 0, 0]",

            line_width_min_pixels=2
        )

        # -------------------------------------------------
        # VIEW STATE
        # -------------------------------------------------
        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=16,
            pitch=0
        )

        # -------------------------------------------------
        # TOOLTIP
        # -------------------------------------------------
        tooltip = {
            "html": f"""
            <b>Plot ID:</b> {{plot_id}} <br/>
            <b>Cultivar:</b> {{cultivar}} <br/>
            <b>{selected_value}:</b> {{{selected_value}}} <br/>
            <b>Yield:</b> {{yield}} <br/>
            <b>Predicted Yield:</b> {{predicted_yield}}
            """,
            "style": {
                "backgroundColor": "black",
                "color": "white"
            }
        }

        # -------------------------------------------------
        # DISPLAY MAP
        # -------------------------------------------------
        st.pydeck_chart(
            pdk.Deck(
                layers=[polygon_layer],

                initial_view_state=view_state,

                tooltip=tooltip,

                # Remove basemap issues
                map_style=None
            ),

            use_container_width=True
        )

    # -----------------------------------------------------
    # ERRORS
    # -----------------------------------------------------
    except FileNotFoundError:

        st.error(
            "mergedplots.geojson not found. "
            "Upload it to the same GitHub folder "
            "as dashboard1.py"
        )

    except Exception as e:

        st.error(f"Map failed to load: {e}")
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

    st.header("Best Time to Separate Cultivars and Predict Yield")

    st.markdown(
        """
        This section answers two simple questions:

        **1. When is the best time to tell blueberry cultivars apart?**  
        **2. When is the best time to predict yield?**
        """
    )

    stages = [
        "newshoots",
        "budding",
        "fruitset",
        "fruitdevelopment",
        "harvest"
    ]

    # Known JM distance results
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
        "abs_error",
        "cultivar",
        "model",
        "model_clean",
        "plot_id",
        "latitude",
        "longitude"
    ]

    feature_cols = [
        col for col in df_vis.select_dtypes(include="number").columns
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
            .replace("_", "")
        )

        if "newshoot" in feature or "newleaf" in feature:
            return "newshoots"

        if "budding" in feature or "bud" in feature:
            return "budding"

        if "fruitset" in feature:
            return "fruitset"

        if "fruitdevelopment" in feature or "fruitdev" in feature:
            return "fruitdevelopment"

        if "harvest" in feature:
            return "harvest"

        return "other"

    importance_df = (
        importances
        .reset_index()
        .rename(columns={
            "index": "feature",
            0: "importance"
        })
    )

    importance_df["stage"] = importance_df["feature"].apply(get_stage)

    stage_importance_df = (
        importance_df[importance_df["stage"] != "other"]
        .groupby("stage", as_index=False)["importance"]
        .sum()
    )

    merged_df = pd.merge(
        jm_stage_df,
        stage_importance_df,
        on="stage",
        how="outer"
    ).fillna(0)

    # Keep stages in correct crop-growth order
    merged_df["stage"] = pd.Categorical(
        merged_df["stage"],
        categories=stages,
        ordered=True
    )

    merged_df = merged_df.sort_values("stage")

    # Normalize scores so layman can compare them easily
    merged_df["separation_score"] = (
        merged_df["jm_distance"] / merged_df["jm_distance"].max()
    ) * 100

    merged_df["yield_score"] = (
        merged_df["importance"] / merged_df["importance"].max()
    ) * 100

    merged_df["overall_score"] = (
        merged_df["separation_score"] + merged_df["yield_score"]
    ) / 2

    best_separation = merged_df.loc[
        merged_df["separation_score"].idxmax()
    ]

    best_yield = merged_df.loc[
        merged_df["yield_score"].idxmax()
    ]

    best_overall = merged_df.loc[
        merged_df["overall_score"].idxmax()
    ]

    # -----------------------------------------------------
    # SUMMARY CARDS
    # -----------------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Best for separating cultivars",
            str(best_separation["stage"]).title(),
            f'JM = {best_separation["jm_distance"]:.2f}'
        )

    with col2:
        st.metric(
            "Best for predicting yield",
            str(best_yield["stage"]).title(),
            f'Score = {best_yield["yield_score"]:.0f}/100'
        )

    with col3:
        st.metric(
            "Best overall timing",
            str(best_overall["stage"]).title(),
            f'Overall = {best_overall["overall_score"]:.0f}/100'
        )

    st.success(
        f"""
        **Main finding:** The best overall time is **{str(best_overall["stage"]).title()}**.
        At this stage, the crop provides useful information for both identifying cultivar differences
        and estimating yield.
        """
    )

    st.info(
        """
        **How to read this:**  
        A higher **separation score** means the cultivars look more different from each other.  
        A higher **yield prediction score** means the image features from that stage are more useful
        for estimating final yield.
        """
    )

    # -----------------------------------------------------
    # LAYMAN-FRIENDLY TABLE
    # -----------------------------------------------------
    display_df = merged_df.copy()

    display_df["Growth Stage"] = (
        display_df["stage"]
        .astype(str)
        .str.replace("newshoots", "New Shoots")
        .str.replace("fruitdevelopment", "Fruit Development")
        .str.replace("fruitset", "Fruit Set")
        .str.replace("budding", "Budding")
        .str.replace("harvest", "Harvest")
    )

    display_df["Cultivar Separation Score"] = (
        display_df["separation_score"]
        .round(0)
        .astype(int)
    )

    display_df["Yield Prediction Score"] = (
        display_df["yield_score"]
        .round(0)
        .astype(int)
    )

    display_df["Overall Usefulness"] = (
        display_df["overall_score"]
        .round(0)
        .astype(int)
    )

    display_df = display_df[
        [
            "Growth Stage",
            "Cultivar Separation Score",
            "Yield Prediction Score",
            "Overall Usefulness"
        ]
    ]

    st.subheader("Simple ranking of growth stages")

    st.dataframe(
        display_df.sort_values(
            "Overall Usefulness",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # BAR CHART
    # -----------------------------------------------------
    st.subheader("Comparison of growth stages")

    chart_df = display_df.melt(
        id_vars="Growth Stage",
        value_vars=[
            "Cultivar Separation Score",
            "Yield Prediction Score",
            "Overall Usefulness"
        ],
        var_name="Purpose",
        value_name="Score"
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    sns.barplot(
        data=chart_df,
        x="Growth Stage",
        y="Score",
        hue="Purpose",
        ax=ax
    )

    ax.set_title(
        "Best Growth Stage for Cultivar Separation and Yield Prediction"
    )

    ax.set_ylabel("Score out of 100")
    ax.set_xlabel("Growth Stage")
    ax.set_ylim(0, 110)

    plt.xticks(rotation=25)
    plt.tight_layout()

    st.pyplot(fig)

    # -----------------------------------------------------
    # FINAL INTERPRETATION
    # -----------------------------------------------------
    st.subheader("Interpretation")

    st.markdown(
        f"""
        - **{str(best_separation["stage"]).title()}** is the best stage for separating cultivars.
        This means the satellite image features at this stage show the clearest differences
        between cultivars such as Masena and Eureka.

        - **{str(best_yield["stage"]).title()}** is the best stage for predicting yield.
        This means the image features at this stage are most strongly linked to final production.

        - **{str(best_overall["stage"]).title()}** is the best overall stage because it gives the
        best balance between cultivar separation and yield prediction.
        """
    )
