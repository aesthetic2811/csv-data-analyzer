import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DataScope AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# 2. CUSTOM CSS — Dark Glassmorphism UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;600;800&display=swap');

/* ── BASE ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #050811 !important;
    font-family: 'Outfit', sans-serif;
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(99,102,241,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(236,72,153,0.14) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 50% 50%, rgba(6,182,212,0.07) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: rgba(10,12,28,0.95) !important;
    border-right: 1px solid rgba(99,102,241,0.25) !important;
}
[data-testid="stSidebar"] * { color: #c7d2fe !important; }
[data-testid="stSidebar"] .stFileUploader label { color: #a5b4fc !important; }

/* ── MAIN TITLE AREA ── */
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 40%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    line-height: 1.1;
}
.hero-sub {
    color: #64748b;
    font-size: 1.05rem;
    margin-top: 0.4rem;
    font-weight: 300;
    letter-spacing: 0.03em;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15,17,40,0.8) !important;
    border-radius: 14px !important;
    padding: 6px !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 8px 18px !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #c084fc) !important;
    color: white !important;
}

/* ── METRIC CARDS ── */
.metric-card {
    background: rgba(15,17,40,0.85);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(192,132,252,0.5);
}
.metric-card .val {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-card .lbl {
    color: #64748b;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 4px;
}

/* ── SECTION HEADERS ── */
.section-header {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    color: #e2e8f0;
    border-left: 3px solid #6366f1;
    padding-left: 12px;
    margin: 1.5rem 0 1rem;
}

/* ── ML RESULT BOX ── */
.ml-result-box {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 14px;
    padding: 1.4rem;
    margin: 1rem 0;
}
.ml-result-box code {
    background: rgba(0,0,0,0.3);
    color: #a5b4fc;
    padding: 2px 7px;
    border-radius: 5px;
    font-family: 'Space Mono', monospace;
}

/* ── EXPLANATION BOX ── */
.explain-box {
    background: rgba(6,182,212,0.07);
    border: 1px solid rgba(6,182,212,0.25);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    color: #94a3b8;
    font-size: 0.92rem;
    line-height: 1.7;
    margin-top: 1rem;
}
.explain-box strong { color: #67e8f9; }
.explain-box .step-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    display: block;
    margin-bottom: 6px;
}

/* ── GENERAL TEXT ── */
p, li, label, .stMarkdown { color: #94a3b8 !important; }
h1, h2, h3 { color: #e2e8f0 !important; }
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #c084fc) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.5rem !important;
    transition: opacity 0.2s, transform 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

/* ── SELECT/INPUT ── */
.stSelectbox > div > div, .stSlider > div {
    background: rgba(15,17,40,0.9) !important;
    border-color: rgba(99,102,241,0.3) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}

/* ── INFO / WARNING ── */
.stInfo { background: rgba(99,102,241,0.12) !important; border-color: #6366f1 !important; color: #a5b4fc !important; }

/* ── DIVIDER ── */
hr { border-color: rgba(99,102,241,0.2) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 DataScope AI")
    st.markdown('<p style="color:#4b5563;font-size:0.82rem;margin-top:-8px;">ML-Powered CSV Explorer</p>', unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.file_uploader("📂 Upload CSV File", type="csv")

    st.markdown("---")
    st.markdown('<p style="color:#4b5563;font-size:0.78rem;">Built by <strong style="color:#818cf8">Janvi Sharma</strong><br>Streamlit · Plotly · scikit-learn</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 4. HERO HEADER
# ─────────────────────────────────────────────
st.markdown('<h1 class="hero-title">DataScope AI 🧠</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload any CSV → Explore data · Run statistics · Train ML models instantly</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 5. MAIN LOGIC
# ─────────────────────────────────────────────
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    all_cols = df.columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # ── TOP METRIC CARDS ──
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (df.shape[0], "Total Rows"),
        (df.shape[1], "Total Columns"),
        (len(numeric_cols), "Numeric Cols"),
        (int(df.isnull().sum().sum()), "Missing Values"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ──
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📄 Dataset", "📊 Statistics", "📈 Visualize",
        "🔮 Regression", "🎯 Clustering", "🌲 Classifier"
    ])

    # ══════════════════════════════════════════
    # TAB 1 — DATASET OVERVIEW
    # ══════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-header">Raw Data Preview (Top 10)</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        with st.expander("📋 All Column Names"):
            st.write(list(df.columns))

    # ══════════════════════════════════════════
    # TAB 2 — STATISTICS
    # ══════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
        st.dataframe(df.describe(), use_container_width=True)
        st.markdown('<div class="section-header">Missing Values per Column</div>', unsafe_allow_html=True)
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Count"]
        st.dataframe(missing, use_container_width=True)

    # ══════════════════════════════════════════
    # TAB 3 — VISUALIZATIONS
    # ══════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-header">Custom Visualization</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: x_axis = st.selectbox("X-Axis", all_cols)
        with c2: y_axis = st.selectbox("Y-Axis", numeric_cols)
        with c3: plot_type = st.selectbox("Chart Type", ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot"])

        if st.button("Generate Chart"):
            color_seq = px.colors.sequential.Plasma
            if plot_type == "Bar Chart":
                fig = px.bar(df, x=x_axis, y=y_axis, color=x_axis, color_discrete_sequence=px.colors.qualitative.Vivid)
            elif plot_type == "Line Chart":
                fig = px.line(df, x=x_axis, y=y_axis, color_discrete_sequence=["#818cf8"])
            elif plot_type == "Scatter Plot":
                fig = px.scatter(df, x=x_axis, y=y_axis, color=x_axis, color_discrete_sequence=px.colors.qualitative.Pastel)
            elif plot_type == "Histogram":
                fig = px.histogram(df, x=x_axis, color_discrete_sequence=["#c084fc"])
            elif plot_type == "Box Plot":
                fig = px.box(df, y=y_axis, color_discrete_sequence=["#f472b6"])

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(10,12,28,0.6)',
                font_color='#94a3b8',
                title_font_color='#e2e8f0',
                xaxis=dict(gridcolor='rgba(99,102,241,0.15)', color='#64748b'),
                yaxis=dict(gridcolor='rgba(99,102,241,0.15)', color='#64748b'),
                legend=dict(bgcolor='rgba(0,0,0,0)')
            )
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════
    # TAB 4 — LINEAR REGRESSION (ML MODEL 1)
    # ══════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-header">🔮 ML Model 1 — Linear Regression</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="explain-box">
        <span class="step-label">What is Linear Regression?</span>
        <strong>Linear Regression</strong> ek supervised ML algorithm hai jo ek numeric column (target/Y) 
        ko doosre numeric columns (features/X) ke basis par <strong>predict</strong> karta hai. 
        Yeh X aur Y ke beech ek <strong>straight line</strong> fit karta hai — jaise trend line in Excel.<br><br>
        <strong>Formula:</strong> Y = m₁X₁ + m₂X₂ + ... + b (m = slope/coefficient, b = intercept)
        </div>
        """, unsafe_allow_html=True)

        if len(numeric_cols) < 2:
            st.warning("Regression ke liye kam se kam 2 numeric columns chahiye!")
        else:
            c1, c2 = st.columns(2)
            with c1:
                target_col = st.selectbox("🎯 Target Column (Y — jo predict karna hai)", numeric_cols, key="reg_target")
            with c2:
                feature_options = [c for c in numeric_cols if c != target_col]
                feature_cols = st.multiselect("📥 Feature Columns (X — input doge)", feature_options, default=feature_options[:min(2, len(feature_options))], key="reg_feat")

            test_size = st.slider("Test Split (%)", 10, 40, 20, key="reg_split")

            if st.button("Train Linear Regression Model", key="btn_reg") and feature_cols:
                data_clean = df[feature_cols + [target_col]].dropna()
                X = data_clean[feature_cols].values
                y = data_clean[target_col].values

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                # Show metrics
                mc1, mc2, mc3 = st.columns(3)
                with mc1: st.markdown(f'<div class="metric-card"><div class="val">{r2:.3f}</div><div class="lbl">R² Score</div></div>', unsafe_allow_html=True)
                with mc2: st.markdown(f'<div class="metric-card"><div class="val">{rmse:.2f}</div><div class="lbl">RMSE</div></div>', unsafe_allow_html=True)
                with mc3: st.markdown(f'<div class="metric-card"><div class="val">{len(X_train)}</div><div class="lbl">Training Samples</div></div>', unsafe_allow_html=True)

                # Actual vs Predicted Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers',
                    marker=dict(color='#818cf8', size=7, opacity=0.75),
                    name='Predicted vs Actual'))
                fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                    mode='lines', line=dict(color='#f472b6', dash='dash', width=2), name='Perfect Prediction'))
                fig.update_layout(
                    title="Actual vs Predicted Values",
                    xaxis_title="Actual", yaxis_title="Predicted",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,12,28,0.6)',
                    font_color='#94a3b8', title_font_color='#e2e8f0',
                    xaxis=dict(gridcolor='rgba(99,102,241,0.15)'),
                    yaxis=dict(gridcolor='rgba(99,102,241,0.15)'),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Coefficients
                coef_df = pd.DataFrame({"Feature": feature_cols, "Coefficient": model.coef_})
                st.markdown('<div class="section-header">Model Coefficients</div>', unsafe_allow_html=True)
                st.dataframe(coef_df, use_container_width=True)

                # Teacher explanation
                st.markdown(f"""
                <div class="explain-box">
                <span class="step-label">Teacher ko kaise explain karein</span>
                Maine <strong>LinearRegression</strong> class use ki hai <code>sklearn.linear_model</code> se.<br>
                Data ko <strong>{100-test_size}% training</strong> aur <strong>{test_size}% testing</strong> mein split kiya using <code>train_test_split()</code>.<br>
                Model ne features <code>{', '.join(feature_cols)}</code> se target <code>{target_col}</code> predict karna seekha.<br>
                <strong>R² = {r2:.3f}</strong> matlab model {r2*100:.1f}% variance explain kar sakta hai data ka.<br>
                <strong>RMSE = {rmse:.2f}</strong> matlab average prediction error approximately {rmse:.2f} units hai.
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # TAB 5 — K-MEANS CLUSTERING (ML MODEL 2)
    # ══════════════════════════════════════════
    with tab5:
        st.markdown('<div class="section-header">🎯 ML Model 2 — K-Means Clustering</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="explain-box">
        <span class="step-label">What is K-Means Clustering?</span>
        <strong>K-Means</strong> ek <strong>unsupervised</strong> ML algorithm hai — matlab isko koi label/answer 
        nahi dete, yeh khud data mein <strong>hidden groups (clusters)</strong> dhundh leta hai.<br><br>
        Jaise agar customers ka data do — yeh automatically <em>"budget customers"</em>, <em>"premium customers"</em> 
        type groups bana dega bina bataye!<br>
        <strong>K</strong> = number of clusters jo tum decide karte ho.
        </div>
        """, unsafe_allow_html=True)

        if len(numeric_cols) < 2:
            st.warning("Clustering ke liye kam se kam 2 numeric columns chahiye!")
        else:
            c1, c2 = st.columns(2)
            with c1: x_feat = st.selectbox("X Feature", numeric_cols, key="km_x")
            with c2: y_feat = st.selectbox("Y Feature", numeric_cols, index=min(1, len(numeric_cols)-1), key="km_y")
            k = st.slider("Number of Clusters (K)", 2, 8, 3, key="km_k")

            if st.button("Run K-Means Clustering", key="btn_km"):
                data_clean = df[[x_feat, y_feat]].dropna()
                X = data_clean.values

                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                data_clean = data_clean.copy()
                data_clean["Cluster"] = labels.astype(str)

                centers = kmeans.cluster_centers_

                fig = px.scatter(data_clean, x=x_feat, y=y_feat, color="Cluster",
                    title=f"K-Means Clustering (K={k})",
                    color_discrete_sequence=px.colors.qualitative.Vivid)

                fig.add_trace(go.Scatter(
                    x=centers[:, 0], y=centers[:, 1],
                    mode='markers',
                    marker=dict(symbol='x', size=16, color='white', line=dict(width=2)),
                    name='Centroids'
                ))

                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,12,28,0.6)',
                    font_color='#94a3b8', title_font_color='#e2e8f0',
                    xaxis=dict(gridcolor='rgba(99,102,241,0.15)'),
                    yaxis=dict(gridcolor='rgba(99,102,241,0.15)'),
                )
                st.plotly_chart(fig, use_container_width=True)

                mc1, mc2 = st.columns(2)
                with mc1: st.markdown(f'<div class="metric-card"><div class="val">{k}</div><div class="lbl">Clusters Found</div></div>', unsafe_allow_html=True)
                with mc2: st.markdown(f'<div class="metric-card"><div class="val">{len(data_clean)}</div><div class="lbl">Points Clustered</div></div>', unsafe_allow_html=True)

                st.markdown(f"""
                <div class="explain-box">
                <span class="step-label">Teacher ko kaise explain karein</span>
                Maine <strong>KMeans</strong> class use ki hai <code>sklearn.cluster</code> se.<br>
                K={k} set kiya — matlab maine model ko bola ki <strong>{k} groups</strong> banao.<br>
                Algorithm ne <code>fit_predict()</code> se har data point ko ek cluster number (0 to {k-1}) assign kiya.<br>
                White ✕ marks <strong>centroids</strong> hain — har cluster ka central point.<br>
                Yeh <strong>unsupervised learning</strong> hai — no labels, model khud pattern dhundha!
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # TAB 6 — DECISION TREE (ML MODEL 3)
    # ══════════════════════════════════════════
    with tab6:
        st.markdown('<div class="section-header">🌲 ML Model 3 — Decision Tree Classifier</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="explain-box">
        <span class="step-label">What is Decision Tree?</span>
        <strong>Decision Tree</strong> ek <strong>supervised classification</strong> algorithm hai. 
        Yeh data se <em>if-else rules</em> seekhta hai — bilkul aise jaise ek doctor diagnose karta hai.<br><br>
        Jaise: <em>"Agar salary &gt; 50k AND experience &gt; 3 years → Senior Employee"</em><br>
        Model khud yeh rules data se seekh leta hai! Target column categorical hona chahiye (like Yes/No, Category names).
        </div>
        """, unsafe_allow_html=True)

        target_options = categorical_cols + numeric_cols
        if not target_options:
            st.warning("Classifier ke liye columns nahi mile.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                dt_target = st.selectbox("🎯 Target (Class) Column", target_options, key="dt_target")
            with c2:
                dt_features = st.multiselect("📥 Feature Columns (X)", [c for c in numeric_cols if c != dt_target],
                    default=[c for c in numeric_cols if c != dt_target][:min(3, len(numeric_cols))], key="dt_feat")

            max_depth = st.slider("Max Tree Depth", 2, 8, 3, key="dt_depth")

            if st.button("Train Decision Tree", key="btn_dt") and dt_features:
                data_clean = df[dt_features + [dt_target]].dropna()
                X = data_clean[dt_features].values

                # Encode target
                le = LabelEncoder()
                y = le.fit_transform(data_clean[dt_target].astype(str))

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                acc = accuracy_score(y_test, y_pred)

                mc1, mc2, mc3 = st.columns(3)
                with mc1: st.markdown(f'<div class="metric-card"><div class="val">{acc*100:.1f}%</div><div class="lbl">Accuracy</div></div>', unsafe_allow_html=True)
                with mc2: st.markdown(f'<div class="metric-card"><div class="val">{max_depth}</div><div class="lbl">Tree Depth</div></div>', unsafe_allow_html=True)
                with mc3: st.markdown(f'<div class="metric-card"><div class="val">{len(le.classes_)}</div><div class="lbl">Classes</div></div>', unsafe_allow_html=True)

                # Feature Importance Plot
                importances = clf.feature_importances_
                fi_df = pd.DataFrame({"Feature": dt_features, "Importance": importances}).sort_values("Importance", ascending=True)
                fig = px.bar(fi_df, x="Importance", y="Feature", orientation='h',
                    title="Feature Importances",
                    color="Importance", color_continuous_scale="Plasma")
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(10,12,28,0.6)',
                    font_color='#94a3b8', title_font_color='#e2e8f0',
                    xaxis=dict(gridcolor='rgba(99,102,241,0.15)'),
                    yaxis=dict(gridcolor='rgba(99,102,241,0.15)'),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Tree Rules
                with st.expander("📋 View Decision Tree Rules (if-else logic)"):
                    tree_rules = export_text(clf, feature_names=dt_features)
                    st.code(tree_rules, language="text")

                st.markdown(f"""
                <div class="explain-box">
                <span class="step-label">Teacher ko kaise explain karein</span>
                Maine <strong>DecisionTreeClassifier</strong> use kiya hai <code>sklearn.tree</code> se.<br>
                Target column <code>{dt_target}</code> ko <strong>LabelEncoder</strong> se numeric banaya (text → numbers).<br>
                <code>max_depth={max_depth}</code> set kiya — tree zyada deep na ho, overfitting se bachne ke liye.<br>
                <strong>Accuracy = {acc*100:.1f}%</strong> — yeh model {acc*100:.1f}% test cases sahi classify kar pa raha hai.<br>
                <strong>Feature Importance</strong> graph dikhata hai kaunsa feature sabse zyada decision mein help karta hai.<br>
                <code>export_text()</code> se human-readable if-else rules dekh sakte hain!
                </div>
                """, unsafe_allow_html=True)

else:
    # ── EMPTY STATE ──
    st.markdown("""
    <div style="text-align:center; padding: 5rem 2rem; border: 1px dashed rgba(99,102,241,0.3); border-radius: 20px; margin-top: 2rem;">
        <div style="font-size:4rem; margin-bottom:1rem;">🧠</div>
        <div style="font-family:'Outfit',sans-serif; font-size:1.4rem; font-weight:700; color:#e2e8f0;">Ready to Analyze</div>
        <div style="color:#4b5563; margin-top:0.5rem; font-size:0.95rem;">Upload a CSV file from the sidebar to begin →</div>
        <div style="margin-top:1.5rem; display:flex; gap:1rem; justify-content:center; flex-wrap:wrap;">
            <span style="background:rgba(99,102,241,0.15); color:#818cf8; padding:6px 16px; border-radius:20px; font-size:0.82rem; border:1px solid rgba(99,102,241,0.3);">📊 Statistics</span>
            <span style="background:rgba(192,132,252,0.15); color:#c084fc; padding:6px 16px; border-radius:20px; font-size:0.82rem; border:1px solid rgba(192,132,252,0.3);">🔮 Regression</span>
            <span style="background:rgba(244,114,182,0.15); color:#f472b6; padding:6px 16px; border-radius:20px; font-size:0.82rem; border:1px solid rgba(244,114,182,0.3);">🎯 Clustering</span>
            <span style="background:rgba(6,182,212,0.15); color:#67e8f9; padding:6px 16px; border-radius:20px; font-size:0.82rem; border:1px solid rgba(6,182,212,0.3);">🌲 Classifier</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
