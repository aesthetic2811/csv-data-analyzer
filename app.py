import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Page Configuration (Browser Tab Title & Icon)
st.set_page_config(
    page_title="DataScope Analytics",
    page_icon="🚀",
    layout="wide"
)

# 2. Sidebar (Left Side Menu)
with st.sidebar:
    st.header("📂 Configuration")
    st.write("Upload your CSV file to generate insights instantly.")
    
    # File Uploader in Sidebar
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    st.markdown("---")
    st.write("Developed by **[Janvi Sharma]**") 
    st.write("Powered by Streamlit & Plotly")

# 3. Main Page Title
st.title("🚀 DataScope: Interactive CSV Explorer")
st.markdown("Analyze your data with **Statistics** and **Visualizations** in one place.")

# 4. Main App Logic
if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)

    # --- TABS LAYOUT (Professional Look) ---
    tab1, tab2, tab3 = st.tabs(["📄 Dataset Overview", "📊 Statistical Summary", "📈 Data Visualization"])

    # --- TAB 1: DATASET OVERVIEW ---
    with tab1:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(10)) # Top 10 rows
        
        # Display Shape neatly
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Total Rows:** {df.shape[0]}")
        with col2:
            st.info(f"**Total Columns:** {df.shape[1]}")
            
        # Show Column Names
        with st.expander("Show All Column Names"):
            st.write(list(df.columns))

    # --- TAB 2: STATISTICAL SUMMARY ---
    with tab2:
        st.subheader("Descriptive Statistics")
        st.write(df.describe())
        
        # Missing Values Check
        st.subheader("Missing Values")
        st.write(df.isnull().sum())

    # --- TAB 3: VISUALIZATION ---
    with tab3:
        st.subheader("Create Custom Graphs")
        
        # Grid layout for inputs
        c1, c2, c3 = st.columns(3)
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        all_cols = df.columns.tolist()

        with c1:
            x_axis = st.selectbox("Select X-Axis", all_cols)
        with c2:
            y_axis = st.selectbox("Select Y-Axis", numeric_cols)
        with c3:
            plot_type = st.selectbox("Select Graph Type", ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot"])

        # Generate Plot
        if st.button("Generate Visualization"):
            if plot_type == "Bar Chart":
                fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}", color=x_axis)
            elif plot_type == "Line Chart":
                fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} Trend")
            elif plot_type == "Scatter Plot":
                fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Relationship: {x_axis} vs {y_axis}", color=x_axis)
            elif plot_type == "Histogram":
                fig = px.histogram(df, x=x_axis, title=f"Distribution of {x_axis}")
            elif plot_type == "Box Plot":
                fig = px.box(df, y=y_axis, title=f"Box Plot of {y_axis}")
            
            st.plotly_chart(fig, use_container_width=True)

else:
    # Empty State (Jab file upload nahi hui ho)
    st.info("👈 Please upload a CSV file from the sidebar to get started!")