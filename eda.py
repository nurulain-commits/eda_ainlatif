import streamlit as st
import pandas as pd
import numpy as np
import dtale
import plotly.express as px
import streamlit.components.v1 as components
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="EDA Web App by Dr Ku",
    page_icon="",
    layout="wide"
)

# =========================
# Sidebar: Logo and Developer
# =========================
st.sidebar.image(
    "https://brand.umpsa.edu.my/images/logo-umpsa-full-color2.png", 
    use_container_width=True
)

st.sidebar.image(
    "https://www.majalahsains.com/wp-content/uploads/2012/05/Logo-Agensi-Nuklear-Malaysia.png",
    use_container_width=True
)    

st.sidebar.markdown("## EDA Analytics Dashboard")
st.sidebar.markdown("---")

st.sidebar.markdown("### Developers:")
st.sidebar.write("***Assoc. Prof. Dr. Ku Muhammad Naim Ku Khalif***")
st.sidebar.write("Centre for Mathematical Sciences")
st.sidebar.write("Universiti Malaysia Pahang Al-Sultan Abdullah")
st.sidebar.write("Email: kunaim@umpsa.edu.my")

st.sidebar.write("***Dr. Nurul A'in binti Ahmad Latif***")
st.sidebar.write("Bahagian Teknologi Industri (BTI)")
st.sidebar.write("Agensi Nuklear Malaysia")
st.sidebar.write("Email: nurul_ain@nm.gov.my")

st.sidebar.markdown("---")

# =========================
# Main title
# =========================
st.title("Exploratory Data Analysis Dashboard")
st.caption("Interactive EDA using Streamlit and D-Tale")

# =========================
# File Upload
# =========================
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

# =========================
# Main App
# =========================
if uploaded_file is not None:
    df = load_data(uploaded_file)

    st.success("Dataset uploaded successfully.")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Dataset Information")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.subheader("Column Summary")
    summary = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })
    st.dataframe(summary, use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.subheader("Missing Values by Column")
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ["Column", "Missing Values"]
    missing_df = missing_df[missing_df["Missing Values"] > 0]

    if not missing_df.empty:
        fig_missing = px.bar(
            missing_df,
            x="Column",
            y="Missing Values",
            title="Missing Values Distribution"
        )
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.info("No missing values found.")

    st.subheader("Data Visualization")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if numeric_cols:
        selected_col = st.selectbox(
            "Select numerical column",
            numeric_cols
        )

        fig_hist = px.histogram(
            df,
            x=selected_col,
           nbins=30,
            title=f"Distribution of {selected_col}"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        if len(numeric_cols) >= 2:
            # Include datetime columns for X-axis
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            x_axis_cols = numeric_cols + datetime_cols
            
            x_col = st.selectbox("X-axis", x_axis_cols, index=0)
            y_col = st.selectbox("Y-axis", numeric_cols, index=1)

            fig_scatter = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f"{x_col} vs {y_col}"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Correlation Heatmap")
        corr = df[numeric_cols].corr()

        fig_corr = px.imshow(
            corr,
            text_auto=True,
            title="Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("No numerical columns available for visualization.")

    # =========================
    # Data Cleaning Section
    # =========================
    st.subheader("Data Cleaning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.checkbox("Handle Missing Values"):
            st.write("**Missing Values Strategy:**")
            
            missing_cols = df.columns[df.isnull().any()].tolist()
            
            if missing_cols:
                for col in missing_cols:
                    strategy = st.selectbox(
                        f"Strategy for {col}",
                        ["Drop", "Mean", "Median", "Mode", "Forward Fill"],
                        key=f"missing_{col}"
                    )
                    
                    if strategy == "Drop":
                        df = df.dropna(subset=[col])
                    elif strategy == "Mean" and df[col].dtype in [np.int64, np.float64]:
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif strategy == "Median" and df[col].dtype in [np.int64, np.float64]:
                        df[col].fillna(df[col].median(), inplace=True)
                    elif strategy == "Mode":
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    elif strategy == "Forward Fill":
                        df[col].fillna(method="ffill", inplace=True)
                
                st.success("Missing values handled successfully!")
            else:
                st.info("No missing values found in the dataset.")
    
    with col2:
        if st.checkbox("Remove Duplicates"):
            duplicate_count = int(df.duplicated().sum())
            if duplicate_count > 0:
                if st.button(f"Remove {duplicate_count} duplicate rows"):
                    df = df.drop_duplicates()
                    st.success(f"Removed {duplicate_count} duplicate rows!")
            else:
                st.info("No duplicate rows found.")
    
    
        # =========================
    # Download Cleaned Dataset
    # =========================
    st.subheader("Download Cleaned Dataset")
    cleaned_filename = f"cleaned_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Cleaned Dataset as CSV",
        data=csv,
        file_name=cleaned_filename,
        mime="text/csv",
    )
    
    # =========================
    # Feature Engineering Section
    # =========================
    st.subheader("Feature Engineering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.checkbox("Normalize Numeric Columns"):
            numeric_cols_fe = df.select_dtypes(include=np.number).columns.tolist()
            
            if numeric_cols_fe:
                cols_to_normalize = st.multiselect(
                    "Select columns to normalize",
                    numeric_cols_fe,
                    key="normalize_cols"
                )
                
                normalize_method = st.selectbox(
                    "Normalization method",
                    ["Min-Max (0-1)", "Z-Score Standardization"],
                    key="normalize_method"
                )
                
                if st.button("Apply Normalization"):
                    for col in cols_to_normalize:
                        if normalize_method == "Min-Max (0-1)":
                            min_val = df[col].min()
                            max_val = df[col].max()
                            df[f"{col}_normalized"] = (df[col] - min_val) / (max_val - min_val)
                        else:  # Z-Score
                            mean_val = df[col].mean()
                            std_val = df[col].std()
                            df[f"{col}_normalized"] = (df[col] - mean_val) / std_val
                    
                    st.success("Normalization applied successfully!")
    
    with col2:
        if st.checkbox("Create Interaction Features"):
            numeric_cols_fe = df.select_dtypes(include=np.number).columns.tolist()
            
            if len(numeric_cols_fe) >= 2:
                col_1 = st.selectbox("First column", numeric_cols_fe, key="interaction_1")
                col_2 = st.selectbox("Second column", numeric_cols_fe, key="interaction_2", index=1)
                
                interaction_type = st.selectbox(
                    "Interaction type",
                    ["Multiply", "Add", "Subtract", "Divide"],
                    key="interaction_type"
                )
                
                if st.button("Create Interaction"):
                    if interaction_type == "Multiply":
                        df[f"{col_1}_x_{col_2}"] = df[col_1] * df[col_2]
                    elif interaction_type == "Add":
                        df[f"{col_1}_add_{col_2}"] = df[col_1] + df[col_2]
                    elif interaction_type == "Subtract":
                        df[f"{col_1}_sub_{col_2}"] = df[col_1] - df[col_2]
                    elif interaction_type == "Divide":
                        df[f"{col_1}_div_{col_2}"] = df[col_1] / df[col_2].replace(0, np.nan)
                    
                    st.success("Interaction feature created successfully!")

    # =========================
    # K-Means Clustering Section
    # =========================
    st.subheader("K-Means Clustering Analysis")
    
    numeric_cols_km = df.select_dtypes(include=np.number).columns.tolist()
    
    if len(numeric_cols_km) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Select columns for clustering
            clustering_cols = st.multiselect(
                "Select numerical columns for clustering",
                numeric_cols_km,
                default=numeric_cols_km[:2] if len(numeric_cols_km) >= 2 else numeric_cols_km,
                key="clustering_cols"
            )
        
        with col2:
            # Select number of clusters
            n_clusters = st.slider(
                "Number of clusters (k)",
                min_value=2,
                max_value=10,
                value=3,
                key="n_clusters"
            )
        
        if clustering_cols:
            if st.button("Run K-Means Clustering"):
                try:
                    # Prepare data
                    X = df[clustering_cols].copy()
                    
                    # Handle missing values if any
                    if X.isnull().any().any():
                        st.warning("Missing values detected in selected columns. Dropping rows with missing values.")
                        X = X.dropna()
                    
                    # Standardize the features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Apply K-Means
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(X_scaled)
                    
                    # Add cluster labels to dataframe
                    df_plot = X.copy()
                    df_plot['Cluster'] = clusters
                    
                    st.success(f"K-Means clustering completed with {n_clusters} clusters!")
                    
                    # Display cluster information
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Clusters", n_clusters)
                    col2.metric("Inertia", f"{kmeans.inertia_:.2f}")
                    col3.metric("Data Points", len(df_plot))
                    
                    # Display cluster distribution
                    st.write("**Cluster Distribution:**")
                    cluster_counts = pd.Series(clusters).value_counts().sort_index()
                    fig_cluster_dist = px.bar(
                        x=cluster_counts.index.astype(str),
                        y=cluster_counts.values,
                        labels={"x": "Cluster", "y": "Number of Points"},
                        title="Number of Points per Cluster"
                    )
                    st.plotly_chart(fig_cluster_dist, use_container_width=True)
                    
                    # Visualization - 2D scatter plot
                    if len(clustering_cols) >= 2:
                        fig_scatter = px.scatter(
                            df_plot,
                            x=clustering_cols[0],
                            y=clustering_cols[1],
                            color='Cluster',
                            title=f"K-Means Clustering: {clustering_cols[0]} vs {clustering_cols[1]}",
                            color_continuous_scale="Viridis",
                            hover_data=clustering_cols
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Visualization - 3D scatter plot if 3+ columns selected
                    if len(clustering_cols) >= 3:
                        fig_3d = px.scatter_3d(
                            df_plot,
                            x=clustering_cols[0],
                            y=clustering_cols[1],
                            z=clustering_cols[2],
                            color='Cluster',
                            title="3D K-Means Clustering Visualization",
                            color_continuous_scale="Viridis"
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)
                    
                    # Cluster centers
                    st.write("**Cluster Centers (Original Scale):**")
                    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
                    centers_df = pd.DataFrame(
                        centers_original,
                        columns=clustering_cols
                    )
                    st.dataframe(centers_df, use_container_width=True)
                    
                    # Add clusters to main dataframe
                    df['KMeans_Cluster'] = clusters
                    
                    # Display clustered data sample
                    st.write("**Sample of Clustered Data:**")
                    st.dataframe(df[[*clustering_cols, 'KMeans_Cluster']].head(20), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during clustering: {str(e)}")
    else:
        st.warning("At least 2 numerical columns are required for clustering.")

    # =========================
    # D-Tale Section
    # =========================
    st.subheader("Interactive D-Tale EDA")

    d = dtale.show(
        df,
        subprocess=False,
        open_browser=False
    )

    dtale_url = d._main_url

    st.info("Click the link below to open the full interactive D-Tale analysis.")

    st.markdown(
        f"""
        ### [Open D-Tale Interactive EDA]({dtale_url})
        """
    )

    components.iframe(
        dtale_url,
        height=850,
        scrolling=True
    )

else:
    st.info("Please upload a CSV or Excel dataset from the sidebar.")
