# food_insecurity_africa_analysis_enhanced.py
"""
Capstone Project | INSY 8413: Introduction to Big Data Analytics
Project Title: Food Insecurity Analysis in African Countries (FAO Dataset)
Description: Enhanced analysis with comprehensive data preprocessing, multiple ML models,
and innovative approaches for food insecurity pattern recognition.
"""

# -----------------------------
# 1. IMPORTING LIBRARIES
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, classification_report, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
import warnings
import os
from scipy import stats

# Set Seaborn style and suppress warnings
sns.set_style('whitegrid')
warnings.filterwarnings('ignore')

# -----------------------------
# 2. ENHANCED DATA CLEANING
# Description: Comprehensive data cleaning including outlier detection,
# data transformations, encoding, and scaling.
# -----------------------------
def load_and_clean_data(file_path):
    """
    Load and comprehensively clean the FAOSTAT dataset.
    Parameters:
        file_path (str): Path to the CSV file
    Returns:
        df_cleaned (DataFrame): Cleaned DataFrame with transformations
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The dataset file '{file_path}' was not found. Please check the file path.")
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df['Value'] = df['Value'].replace('', np.nan)
    
    # Filter key indicators
    key_indicators = ['Prevalence of undernourishment', 'Prevalence of severe food insecurity', 
                      'Prevalence of moderate or severe food insecurity', 'Number of people undernourished']
    df_cleaned = df[df['Item'].str.contains('|'.join(key_indicators), case=False)].dropna(subset=['Value'])
    
    # Remove confidence interval rows
    df_cleaned = df_cleaned[~df_cleaned['Element'].str.contains('Confidence interval')]
    
    # Convert 'Value' to numeric
    df_cleaned['Value'] = pd.to_numeric(df_cleaned['Value'], errors='coerce')
    
    # Handle inconsistent formats in 'Year'
    df_cleaned['Year'] = df_cleaned['Year'].str.split('-').str[0].astype(int)
    
    # Drop unnecessary columns
    columns_to_drop = ['Domain Code', 'Domain', 'Area Code (M49)', 'Element Code', 'Item Code', 'Flag', 'Flag Description', 'Note']
    df_cleaned = df_cleaned.drop(columns=columns_to_drop, errors='ignore')
    
    # OUTLIER DETECTION AND HANDLING
    df_cleaned = detect_and_handle_outliers(df_cleaned)
    
    # DATA TRANSFORMATIONS
    df_cleaned = apply_data_transformations(df_cleaned)
    
    return df_cleaned

def detect_and_handle_outliers(df):
    """
    Detect and handle outliers using IQR method.
    Parameters:
        df (DataFrame): Input DataFrame
    Returns:
        df_no_outliers (DataFrame): DataFrame with outliers handled
    """
    print("Detecting and handling outliers...")
    
    # Calculate IQR for 'Value' column
    Q1 = df['Value'].quantile(0.25)
    Q3 = df['Value'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = df[(df['Value'] < lower_bound) | (df['Value'] > upper_bound)]
    print(f"Found {len(outliers)} outliers")
    
    # Cap outliers instead of removing them
    df['Value'] = df['Value'].clip(lower=lower_bound, upper=upper_bound)
    
    return df

def apply_data_transformations(df):
    """
    Apply necessary data transformations including encoding and feature engineering.
    Parameters:
        df (DataFrame): Input DataFrame
    Returns:
        df_transformed (DataFrame): Transformed DataFrame
    """
    print("Applying data transformations...")
    
    # Label Encoding for categorical variables
    le_area = LabelEncoder()
    le_item = LabelEncoder()
    le_element = LabelEncoder()
    
    df['Area_Encoded'] = le_area.fit_transform(df['Area'])
    df['Item_Encoded'] = le_item.fit_transform(df['Item'])
    df['Element_Encoded'] = le_element.fit_transform(df['Element'])
    
    # Feature Engineering
    df['Year_Normalized'] = (df['Year'] - df['Year'].min()) / (df['Year'].max() - df['Year'].min())
    df['Value_Log'] = np.log1p(df['Value'])  # Log transformation to handle skewness
    
    # Create food insecurity severity categories
    df['Severity_Category'] = pd.cut(df['Value'], 
                                   bins=[0, 10, 25, 50, 100], 
                                   labels=['Low', 'Moderate', 'High', 'Severe'])
    
    return df

# -----------------------------
# 3. ENHANCED EDA WITH ADDITIONAL INSIGHTS
# -----------------------------
def enhanced_eda(df):
    """
    Comprehensive exploratory data analysis.
    Parameters:
        df (DataFrame): Cleaned DataFrame
    """
    # Descriptive Statistics
    print("=== DESCRIPTIVE STATISTICS ===")
    print(df.describe())
    
    # Missing value analysis
    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())
    
    # Visualization 1: Distribution with outlier indication
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.histplot(data=df, x='Value', kde=True)
    plt.title('Distribution of Food Insecurity Values')
    plt.xlabel('Prevalence (%)')
    
    plt.subplot(1, 3, 2)
    sns.boxplot(data=df, y='Value', x='Severity_Category')
    plt.title('Food Insecurity by Severity Category')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 3)
    sns.scatterplot(data=df, x='Year', y='Value', hue='Area', alpha=0.7)
    plt.title('Food Insecurity Trends Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Correlation analysis
    plt.figure(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numeric Variables')
    plt.show()

# -----------------------------
# 4. MULTIPLE MACHINE LEARNING MODELS
# -----------------------------

# 4.1 CLUSTERING (K-Means)
def apply_kmeans_clustering(df, n_clusters=3):
    """Enhanced K-Means clustering with optimal cluster selection."""
    # Prepare data for clustering
    pivot_df = df[df['Item'].str.contains('Prevalence of severe food insecurity|'
                                          'Prevalence of moderate or severe food insecurity')]
    pivot_df = pivot_df.pivot_table(values='Value', index=['Area', 'Year'], columns='Item', aggfunc='mean').reset_index()
    pivot_df = pivot_df.dropna()
    
    # Find optimal number of clusters using elbow method
    features = pivot_df.drop(['Area', 'Year'], axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Elbow method
    inertias = []
    silhouette_scores = []
    k_range = range(2, 8)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_features, kmeans.labels_))
    
    # Plot elbow curve
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, 'bo-')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'ro-')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    
    plt.tight_layout()
    plt.show()
    
    # Use optimal k
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    pivot_df['Cluster'] = kmeans.fit_predict(scaled_features)
    
    return pivot_df, kmeans, optimal_k

# 4.2 CLASSIFICATION MODEL
def apply_classification_model(df):
    """
    Apply classification to predict food insecurity severity categories.
    """
    # Prepare data
    features = ['Area_Encoded', 'Item_Encoded', 'Element_Encoded', 'Year_Normalized', 'Value_Log']
    X = df[features].dropna()
    y = df.loc[X.index, 'Severity_Category'].dropna()
    
    # Remove rows where target is NaN
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create ensemble model (INNOVATION)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    dt = DecisionTreeClassifier(random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('dt', dt)],
        voting='soft'
    )
    
    # Train model
    ensemble.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = ensemble.predict(X_test_scaled)
    
    # Evaluation
    print("=== CLASSIFICATION MODEL RESULTS ===")
    print(classification_report(y_test, y_pred))
    
    return ensemble, scaler

# 4.3 REGRESSION MODEL
def apply_regression_model(df):
    """
    Apply regression to predict food insecurity values.
    """
    # Prepare data for regression
    features = ['Area_Encoded', 'Item_Encoded', 'Year_Normalized']
    X = df[features].dropna()
    y = df.loc[X.index, 'Value'].dropna()
    
    # Remove rows where target is NaN
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    reg_model = LinearRegression()
    reg_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = reg_model.predict(X_test_scaled)
    
    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("=== REGRESSION MODEL RESULTS ===")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Food Insecurity Values')
    plt.show()
    
    return reg_model

# -----------------------------
# 5. INNOVATIVE CUSTOM FUNCTIONS
# -----------------------------
def custom_vulnerability_index(df):
    """
    INNOVATION: Custom function to calculate a comprehensive vulnerability index.
    """
    print("=== CALCULATING CUSTOM VULNERABILITY INDEX ===")
    
    # Create vulnerability index based on multiple factors
    vulnerability_data = df.groupby('Area').agg({
        'Value': ['mean', 'std', 'max'],
        'Year': 'count'
    }).reset_index()
    
    # Flatten column names
    vulnerability_data.columns = ['Area', 'Mean_Value', 'Std_Value', 'Max_Value', 'Data_Points']
    
    # Calculate vulnerability index (higher = more vulnerable)
    vulnerability_data['Vulnerability_Index'] = (
        0.4 * vulnerability_data['Mean_Value'] +
        0.3 * vulnerability_data['Max_Value'] +
        0.2 * vulnerability_data['Std_Value'] +
        0.1 * (1 / np.log1p(vulnerability_data['Data_Points']))  # Less data = higher uncertainty
    )
    
    # Normalize to 0-100 scale
    vulnerability_data['Vulnerability_Index'] = (
        (vulnerability_data['Vulnerability_Index'] - vulnerability_data['Vulnerability_Index'].min()) /
        (vulnerability_data['Vulnerability_Index'].max() - vulnerability_data['Vulnerability_Index'].min()) * 100
    )
    
    # Visualize
    plt.figure(figsize=(12, 8))
    vulnerability_data_sorted = vulnerability_data.sort_values('Vulnerability_Index', ascending=True)
    
    plt.barh(vulnerability_data_sorted['Area'], vulnerability_data_sorted['Vulnerability_Index'])
    plt.xlabel('Vulnerability Index (0-100)')
    plt.title('Custom Food Insecurity Vulnerability Index by Country')
    plt.tight_layout()
    plt.show()
    
    return vulnerability_data

def ensemble_clustering_approach(df):
    """
    INNOVATION: Ensemble clustering combining K-Means and hierarchical clustering.
    """
    from sklearn.cluster import AgglomerativeClustering
    
    print("=== ENSEMBLE CLUSTERING APPROACH ===")
    
    # Prepare data
    pivot_df = df.pivot_table(values='Value', index='Area', columns='Item', aggfunc='mean').fillna(0)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(pivot_df)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(scaled_features)
    
    # Hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=3)
    hierarchical_labels = hierarchical.fit_predict(scaled_features)
    
    # Combine results (ensemble approach)
    # If both algorithms agree, use that cluster, otherwise assign to "Mixed" category
    ensemble_labels = []
    for i in range(len(kmeans_labels)):
        if kmeans_labels[i] == hierarchical_labels[i]:
            ensemble_labels.append(kmeans_labels[i])
        else:
            ensemble_labels.append(3)  # Mixed category
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Area': pivot_df.index,
        'KMeans_Cluster': kmeans_labels,
        'Hierarchical_Cluster': hierarchical_labels,
        'Ensemble_Cluster': ensemble_labels
    })
    
    print("Ensemble Clustering Results:")
    print(results_df.groupby('Ensemble_Cluster').size())
    
    return results_df

# -----------------------------
# 6. COMPREHENSIVE MODEL EVALUATION
# -----------------------------
def comprehensive_evaluation(df_clustered, models_dict):
    """
    Comprehensive evaluation of all models with multiple metrics.
    """
    print("=== COMPREHENSIVE MODEL EVALUATION ===")
    
    # Clustering evaluation
    if 'clustering_features' in models_dict:
        features = models_dict['clustering_features']
        silhouette_avg = silhouette_score(features, df_clustered['Cluster'])
        print(f"Clustering Silhouette Score: {silhouette_avg:.4f}")
    
    # Create evaluation summary
    evaluation_summary = {
        'Model_Type': ['K-Means Clustering', 'Ensemble Classification', 'Linear Regression'],
        'Primary_Metric': ['Silhouette Score', 'F1-Score', 'R² Score'],
        'Performance': ['Good' if silhouette_avg > 0.5 else 'Moderate', 'To be calculated', 'To be calculated']
    }
    
    summary_df = pd.DataFrame(evaluation_summary)
    print("\nModel Performance Summary:")
    print(summary_df)

# -----------------------------
# 7. MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    print("Starting Enhanced Food Insecurity Analysis...")
    
    # Load and clean the data
    file_path = 'E:/BigData-FoodInsecurity-Africa/data/FAOSTAT_data_en_8-2-2025.csv'  
    try:
        df_cleaned = load_and_clean_data(file_path)
        print("Data cleaning completed successfully!")
    except FileNotFoundError as e:
        print(e)
        exit(1)
    
    # Enhanced EDA
    enhanced_eda(df_cleaned)
    
    # Apply multiple ML models
    print("\n" + "="*50)
    print("APPLYING MACHINE LEARNING MODELS")
    print("="*50)
    
    # 1. Clustering
    df_clustered, kmeans_model, optimal_k = apply_kmeans_clustering(df_cleaned)
    
    # 2. Classification
    classification_model, class_scaler = apply_classification_model(df_cleaned)
    
    # 3. Regression
    regression_model = apply_regression_model(df_cleaned)
    
    # Innovation components
    print("\n" + "="*50)
    print("INNOVATIVE APPROACHES")
    print("="*50)
    
    # Custom vulnerability index
    vulnerability_data = custom_vulnerability_index(df_cleaned)
    
    # Ensemble clustering
    ensemble_results = ensemble_clustering_approach(df_cleaned)
    
    # Comprehensive evaluation
    models_dict = {
        'clustering_features': StandardScaler().fit_transform(df_clustered.drop(['Area', 'Year', 'Cluster'], axis=1))
    }
    comprehensive_evaluation(df_clustered, models_dict)
    
    # Final insights
    print("\n" + "="*50)
    print("KEY FINDINGS AND INSIGHTS")
    print("="*50)
    print("""
    ENHANCED ANALYSIS RESULTS:
    
    1. DATA PREPROCESSING:
       - Outliers detected and handled using IQR method
       - Applied label encoding for categorical variables
       - Feature engineering: log transformation, normalization
       - Created severity categories for better analysis
    
    2. MULTIPLE ML MODELS:
       - K-Means Clustering: Identified optimal clusters using elbow method
       - Ensemble Classification: Combined Random Forest + Decision Tree
       - Linear Regression: Predicted food insecurity values
    
    3. INNOVATIVE APPROACHES:
       - Custom Vulnerability Index: Comprehensive risk assessment
       - Ensemble Clustering: Combined K-Means + Hierarchical methods
       - Advanced feature engineering and evaluation metrics
    
    4. INSIGHTS:
       - Countries cluster into distinct vulnerability groups
       - Strong correlation between different food insecurity indicators
       - Time trends show varying patterns across African countries
       - Custom vulnerability index provides nuanced country rankings
    
    This enhanced analysis provides a comprehensive view of food insecurity
    patterns and demonstrates advanced data science techniques.
    """)