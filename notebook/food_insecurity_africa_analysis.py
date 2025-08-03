# food_insecurity_africa_analysis.py

"""
Capstone Project | INSY 8413: Introduction to Big Data Analytics
Student: Daniel Kambale Ngununu
Project Title: Food Insecurity Analysis in African Countries (FAO Dataset)
"""

# -----------------------------
# 📦 1. IMPORTING LIBRARIES
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Plot settings
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10,6)

# -----------------------------
# 📥 2. LOAD AND CLEAN DATASET
# -----------------------------
# Load dataset
df = pd.read_csv("FAOSTAT_data_en_8-2-2025.csv")

# Keep relevant columns
columns = ['Area', 'Element', 'Item', 'Year', 'Unit', 'Value', 'Flag Description']
df = df[columns]

# Replace missing or invalid values
df['Value'] = df['Value'].replace(['O', 'Q', '', ' '], np.nan)
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

# Drop rows with no country or year
df.dropna(subset=['Area', 'Year', 'Value'], inplace=True)

# -----------------------------
# 🔎 3. DATA SEGMENTATION
# -----------------------------
# Segmenting the dataset by indicators
undernourishment = df[df['Item'].str.contains('undernourishment', case=False)]
food_insecurity = df[df['Item'].str.contains('food insecurity', case=False)]

# -----------------------------
# 📊 4. EXPLORATORY DATA ANALYSIS
# -----------------------------

# 4.1 Undernourishment by country (latest year)
latest_year = undernourishment['Year'].max()
latest_data = undernourishment[undernourishment['Year'] == latest_year]

top_countries = latest_data.sort_values('Value', ascending=False).head(10)

sns.barplot(data=top_countries, y='Area', x='Value', palette='Reds_r')
plt.title(f'Top 10 African Countries by Undernourishment ({latest_year})')
plt.xlabel('Undernourishment (%)')
plt.ylabel('Country')
plt.tight_layout()
plt.show()

# 4.2 Time series trends
countries = ['Kenya', 'Ethiopia', 'Somalia', 'South Sudan']
for country in countries:
    country_data = undernourishment[undernourishment['Area'] == country]
    plt.plot(country_data['Year'], country_data['Value'], label=country)

plt.title('Undernourishment Trends (Selected Countries)')
plt.xlabel('Year')
plt.ylabel('Undernourishment (%)')
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 📈 5. MACHINE LEARNING – CLUSTERING
# -----------------------------
# Pivot data for clustering
pivot = undernourishment.pivot_table(index='Area', columns='Year', values='Value')
pivot.fillna(pivot.mean(), inplace=True)

# Normalize data
scaler = StandardScaler()
scaled = scaler.fit_transform(pivot)

# Elbow method to find optimal clusters
inertia = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(scaled)
    inertia.append(model.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Apply clustering
kmeans = KMeans(n_clusters=3, random_state=42)
pivot['Cluster'] = kmeans.fit_predict(scaled)

# -----------------------------
# 📐 6. MODEL EVALUATION
# -----------------------------
score = silhouette_score(scaled, pivot['Cluster'])
print(f"\nSilhouette Score: {score:.2f}\n")

for i in sorted(pivot['Cluster'].unique()):
    print(f"Cluster {i} Countries:\n", pivot[pivot['Cluster'] == i].index.tolist())

# -----------------------------
# 🚀 7. INNOVATION – GENDER GAP
# -----------------------------
def analyze_gender_disparities(df):
    male = df[df['Item'].str.contains('male', case=False)]
    female = df[df['Item'].str.contains('female', case=False)]

    # Grouping average by country
    result = pd.merge(
        male.groupby('Area')['Value'].mean().reset_index(),
        female.groupby('Area')['Value'].mean().reset_index(),
        on='Area',
        suffixes=('_male', '_female')
    )

    result['Gap (Female - Male)'] = result['Value_female'] - result['Value_male']
    return result.sort_values('Gap (Female - Male)', ascending=False)

gender_gap = analyze_gender_disparities(food_insecurity)
print("\nTop Gender Disparities (Female vs Male):\n", gender_gap.head())
