# 🌍 Food Insecurity in Africa – Big Data Capstone Project

> **Course**: INSY 8413 – Introduction to Big Data Analytics  
> **Student**: Daniel Kambale Ngununu  
> **Dataset Source**: [FAO Food Security Indicators](https://www.fao.org/faostat/)

---

## 🔍 Problem Statement

**Can we analyze patterns and trends in food insecurity across African countries to identify the most vulnerable populations and regions?**

---

## 🧠 Objectives

- Identify countries with the highest undernourishment.
- Discover hidden clusters using unsupervised learning.
- Investigate gender-based food insecurity.
- Present insights via interactive Power BI dashboard.

---

## 📊 Dataset Overview

- **Rows**: ~1,000  
- **Columns**: 15  
- **Format**: CSV  
- **Content**: Indicators of food insecurity, malnutrition, and gender breakdown by African country.

---

## 🧪 Python Tasks Summary

| Task                     | Techniques Used                                 |
|--------------------------|-------------------------------------------------|
| Data Cleaning            | Missing value handling, filtering               |
| EDA                      | Grouping, time series, seaborn/Matplotlib       |
| Clustering               | KMeans + Elbow method + Silhouette evaluation   |
| Innovation               | Custom gender gap analysis                      |

---

## 🧠 Power BI Dashboard Features

> *(File: `FoodSecurity_Africa.pbix`)*

- Interactive filters by country and year  
- Dynamic visuals (bar, line, map)  
- Tooltips and slicers for deep dive  
- Cluster insights with exportable summary  
- Gender disparity page (innovative page)

---

## 🧾 How to Run

1. Clone this repository.
2. Open the notebook: `food_insecurity_africa_analysis.ipynb`
3. Install requirements if needed:

```bash
pip install pandas matplotlib seaborn scikit-learn

---

## 💡 Future Work

- Incorporate time-based forecasting (e.g., Prophet)
- Extend dataset with economic indicators (World Bank)
- Develop API to deliver visual insights

