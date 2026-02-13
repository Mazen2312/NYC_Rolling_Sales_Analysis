
# 🏙 NYC Real Estate Market Analysis

An exploratory data analysis project on New York City real estate sales data.  
This project analyzes property prices, market trends, and key factors affecting real estate value using Python and Streamlit.

---

## 📖 Project Overview

The goal of this project is to analyze NYC property sales data to understand:

- What factors influence property prices
- How location impacts real estate value
- Market behavior over time
- Price distribution patterns
- Relationship between size, age, and sale price

The project includes data cleaning, feature engineering, exploratory data analysis (EDA), and an interactive dashboard built with Streamlit.

---

## 📂 Dataset

- Dataset: NYC Rolling Sales Data
- Source: Public NYC real estate sales dataset
- Target variable: `sale_price`

The dataset contains information about:
- Borough
- Sale price
- Gross square feet
- Land square feet
- Year built
- Total units
- Sale date

---

## 🧹 Data Cleaning & Preparation

The following preprocessing steps were applied:

- Removed invalid sale prices
- Converted numeric columns
- Replaced zero values with NaN
- Handled missing values using median
- Removed extreme outliers using IQR method
- Created new features:
  - `price_per_sqft`
  - `property_age`
  - `sale_year`

---

## 📊 Exploratory Data Analysis (EDA)

Key analyses performed:

- Sale price distribution
- Log-transformed price distribution
- Borough comparison
- Price per square foot analysis
- Property size vs sale price
- Property age vs sale price
- Market trends over time
- Correlation analysis

---

## 🔎 Key Insights

- Sale prices are highly skewed due to extreme high-value transactions.
- Location significantly impacts both total price and price per square foot.
- Larger properties generally sell at higher prices.
- Newer properties tend to have higher market value.
- Residential properties dominate the transaction volume.
- Market trends fluctuate over time.

---

## 📈 Interactive Dashboard

An interactive dashboard was built using Streamlit to allow users to:

- Filter by borough
- Filter by year
- Filter by price range
- View KPIs and summary statistics
- Explore visual insights dynamically

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Plotly
- Streamlit

---
## 🔗 Reference
This project is developed under the Epsilon AI DS Diploma.
