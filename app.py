import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="NYC Real Estate Dashboard",
    page_icon="🏙️",
    layout="wide"
)

# =============================
# Load & Prepare Data
# =============================

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_nyc_rolling_sales.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Create additional features if not present
    if 'log_sale_price' not in df.columns:
        df['log_sale_price'] = np.log1p(df['sale_price'])

    if 'price_category' not in df.columns:
        df['price_category'] = pd.qcut(df['sale_price'], q=3, 
                                        labels=['Low', 'Medium', 'High'], 
                                        duplicates='drop')

    return df

df = load_data()

# =============================
# Header
# =============================
st.title("🏙️ NYC Real Estate Analysis Dashboard")
st.markdown("**Explore property sales data across all 5 NYC boroughs**")
st.markdown("---")

# =============================
# Sidebar Filters
# =============================
st.sidebar.header("📊 Filters")

# Borough filter
borough = st.sidebar.multiselect(
    "Select Borough(s)",
    options=sorted(df["borough"].unique()),
    default=df["borough"].unique()
)

# Year filter
available_years = sorted(df["sale_year"].unique())
year = st.sidebar.multiselect(
    "Select Year(s)",
    options=available_years,
    default=available_years
)

# Price range filter
price_min = int(df["sale_price"].min())
price_max = int(df["sale_price"].max())

min_price, max_price = st.sidebar.slider(
    "Price Range ($)",
    min_value=price_min,
    max_value=price_max,
    value=(10000, 1000000),
    step=10000,
    format="$%d"
)

# Property size filter
if st.sidebar.checkbox("Filter by Property Size", value=False):
    size_min = int(df["gross_square_feet"].min())
    size_max = int(df["gross_square_feet"].max())

    min_size, max_size = st.sidebar.slider(
        "Property Size (sqft)",
        min_value=size_min,
        max_value=min(size_max, 10000),
        value=(500, 5000),
        step=100
    )

    filtered_df = df[
        (df["borough"].isin(borough)) &
        (df["sale_year"].isin(year)) &
        (df["sale_price"].between(min_price, max_price)) &
        (df["gross_square_feet"].between(min_size, max_size))
    ]
else:
    filtered_df = df[
        (df["borough"].isin(borough)) &
        (df["sale_year"].isin(year)) &
        (df["sale_price"].between(min_price, max_price))
    ]

# Display filter info
st.sidebar.markdown("---")
st.sidebar.info(f"**{len(filtered_df):,}** properties match your filters")

# =============================
# Key Metrics (KPIs)
# =============================
st.subheader("📈 Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    avg_price = filtered_df['sale_price'].mean()
    st.metric(
        label="Average Price",
        value=f"${avg_price:,.0f}"
    )

with col2:
    median_price = filtered_df['sale_price'].median()
    st.metric(
        label="Median Price",
        value=f"${median_price:,.0f}"
    )

with col3:
    avg_sqft_price = filtered_df['price_per_sqft'].mean()
    st.metric(
        label="Avg Price/Sqft",
        value=f"${avg_sqft_price:,.0f}"
    )

with col4:
    avg_size = filtered_df['gross_square_feet'].mean()
    st.metric(
        label="Avg Size",
        value=f"{avg_size:,.0f} sqft"
    )

with col5:
    total_sales = len(filtered_df)
    st.metric(
        label="Total Sales",
        value=f"{total_sales:,}"
    )

st.markdown("---")

# =============================
# Tabs
# =============================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Price Analysis",
    "🗺️ Borough Comparison",
    "📈 Trends Over Time",
    "🔍 Property Analysis"
])

# -------------------------
# Tab 1 - Price Analysis
# -------------------------
with tab1:
    st.header("Price Distribution Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sale Price Distribution")
        fig1 = px.histogram(
            filtered_df, 
            x="sale_price", 
            nbins=40,
            title="Distribution of Sale Prices",
            labels={'sale_price': 'Sale Price ($)', 'count': 'Number of Properties'},
            color_discrete_sequence=['#1f77b4']
        )
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

        # Show statistics
        st.markdown(f"""
        **Statistics:**
        - Mean: ${filtered_df['sale_price'].mean():,.0f}
        - Median: ${filtered_df['sale_price'].median():,.0f}
        - Std Dev: ${filtered_df['sale_price'].std():,.0f}
        """)

    with col2:
        st.subheader("Log-Transformed Distribution")
        fig2 = px.histogram(
            filtered_df, 
            x="log_sale_price", 
            nbins=40,
            title="Log-Transformed Sale Price Distribution",
            labels={'log_sale_price': 'Log(Sale Price)', 'count': 'Number of Properties'},
            color_discrete_sequence=['#ff7f0e']
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

        st.info("Log transformation helps visualize the distribution when data is skewed.")

    # Price per sqft distribution
    st.subheader("Price per Square Foot Distribution")
    fig3 = px.histogram(
        filtered_df, 
        x="price_per_sqft", 
        nbins=50,
        title="Distribution of Price per Square Foot",
        labels={'price_per_sqft': 'Price per Sqft ($)', 'count': 'Number of Properties'},
        color_discrete_sequence=['#2ca02c']
    )
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    # Market segmentation
    st.subheader("Market Segmentation")
    col1, col2 = st.columns([1, 2])

    with col1:
        category_counts = filtered_df['price_category'].value_counts()
        fig4 = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Properties by Price Category",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        st.markdown("""
        ### Price Categories:

        Properties are divided into three equal groups:
        - **Low**: Bottom 33% of prices
        - **Medium**: Middle 33% of prices  
        - **High**: Top 33% of prices

        This segmentation helps identify market distribution and opportunities 
        in different price ranges.
        """)

# -------------------------
# Tab 2 - Borough Comparison
# -------------------------
with tab2:
    st.header("Borough-Level Analysis")

    # Average price by borough
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Sale Price by Borough")
        avg_price_borough = filtered_df.groupby("borough")["sale_price"].mean().reset_index()
        avg_price_borough = avg_price_borough.sort_values("sale_price", ascending=False)

        fig5 = px.bar(
            avg_price_borough, 
            x="borough", 
            y="sale_price",
            title="Average Sale Price by Borough",
            labels={'borough': 'Borough', 'sale_price': 'Average Sale Price ($)'},
            text_auto='.2s',
            color='sale_price',
            color_continuous_scale='Blues'
        )
        fig5.update_layout(showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        st.subheader("Average Price/Sqft by Borough")
        avg_sqft_borough = filtered_df.groupby("borough")["price_per_sqft"].mean().reset_index()
        avg_sqft_borough = avg_sqft_borough.sort_values("price_per_sqft", ascending=False)

        fig6 = px.bar(
            avg_sqft_borough, 
            x="borough", 
            y="price_per_sqft",
            title="Average Price per Sqft by Borough",
            labels={'borough': 'Borough', 'price_per_sqft': 'Avg Price/Sqft ($)'},
            text_auto='.2s',
            color='price_per_sqft',
            color_continuous_scale='Oranges'
        )
        fig6.update_layout(showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)

    # Box plot comparison
    st.subheader("Price Distribution by Borough")
    fig7 = px.box(
        filtered_df, 
        x="borough", 
        y="sale_price",
        title="Sale Price Distribution Across Boroughs",
        labels={'borough': 'Borough', 'sale_price': 'Sale Price ($)'},
        color='borough',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig7.update_layout(showlegend=False)
    st.plotly_chart(fig7, use_container_width=True)

    # Borough statistics table
    st.subheader("Detailed Borough Statistics")
    borough_stats = filtered_df.groupby('borough').agg({
        'sale_price': ['count', 'mean', 'median', 'std'],
        'price_per_sqft': 'mean',
        'gross_square_feet': 'mean'
    }).round(2)
    borough_stats.columns = ['Count', 'Avg Price', 'Median Price', 'Std Dev', 'Avg $/Sqft', 'Avg Size (sqft)']
    st.dataframe(borough_stats, use_container_width=True)

# -------------------------
# Tab 3 - Trends Over Time
# -------------------------
with tab3:
    st.header("Temporal Trends Analysis")

    # Price trend over time
    st.subheader("Average Sale Price Over Time")
    yearly_avg = filtered_df.groupby("sale_year")["sale_price"].mean().reset_index()

    fig8 = px.line(
        yearly_avg, 
        x="sale_year", 
        y="sale_price",
        title="Average Sale Price Trend",
        labels={'sale_year': 'Year', 'sale_price': 'Average Sale Price ($)'},
        markers=True,
        color_discrete_sequence=['#d62728']
    )
    fig8.update_traces(line=dict(width=3), marker=dict(size=10))
    st.plotly_chart(fig8, use_container_width=True)

    # Year-over-year growth
    if len(yearly_avg) > 1:
        yearly_avg['yoy_growth'] = yearly_avg['sale_price'].pct_change() * 100
        st.markdown("**Year-over-Year Growth Rate:**")
        st.dataframe(
            yearly_avg[['sale_year', 'sale_price', 'yoy_growth']].dropna(),
            use_container_width=True
        )

    # Sales volume over time
    st.subheader("Sales Volume Over Time")
    yearly_volume = filtered_df.groupby("sale_year").size().reset_index(name='count')

    fig9 = px.bar(
        yearly_volume,
        x="sale_year",
        y="count",
        title="Number of Sales by Year",
        labels={'sale_year': 'Year', 'count': 'Number of Sales'},
        text_auto=True,
        color='count',
        color_continuous_scale='Greens'
    )
    st.plotly_chart(fig9, use_container_width=True)

    # Trend by borough
    st.subheader("Price Trends by Borough")
    yearly_borough = filtered_df.groupby(["sale_year", "borough"])["sale_price"].mean().reset_index()

    fig10 = px.line(
        yearly_borough,
        x="sale_year",
        y="sale_price",
        color="borough",
        title="Average Price Trend by Borough",
        labels={'sale_year': 'Year', 'sale_price': 'Avg Price ($)', 'borough': 'Borough'},
        markers=True
    )
    st.plotly_chart(fig10, use_container_width=True)

# -------------------------
# Tab 4 - Property Analysis
# -------------------------
with tab4:
    st.header("Property Characteristics Analysis")

    # Property size vs price
    st.subheader("Property Size vs Sale Price")

    # Sample for better performance
    sample_size = min(5000, len(filtered_df))
    df_sample = filtered_df.sample(n=sample_size, random_state=42)

    fig11 = px.scatter(
        df_sample,
        x="gross_square_feet",
        y="sale_price",
        color="borough",
        title=f"Property Size vs Price (Sample of {sample_size:,} properties)",
        labels={'gross_square_feet': 'Size (sqft)', 'sale_price': 'Price ($)', 'borough': 'Borough'},
        opacity=0.6,
        hover_data=['price_per_sqft']
    )
    st.plotly_chart(fig11, use_container_width=True)

    correlation = filtered_df['sale_price'].corr(filtered_df['gross_square_feet'])
    st.info(f"**Correlation between size and price:** {correlation:.3f}")

    # Property age vs price
    st.subheader("Property Age vs Sale Price")

    fig12 = px.scatter(
        df_sample,
        x="property_age",
        y="sale_price",
        color="borough",
        title="Property Age vs Price",
        labels={'property_age': 'Age (years)', 'sale_price': 'Price ($)', 'borough': 'Borough'},
        opacity=0.6
    )
    st.plotly_chart(fig12, use_container_width=True)

    age_corr = filtered_df['property_age'].corr(filtered_df['sale_price'])
    st.info(f"**Correlation between age and price:** {age_corr:.3f}")

    # Units vs price
    st.subheader("Number of Units vs Sale Price")

    df_units = filtered_df[filtered_df['total_units'] <= 50]
    df_sample_units = df_units.sample(n=min(3000, len(df_units)), random_state=42)

    fig13 = px.scatter(
        df_sample_units,
        x="total_units",
        y="sale_price",
        color="price_per_sqft",
        title="Total Units vs Price",
        labels={'total_units': 'Number of Units', 'sale_price': 'Price ($)', 
                'price_per_sqft': 'Price/Sqft ($)'},
        opacity=0.6,
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig13, use_container_width=True)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")

    numeric_cols = ['sale_price', 'gross_square_feet', 'land_square_feet', 
                    'total_units', 'property_age', 'price_per_sqft']
    corr_matrix = filtered_df[numeric_cols].corr()

    fig14 = px.imshow(
        corr_matrix,
        text_auto='.2f',
        title="Correlation Matrix of Key Variables",
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    st.plotly_chart(fig14, use_container_width=True)

    st.markdown("""
    **Interpretation:**
    - Values close to 1: Strong positive correlation
    - Values close to -1: Strong negative correlation
    - Values close to 0: Weak or no correlation
    """)

# =============================
# Footer
# =============================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>NYC Real Estate Analysis Dashboard</strong></p>
    <p>Data source: NYC Department of Finance Rolling Sales Data</p>
</div>
""", unsafe_allow_html=True)
