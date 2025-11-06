import pandas as pd
path = r"C:\Users\ankit\PycharmProjects\Assignment1\DataCoSupplyChainDataset.csv"

try:
    # Try most common Windows encodings
    df = pd.read_csv(
        path,
        encoding="latin-1",      # handles non-UTF8 characters
        low_memory=False,        # safer type detection for large files
        on_bad_lines="warn"      # skip or warn on bad rows
    )
    print("✅ File loaded successfully!")
    print("Shape:", df.shape)
    print("First 10 columns:", list(df.columns[:10]))
except Exception as e:
    print("❌ Error loading file:", e)

import pandas as pd
# Count missing values per column
missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df)) * 100

missing_summary = pd.DataFrame({
    'Missing Values': missing,
    'Percentage': missing_pct
})

print(missing_summary.head(20))

# Drop useless or heavily missing columns
df = df.drop(columns=['Product Description', 'Order Zipcode'])

# Fill tiny missing values
df['Customer Lname'] = df['Customer Lname'].fillna('Unknown')
df['Customer Zipcode'] = df['Customer Zipcode'].fillna('00000')

print("✅ Cleaned dataframe shape:", df.shape)

import numpy as np

# Choose numeric columns
num_cols = df.select_dtypes(include=np.number).columns

outlier_summary = {}

for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    outlier_summary[col] = len(outliers)

outlier_df = pd.DataFrame.from_dict(outlier_summary, orient='index', columns=['Outlier Count'])
outlier_df.sort_values(by='Outlier Count', ascending=False).head(10)
print(outlier_df.sort_values(by='Outlier Count', ascending=False).head(10))

# ===============================================================
# Supply Chain Analytics Dashboard — Multi-Page Streamlit App
# ===============================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ---------- Page configuration ----------
st.set_page_config(
    page_title="Supply Chain Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Sidebar Navigation ----------
st.sidebar.title("📊 Navigation Menu")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📈 Dashboard",
        "🧾 Data Overview",
        "📦 Product Analysis",
        "🧍 Customer Segmentation",
        "🏭 Supplier Analysis",
        "🚚 Logistics Analysis",
        "🚛 Shipping Mode Impact (RQ4)",
        "🧠 Diagnostic Analytics",
    ]
)

# ---------- Data Loading ----------
@st.cache_data
def load_data():
    path = r"C:\Users\ankit\PycharmProjects\Assignment1\DataCoSupplyChainDataset.csv"
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    return df

df = load_data()

# ---------- HOME PAGE ----------
if page == "🏠 Home":
    st.title("🏠 Welcome to the Supply Chain Analytics Dashboard")
    st.markdown("""
    ### 👋 Hello!
    This interactive dashboard helps you explore, analyze, and visualize retail supply chain data.

    **How to use this app:**
    - Use the **Sidebar Menu** to navigate between analysis sections.
    - Each page covers part of the case study questions:
      - **Dashboard:** Overall KPIs and visual summary  
      - **Data Overview:** Inspect dataset, missing values, correlations  
      - **Product Analysis (RQ1):** Identify top revenue-generating products  
      - **Customer Segmentation (RQ4):** Explore customer groups & segments  
      - **Supplier Analysis (RQ2):** Assess supplier lead time & quality  
      - **Logistics Analysis (RQ3):** Evaluate delivery and route efficiency  
      - **Diagnostic Analytics:** Deeper insights & correlation discovery
    """)

    st.info("Tip: You can upload a different dataset in the sidebar below.")

    uploaded = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("✅ New dataset loaded successfully!")

# ---------- DASHBOARD PAGE ----------
elif page == "📈 Dashboard":
    st.title("📈 Key KPIs and Visual Summary")

    # Example KPIs
    total_revenue = df['Sales per customer'].sum()
    total_profit = df['Benefit per order'].sum()
    late_risk = df['Late_delivery_risk'].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"${total_revenue:,.0f}")
    c2.metric("Total Profit", f"${total_profit:,.0f}")
    c3.metric("Avg. Late Delivery Risk", f"{late_risk:.2%}")

    # Example chart — revenue by category
    st.subheader("Revenue by Product Category")
    cat_rev = df.groupby('Category Name')['Sales per customer'].sum().reset_index()
    fig = px.bar(cat_rev, x='Category Name', y='Sales per customer', title="Revenue by Category")
    st.plotly_chart(fig, width='stretch')

# ---------- DATA OVERVIEW PAGE ----------
elif page == "🧾 Data Overview":
    st.title("🧾 Data Overview & Quality Checks")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    st.subheader("Basic Statistics")
    st.dataframe(df.describe())

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    st.bar_chart(missing)

    st.subheader("Correlations (Numerical Columns)")
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
    st.plotly_chart(fig, width='stretch')

# ---------- PRODUCT ANALYSIS PAGE (RQ1) ----------
elif page == "📦 Product Analysis":
    st.title("📦 Product Analysis — Revenue & Profit Performance (RQ1)")

    # ----- Revenue & Profit by Product Category -----
    st.subheader("1️⃣ Revenue & Profit by Product Category")
    category_perf = df.groupby("Category Name")[["Sales per customer", "Benefit per order"]].sum().reset_index()
    category_perf = category_perf.sort_values(by="Sales per customer", ascending=False)

    fig1 = px.bar(category_perf, x="Category Name", y="Sales per customer", title="Total Revenue by Category")
    st.plotly_chart(fig1, width='stretch')

    fig2 = px.bar(category_perf, x="Category Name", y="Benefit per order", title="Total Profit by Category")
    st.plotly_chart(fig2, width='stretch')

    # ----- Revenue by Region (Country) -----
    # ----- Revenue by Region (Country) -----
    st.subheader("2️⃣ Revenue by Customer Country")

    # Clean country names (remove spaces, unify US naming)
    df['Customer Country'] = df['Customer Country'].str.strip()
    df['Customer Country'] = df['Customer Country'].replace({
        "United States of America": "United States",
        "US": "United States",
        "U.S.A.": "United States",
        "Brasil": "Brazil"
    })

    country_rev = df.groupby("Customer Country")["Sales per customer"].sum().reset_index()

    fig3 = px.choropleth(
        country_rev,
        locations="Customer Country",
        locationmode="country names",
        color="Sales per customer",
        color_continuous_scale="Blues",
        title="Revenue by Country"
    )

    fig3.update_geos(showcountries=True, showcoastlines=True, projection_type="natural earth")
    st.plotly_chart(fig3, width='stretch')

    # ----- Revenue & Profit by Customer Segment -----
    st.subheader("3️⃣ Revenue & Profit by Customer Segment")
    segment_perf = df.groupby("Customer Segment")[["Sales per customer", "Benefit per order"]].sum().reset_index()

    fig4 = px.bar(segment_perf, x="Customer Segment", y="Sales per customer", title="Revenue by Customer Segment")
    st.plotly_chart(fig4, width='stretch')

    fig5 = px.bar(segment_perf, x="Customer Segment", y="Benefit per order", title="Profit by Customer Segment")
    st.plotly_chart(fig5, width='stretch')

    st.markdown("""
    **Interpretation Guide:**
    - Categories with the highest bars in **Revenue** and **Profit** are your best product lines.
    - Countries with darker color in the map contribute more to total sales.
    - Customer segments with higher revenue/profit are the most valuable market groups.
    """)

# ---------- CUSTOMER SEGMENTATION PAGE (RQ4) ----------
elif page == "🧍 Customer Segmentation":
    st.title("🧍 Customer Segmentation (RQ4)")
    st.markdown("Identify customer groups based on region, segment, or purchasing behavior.")

    seg = df.groupby('Customer Segment')['Sales per customer'].agg(['count','sum','mean']).reset_index()
    st.dataframe(seg)

    fig = px.bar(seg, x='Customer Segment', y='sum', title='Total Sales by Customer Segment')
    st.plotly_chart(fig, width='stretch')

# ---------- SUPPLIER ANALYSIS PAGE (RQ2) ----------
elif page == "🏭 Supplier Analysis":
    st.title("🏭 Cancellation & Quality Risk Analysis (RQ2)")
    st.markdown("Identify where cancellations are most common across departments and regions.")

    # Mark cancelled orders
    df['Cancelled'] = df['Order Status'].str.contains("Canceled|Cancelled|Not Delivered|Refunded", case=False, na=False)

    # 1️⃣ Cancellation Rate by Department
    st.subheader("1️⃣ Cancellation Rate by Department")
    dept_cancel = df.groupby("Department Name")["Cancelled"].mean().reset_index()
    fig1 = px.bar(dept_cancel, x="Department Name", y="Cancelled",
                  title="Cancellation Rate by Department")
    st.plotly_chart(fig1, width='stretch')

    # 2️⃣ Cancellation Rate by Product Category
    st.subheader("2️⃣ Cancellation Rate by Product Category")
    cat_cancel = df.groupby("Category Name")["Cancelled"].mean().reset_index()
    fig2 = px.bar(cat_cancel, x="Category Name", y="Cancelled",
                  title="Cancellation Rate by Product Category")
    st.plotly_chart(fig2, width='stretch')

    # 3️⃣ Cancellation Rate by Country
    st.subheader("3️⃣ Cancellation Rate by Customer Country")
    country_cancel = df.groupby("Customer Country")["Cancelled"].mean().reset_index()
    fig3 = px.bar(country_cancel, x="Customer Country", y="Cancelled",
                  title="Cancellation Rate by Country")
    st.plotly_chart(fig3, width='stretch')

    # 4️⃣ Cancellation Rate by Customer Segment
    st.subheader("4️⃣ Cancellation Rate by Customer Segment")
    segment_cancel = df.groupby("Customer Segment")["Cancelled"].mean().reset_index()
    fig4 = px.bar(segment_cancel, x="Customer Segment", y="Cancelled",
                  title="Cancellation Rate by Customer Segment")
    st.plotly_chart(fig4, width='stretch')

    st.markdown("""
    **Interpretation Guide:**
    - Higher bars represent departments, categories, or countries with more canceled orders.
    - These areas may require improvements in product availability, delivery reliability, or customer support.
    """)

# ---------- LOGISTICS ANALYSIS PAGE (RQ3) ----------
elif page == "🚚 Logistics Analysis":
    st.title("🚚 Logistics & Delivery Analysis (RQ3)")
    st.markdown("Evaluate delivery delays, shipment performance, and risk factors influencing late deliveries.")

    # 1) Late delivery risk by delivery status
    st.subheader("1️⃣ Late Delivery Risk by Delivery Status")
    delay = df.groupby('Delivery Status')['Late_delivery_risk'].mean().reset_index()
    fig1 = px.bar(delay, x='Delivery Status', y='Late_delivery_risk', title='Late Delivery Risk by Delivery Status')
    st.plotly_chart(fig1, width='stretch')

    # 2) Impact of Actual Shipping Time on Late Delivery Risk
    st.subheader("2️⃣ Impact of Shipping Duration on Late Delivery Risk")
    if 'Days for shipping (real)' in df.columns:
        fig2 = px.scatter(
            df.sample(2000),
            x='Days for shipping (real)',
            y='Late_delivery_risk',
            trendline="ols",
            title="Longer Shipping Times Correlate with Higher Late Delivery Risk"
        )
        st.plotly_chart(fig2, width='stretch')

    # 3) Late Delivery Risk by Product Category
    st.subheader("3️⃣ Late Delivery Risk by Product Category")
    cat_risk = df.groupby('Category Name')['Late_delivery_risk'].mean().reset_index()
    fig3 = px.bar(cat_risk, x='Category Name', y='Late_delivery_risk', title='Late Delivery Risk by Product Category')
    st.plotly_chart(fig3, width='stretch')

    # 4) Correlation Heatmap of Delivery Related Factors
    st.subheader("4️⃣ Correlation Relationships")
    numeric_cols = ['Days for shipping (real)', 'Days for shipment (scheduled)', 'Late_delivery_risk']
    corr = df[numeric_cols].corr()

    fig4 = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
    st.plotly_chart(fig4, width='stretch')

# ---------- DIAGNOSTIC ANALYTICS PAGE ----------
# ---------- DIAGNOSTIC ANALYTICS PAGE (RQ5) ----------
elif page == "🧠 Diagnostic Analytics":
    st.title("🧠 Predictive Analytics — Which Orders Are Likely to Be Returned? (RQ5)")
    st.markdown("We apply a simple machine learning model to predict the probability of an order being returned or cancelled based on shipping performance and profitability.")

    # Create Returned flag
    df['Returned'] = df['Order Status'].str.contains("Returned|Refunded|Canceled|Cancelled", case=False, na=False)

    # Select model features
    model_df = df[['Sales per customer', 'Benefit per order', 'Late_delivery_risk', 'Days for shipping (real)', 'Returned']].dropna()

    # Prepare features and target
    X = model_df[['Sales per customer', 'Benefit per order', 'Late_delivery_risk', 'Days for shipping (real)']]
    y = model_df['Returned']

    # Split data
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Train the model
    model = LogisticRegression(max_iter=500, class_weight='balanced')
    model.fit(X_train, y_train)

    # Generate predictions & evaluation report
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)

    st.subheader("📊 Model Performance")
    st.text(report)

    # Feature importance
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0]
    }).sort_values(by='Coefficient', ascending=False)

    st.subheader("🔥 What Factors Increase Return Risk?")
    st.bar_chart(importance.set_index('Feature'))

    st.markdown("""
    **Interpretation Guide:**
    - Higher coefficient = higher return risk.
    - Late delivery and long shipping times increase return likelihood.
    - Higher profit per order reduces return likelihood.
    """)

# ---------- SHIPPING MODE IMPACT PAGE (RQ4) ----------
elif page == "🚛 Shipping Mode Impact (RQ4)":
    st.title("🚛 Shipping Mode Impact on Profit, Satisfaction & Delivery Timeliness (RQ4)")
    st.markdown("Analyze how shipping mode choices influence profit levels, customer satisfaction (via delivery timeliness), and operational speed.")

    # Create cancellation indicator
    df['Cancelled'] = df['Order Status'].str.contains("Canceled|Cancelled|Not Delivered|Refunded", case=False, na=False)

    # 1) Profit by Shipping Mode
    st.subheader("1️⃣ Profit by Shipping Mode")
    mode_profit = df.groupby("Shipping Mode")["Benefit per order"].mean().reset_index()
    fig1 = px.bar(mode_profit, x="Shipping Mode", y="Benefit per order",
                  title="Average Profit per Order by Shipping Mode")
    st.plotly_chart(fig1, width="stretch")

    # 2) Late Delivery Risk by Shipping Mode (proxy for satisfaction)
    st.subheader("2️⃣ Customer Satisfaction Proxy (Late Delivery Risk)")
    mode_risk = df.groupby("Shipping Mode")["Late_delivery_risk"].mean().reset_index()
    fig2 = px.bar(mode_risk, x="Shipping Mode", y="Late_delivery_risk",
                  title="Late Delivery Risk by Shipping Mode")
    st.plotly_chart(fig2, width="stretch")

    # 3) Average Shipping Time by Shipping Mode
    st.subheader("3️⃣ Average Shipping Time by Shipping Mode")
    if 'Days for shipping (real)' in df.columns:
        mode_time = df.groupby("Shipping Mode")["Days for shipping (real)"].mean().reset_index()
        fig3 = px.bar(mode_time, x="Shipping Mode", y="Days for shipping (real)",
                      title="Average Shipping Time (Real) by Shipping Mode")
        st.plotly_chart(fig3, width="stretch")

    # 4) Does Longer Shipping Reduce Profit?
    st.subheader("4️⃣ Does Longer Shipping Reduce Profit?")
    fig4 = px.scatter(
        df.sample(2000),
        x="Days for shipping (real)",
        y="Benefit per order",
        trendline="ols",
        title="Shipping Time vs Order Profit"
    )
    st.plotly_chart(fig4, width="stretch")

    st.markdown("""
    **Interpretation Guide:**
    - Higher average profit indicates a financially favorable shipping mode.
    - Lower late delivery risk means higher customer satisfaction.
    - Shorter shipping time indicates better delivery efficiency.
    """)




