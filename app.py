import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Load the dataset and trained model
df = pd.read_csv("Credit_Risk_Dataset_with_Loan_Status.csv")
model = joblib.load("model.pkl")

# Streamlit App Title and Sidebar Navigation
st.set_page_config(page_title="RiskLens: Loan Approval Analysis", layout="wide")
st.title("🤖RiskLens: Loan Approval Analysis")
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Choose Page", ["Raw Data", "Summary", "Graphs & Charts", "Loan Approval Predictor"])

# --- Page 1: Raw Data ---
if page == "Raw Data":
    st.title("📄 Raw Dataset View")
    st.dataframe(df)

# --- Page 2: Summary ---
elif page == "Summary":
    st.title("📋 Dataset Summary")
    st.write(df.describe())

# --- Page 3: Graphs & Charts ---
elif page == "Graphs & Charts":
    st.title("📈 Graphical Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Loan Status Distribution")
        st.bar_chart(df['loan_status'].value_counts())

    with col2:
        st.subheader("Monthly Income vs Debt Ratio (Filtered)")

        # Clean and filter data
        df_filtered = df[(df['debt_ratio'] < 1000) & (df['monthly_inc'] < 300000)].copy()

        # Map loan_status to labels safely
        if df_filtered['loan_status'].dropna().isin([0, 1]).all():
            df_filtered['loan_status'] = df_filtered['loan_status'].map({1: 'Approved', 0: 'Rejected'})
        else:
            df_filtered['loan_status'] = 'Unknown'

        if df_filtered.empty:
            st.warning("No data to display after filtering out extreme debt ratios and income.")
        else:
            fig1, ax1 = plt.subplots()
            sns.scatterplot(data=df_filtered, x="monthly_inc", y="debt_ratio", hue="loan_status", ax=ax1)
            ax1.set_title("Debt Ratio vs Monthly Income")
            ax1.set_xlabel("Monthly Income (₹)")
            ax1.set_ylabel("Debt Ratio")
            st.pyplot(fig1)

    st.subheader("Loan Approval Share (Pie Chart)")
    st.plotly_chart(px.pie(df, names='loan_status', title="Approved vs Not Approved"))

# --- Page 4: Loan Approval Predictor ---
elif page == "Loan Approval Predictor":
    st.title(" Loan Approval Predictor (Logistic Regression)")

    st.write("### Enter applicant details:")

    rev_util = st.number_input("Revolving Utilization (0.0 - 1.0)", 0.0, 1.0, 0.1, step=0.01)
    age = st.number_input("Age", 18, 100, 30)
    late_30_59 = st.number_input("Times 30-59 days late", 0, 10, 0)
    debt_ratio = st.number_input("Debt Ratio", 0.0, 10.0, 0.5)
    monthly_inc = st.number_input("Monthly Income", 0.0, 100000.0, 5000.0)
    open_credit = st.number_input("Open Credit Lines", 0, 50, 5)
    late_90 = st.number_input("Times 90+ days late", 0, 10, 0)
    real_estate = st.number_input("Real Estate Loans", 0, 10, 1)
    late_60_89 = st.number_input("Times 60-89 days late", 0, 10, 0)
    dependents = st.number_input("Number of Dependents", 0, 10, 0)

    if st.button("Predict Loan Approval"):
        user_data = pd.DataFrame([[rev_util, age, late_30_59, debt_ratio, monthly_inc,
                                   open_credit, late_90, real_estate, late_60_89, dependents]],
                                 columns=['rev_util', 'age', 'late_30_59', 'debt_ratio', 'monthly_inc',
                                          'open_credit', 'late_90', 'real_estate', 'late_60_89', 'dependents'])

        prediction = model.predict(user_data)[0]
        proba = model.predict_proba(user_data)[0][1]

        # --- Show Result ---
        if prediction == 1:
            st.success(f"✅ Loan Approved! (Confidence: {proba:.2%})")
        else:
            st.error(f"❌ Loan Rejected. (Confidence: {1 - proba:.2%})")

        # --- Gauge Chart ---
        st.subheader("📉 Prediction Confidence")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            title={'text': "Approval Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green" if prediction == 1 else "red"},
                'steps': [
                    {'range': [0, 50], 'color': "pink"},
                    {'range': [50, 100], 'color': "lightgreen"}
                ]
            }
        ))
        st.plotly_chart(gauge)

        # --- Comparison with Approved User Averages ---
        st.subheader("📊 Your Values vs Approved Averages")

        approved_avg = df[df['loan_approved'] == 1][['monthly_inc', 'debt_ratio', 'rev_util']].dropna().mean()

        default_avg = {
            'monthly_inc': 5377.79,
            'debt_ratio': 0.434,
            'rev_util': 0.189
        }

        for col in default_avg:
            if pd.isna(approved_avg[col]):
                approved_avg[col] = default_avg[col]

        user_vals = user_data[['monthly_inc', 'debt_ratio', 'rev_util']].iloc[0]

        metrics = ['monthly_inc', 'debt_ratio', 'rev_util']
        titles = ['Monthly Income (₹)', 'Debt Ratio', 'Revolving Utilization']

        for i, metric in enumerate(metrics):
            comp_df = pd.DataFrame({
                'Category': ['Approved Avg', 'You'],
                'Value': [approved_avg[metric], user_vals[metric]]
            })

            fig, ax = plt.subplots()
            sns.barplot(data=comp_df, x='Category', y='Value', palette='Set2', ax=ax)
            ax.set_title(f"{titles[i]}")
            st.pyplot(fig)
