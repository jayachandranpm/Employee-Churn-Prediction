import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from scipy.stats import ttest_ind, chi2_contingency
from statsmodels.stats.proportion import proportions_ztest

# Load the saved XGBoost model
model_filename = 'xgb_model.joblib'
xgb_model = joblib.load(model_filename)

# Load the dataset (replace 'HR_Analytics.csv' with your actual dataset)
file_path = 'HR_Analytics.csv'
data = pd.read_csv(file_path)

# Streamlit app
def main():

      # Add a sidebar with menu options
    menu = ["Prediction", "Exploratory Data Analysis (EDA)", "Statistical Analysis"]
    choice = st.sidebar.selectbox("Select Menu", menu)

    if choice == "Prediction":
        st.header("Employee Churn Prediction")
        st.title("Employee Churn Prediction Dashboard")

        # Add input components for user to input data
        satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
        last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.5)
        number_project = st.slider("Number of Projects", 2, 7, 4)
        average_montly_hours = st.slider("Average Monthly Hours", 80, 300, 160)
        time_spend_company = st.slider("Time Spent in Company (years)", 1, 10, 3)
        work_accident = st.checkbox("Work Accident", value=False)
        promotion_last_5years = st.checkbox("Promotion in Last 5 Years", value=False)
        department = st.selectbox("Department", data['Department'].unique())
        salary = st.selectbox("Salary", data['salary'].unique())

        # Button to make predictions
        if st.button("Predict"):
            # Create a DataFrame with the user inputs
            input_data = pd.DataFrame({
                'satisfaction_level': [satisfaction_level],
                'last_evaluation': [last_evaluation],
                'number_project': [number_project],
                'average_montly_hours': [average_montly_hours],
                'time_spend_company': [time_spend_company],
                'Work_accident': [work_accident],
                'promotion_last_5years': [promotion_last_5years],
                'Department_RandD': [1 if department == 'RandD' else 0],
                'Department_accounting': [1 if department == 'accounting' else 0],
                'Department_hr': [1 if department == 'hr' else 0],
                'Department_management': [1 if department == 'management' else 0],
                'Department_marketing': [1 if department == 'marketing' else 0],
                'Department_product_mng': [1 if department == 'product_mng' else 0],
                'Department_sales': [1 if department == 'sales' else 0],
                'Department_support': [1 if department == 'support' else 0],
                'Department_technical': [1 if department == 'technical' else 0],
                'salary_low': [1 if salary == 'low' else 0],
                'salary_medium': [1 if salary == 'medium' else 0]
            })

            # Make predictions using the loaded XGBoost model
            prediction = xgb_model.predict(input_data)

            # Display the prediction
            if prediction[0] == 1:
                st.error("Employee is likely to leave.")
            else:
                st.success("Employee is likely to stay.")




    elif choice == "Exploratory Data Analysis (EDA)":
        st.header("Exploratory Data Analysis")

        # Distribution of Satisfaction Level
        st.subheader("Distribution of Satisfaction Level")
        sns.histplot(data['satisfaction_level'], bins=20, kde=True)
        st.pyplot()

        # Employee Churn by Department
        st.subheader("Employee Churn by Department")
        churn_by_department = data.groupby('Department')['left'].mean().sort_values(ascending=False)
        st.bar_chart(churn_by_department)

        # Word Cloud of Departments
        st.subheader("Word Cloud of Departments")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(data['Department'].value_counts())
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot()

    elif choice == "Statistical Analysis":
        st.header("Statistical Analysis")

        # Compare average satisfaction level between employees who left and those who stayed
        st.subheader("Comparison of Satisfaction Level")
        left_employees = data[data['left'] == 1]['satisfaction_level']
        stayed_employees = data[data['left'] == 0]['satisfaction_level']
        t_stat, p_value = ttest_ind(left_employees, stayed_employees)
        st.write(f'T-test results: t-statistic={t_stat:.4f}, p-value={p_value:.4f}')

        # Chi-square test for independence between 'left' and 'Department'
        st.subheader("Chi-square Test for Independence")
        contingency_table = pd.crosstab(data['left'], data['Department'])
        chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
        st.write(f'Chi-square Test Results - Independence between "left" and "Department": Chi2={chi2_stat:.4f}, p-value={p_val:.4f}')

        # Proportions z-test for 'Work_accident' and 'left'
        st.subheader("Proportions Z-test")
        work_accident_prop = data['Work_accident'].value_counts(normalize=True).loc[1]
        left_prop = data['left'].value_counts(normalize=True).loc[1]
        z_stat, p_val = proportions_ztest(count=data['left'].sum(), nobs=len(data), value=work_accident_prop)
        st.write(f'Proportions Z-test Results - "Work_accident" and "left": Z-statistic={z_stat:.4f}, p-value={p_val:.4f}')

# Run the Streamlit app
if __name__ == '__main__':
    main()