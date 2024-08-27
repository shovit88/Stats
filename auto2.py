import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import shapiro, pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols

def convert_data_types(df):
    # Display info about the dataframe
    st.write(f"Number of columns: {len(df.columns)}")
    st.write(f"Columns: {', '.join(df.columns)}")
    
    # Display the first few rows of the dataframe
    st.write("First few rows of the dataframe:")
    st.write(df.head())

    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def generate_boxplot(df, input_var):
    st.subheader(f"Box Plot: {input_var}")
    fig, ax = plt.subplots(figsize=(16, 5))
    sns.boxplot(x=input_var, data=df, ax=ax)
    ax.set_title(f'Box Plot: {input_var}')
    st.pyplot(fig)


def generate_control_chart(df, variable):
    st.subheader(f"Control Chart for {variable}")
    
    data = df[variable]
    mean = data.mean()
    std_dev = data.std()
    
    ucl = mean + 3 * std_dev  # Upper Control Limit
    lcl = mean - 3 * std_dev  # Lower Control Limit
    
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(data.index, data, marker='o', linestyle='-', color='blue')
    ax.axhline(y=mean, color='green', linestyle='-', label='Mean')
    ax.axhline(y=ucl, color='red', linestyle='--', label='UCL')
    ax.axhline(y=lcl, color='red', linestyle='--', label='LCL')
    
    ax.fill_between(data.index, ucl, lcl, alpha=0.1, color='red')
    
    ax.set_title(f'Control Chart for {variable}')
    ax.set_xlabel('Sample Number')
    ax.set_ylabel('Value')
    ax.legend()
    
    st.pyplot(fig)

def perform_normality_test(df, variable):
    st.subheader(f"Normality Test for {variable}")
    df = df.dropna()
    data = df[variable]
    stat, p = shapiro(data)
    
    st.write(f"Shapiro-Wilk test statistic: {stat:.4f}")
    st.write(f"p-value: {p:.4f}")
    
    if p > 0.05:
        st.write("The data appears to be normally distributed (fail to reject H0)")
    else:
        st.write("The data does not appear to be normally distributed (reject H0)")
    
    # Q-Q plot
    fig, ax = plt.subplots(figsize=(10, 6))
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(f"Q-Q plot for {variable}")
    st.pyplot(fig)



def perform_hypothesis_testing(df, var1, var2):
    df = df.dropna()
    st.subheader(f"Hypothesis Testing: {var1} vs {var2}")
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(df[var1], df[var2])
    
    st.write(f"t-statistic: {t_stat:.4f}")
    st.write(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        st.write("Reject the null hypothesis. There is a significant difference between the two variables.")
    else:
        st.write("Fail to reject the null hypothesis. There is no significant difference between the two variables.")

def perform_anova(df, group_col, value_col):
    groups = df.groupby(group_col)[value_col].apply(list)
    f_val, p_val = stats.f_oneway(*groups)
    return f_val, p_val

def perform_correlation_analysis(df):
    st.subheader("Correlation Analysis")
    correlation_method = st.selectbox("Select correlation method", ["Pearson", "Spearman"])
    
    if correlation_method == "Pearson":
        corr_matrix = df.corr(method='pearson')
    else:
        corr_matrix = df.corr(method='spearman')
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title(f'{correlation_method} Correlation Heatmap')
    st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, classification_report
from scipy.stats import shapiro, pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def perform_simple_linear_regression(df, x_var, y_var):
    st.subheader(f"Simple Linear Regression: {y_var} vs {x_var}")
    df = df.dropna()
    X = df[[x_var]]
    y = df[y_var]
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    st.write(f"Coefficient: {model.coef_[0]:.4f}")
    st.write(f"Intercept: {model.intercept_:.4f}")
    st.write(f"R-squared: {r2:.4f}")
    st.write(f"Mean Squared Error: {mse:.4f}")
    
    # Scatter plot with regression line
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x=x_var, y=y_var, data=df, ax=ax)
    ax.set_title(f"Simple Linear Regression: {y_var} vs {x_var}")
    st.pyplot(fig)
    
    # Add error and MSE columns
    df['Predicted'] = y_pred
    df['Error'] = y - y_pred
    df['MSE'] = (df['Error'] ** 2)
    
    return df

def perform_multiple_linear_regression(df, y_var, x_vars):
    st.subheader(f"Multiple Linear Regression: {y_var} vs {', '.join(x_vars)}")
    df = df.dropna()
    
    X = df[x_vars]
    y = df[y_var]
    
    try:
        # Add constant term to the features
        #X = sm.add_constant(X)
        
        # Fit the model
        model = sm.OLS(y, X).fit()
        
        # Display the summary
        st.write(model.summary())
        
        # Multicollinearity analysis (VIF)
        st.subheader("Multicollinearity Analysis (VIF)")
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        st.write(vif_data)
        
        # Residual plots
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        
        # Residuals vs Fitted
        sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, 
                      scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 1}, ax=axes[0, 0])
        axes[0, 0].set_title('Residuals vs Fitted')
        axes[0, 0].set_xlabel('Fitted values')
        axes[0, 0].set_ylabel('Residuals')
        
        # Q-Q plot
        sm.qqplot(model.resid, fit=True, line="45", ax=axes[0, 1])
        axes[0, 1].set_title('Q-Q plot')
        
        # Scale-Location
        standardized_resid = model.get_influence().resid_studentized_internal
        sns.regplot(x=model.fittedvalues, y=np.sqrt(np.abs(standardized_resid)), 
                    lowess=True, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 1}, ax=axes[1, 0])
        axes[1, 0].set_title('Scale-Location')
        axes[1, 0].set_xlabel('Fitted values')
        axes[1, 0].set_ylabel('$\sqrt{|Standardized Residuals|}$')
        
        # Residuals vs Leverage
        sm.graphics.influence_plot(model, ax=axes[1, 1], criterion="cooks")
        axes[1, 1].set_title('Residuals vs Leverage')
        
        st.pyplot(fig)
        
        # Add error and MSE columns
        df['Predicted'] = model.predict(X)
        df['Error'] = y - df['Predicted']
        df['MSE'] = (df['Error'] ** 2)
        
        return df
        
    except Exception as e:
        st.error(f"An error occurred during multiple linear regression: {str(e)}")
        st.write("Possible reasons for failure:")
        st.write("1. Multicollinearity among independent variables")
        st.write("2. Non-numeric data in selected columns")
        st.write("3. Insufficient data points")
        st.write("4. Perfect collinearity (one variable is a perfect linear combination of others)")
        return None

def display_highest_errors(df, n=10):
    st.subheader(f"Top {n} Highest Errors")
    df_sorted = df.sort_values('MSE', ascending=False).head(n)
    st.write(df_sorted)



def regression_analysis(df, regression_type):
    if regression_type == "Simple Linear Regression":
        x_var = st.sidebar.selectbox('Select Independent Variable (X)', df.columns)
        y_var = st.sidebar.selectbox('Select Dependent Variable (Y)', [col for col in df.columns if col != x_var])
        df = perform_simple_linear_regression(df, x_var, y_var)
    elif regression_type == "Multiple Linear Regression":
        y_var = st.sidebar.selectbox('Select Dependent Variable (Y)', df.columns)
        x_vars = st.sidebar.multiselect('Select Independent Variables (X)', [col for col in df.columns if col != y_var])
        if len(x_vars) > 1:
            df = perform_multiple_linear_regression(df, y_var, x_vars)
        else:
            st.warning("Please select at least two independent variables for multiple linear regression.")
            return None
    
    if df is not None:
        display_highest_errors(df)
    
    return df

def main():
    st.set_page_config(page_title="Statistical Analysis Tool")
    st.sidebar.title("Select task")
    selection = st.sidebar.radio("Go to", ("Box Plot", "Control Chart", "Normality Test", "Correlation Analysis", 
                                           "Simple Linear Regression", "Multiple Linear Regression", 
                                           "Hypothesis Testing", "ANOVA Test"))

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df = convert_data_types(df)

            if selection == "Box Plot":
                input_var = st.sidebar.selectbox('Select Output Variable', df.columns)
                generate_boxplot(df, input_var)
            
            elif selection == "Control Chart":
                variable_for_control = st.sidebar.selectbox('Select Variable for Control Chart', df.columns)
                generate_control_chart(df, variable_for_control)
            
            elif selection == "Normality Test":
                variable_for_normality = st.sidebar.selectbox('Select Variable for Normality Test', df.columns)
                perform_normality_test(df, variable_for_normality)
            
            elif selection == "Correlation Analysis":
                perform_correlation_analysis(df)
            
            elif selection in ["Simple Linear Regression", "Multiple Linear Regression"]:
                df = regression_analysis(df, selection)
            
            
            elif selection == "Hypothesis Testing":
                var1 = st.sidebar.selectbox('Select First Variable', df.columns)
                var2 = st.sidebar.selectbox('Select Second Variable', [col for col in df.columns if col != var1])
                perform_hypothesis_testing(df, var1, var2)
            
            elif selection == "ANOVA Test":
                columns = df.columns.tolist()
                group_col = st.selectbox("Select the column for grouping (independent variable):", columns)
                value_col = st.selectbox("Select the column for values (dependent variable):", columns)

                df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
                df = df.dropna(subset=[value_col])

                if st.button("Perform ANOVA"):
                    try:
                        f_val, p_val = perform_anova(df, group_col, value_col)
                        if pd.isna(f_val) or pd.isna(p_val):
                            st.error("ANOVA test resulted in NaN values. Please check your data for issues such as insufficient data points or identical values in groups.")
                        else:
                            st.write(f"ANOVA Results: F-value = {f_val}, P-value = {p_val}")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
            st.write("Possible reasons for failure:")
            st.write("1. The file is not a valid CSV")
            st.write("2. The file is empty or corrupted")
            st.write("3. The file contains non-numeric data in columns expected to be numeric")

if __name__ == "__main__":
    main()