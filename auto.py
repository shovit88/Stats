import streamlit as st
import pandas as pd
from statsmodels.stats.weightstats import ztest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import shapiro, pearsonr, spearmanr 
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from sklearn.linear_model import LinearRegression, LogisticRegression
from statsmodels.graphics.factorplots import interaction_plot
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.factorplots import interaction_plot
from statsmodels.stats.anova import AnovaRM

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
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create box plot
    sns.boxplot(x=input_var, data=df, ax=ax)
    
    # Calculate statistics
    q0, q1, q2, q3, q4 = df[input_var].quantile([0, 0.25, 0.5, 0.75, 1])
    iqr = q3 - q1
    lower_whisker = max(q0, q1 - 1.5 * iqr)
    upper_whisker = min(q4, q3 + 1.5 * iqr)
    
    # Add vertical lines for quantiles
    ax.axvline(q0, color='r', linestyle='--', alpha=0.7, linewidth=2)
    ax.axvline(q1, color='g', linestyle='--', alpha=0.7, linewidth=2)
    ax.axvline(q2, color='b', linestyle='--', alpha=0.7, linewidth=2)
    ax.axvline(q3, color='g', linestyle='--', alpha=0.7, linewidth=2)
    ax.axvline(q4, color='r', linestyle='--', alpha=0.7, linewidth=2)
    
    # Annotate quantile values
    ax.text(q0, ax.get_ylim()[1] * 0.95, f'q0: {q0:.5f}', ha='center', va='top', color='r', fontsize=12, fontweight='bold')
    ax.text(q1, ax.get_ylim()[1] * 0.95, f'q1: {q1:.5f}', ha='center', va='top', color='g', fontsize=12, fontweight='bold')
    ax.text(q2, ax.get_ylim()[1] * 0.95, f'q2: {q2:.5f}', ha='center', va='top', color='b', fontsize=12, fontweight='bold')
    ax.text(q3, ax.get_ylim()[1] * 0.95, f'q3: {q3:.5f}', ha='center', va='top', color='g', fontsize=12, fontweight='bold')
    ax.text(q4, ax.get_ylim()[1] * 0.95, f'q4: {q4:.5f}', ha='center', va='top', color='r', fontsize=12, fontweight='bold')
    
    ax.set_title(f'Box Plot: {input_var}', fontsize=16, fontweight='bold')
    st.pyplot(fig)

    st.write(f"Interquartile Range (IQR): {iqr:.5f}")
    st.write(f"q0: {q0:.5f}")
    st.write(f"q1: {q1:.5f}")
    st.write(f"q2: {q2:.5f}")
    st.write(f"q3: {q3:.5f}")
    st.write(f"q4: {q4:.5f}")
    st.write(f"Number of Outliers: {sum((df[input_var] < lower_whisker) | (df[input_var] > upper_whisker))}")

def generate_imr_chart(df, variable):
    # Individual Moving Range (IMR) Chart
    data = df[variable]
    individual = data
    moving_range = abs(individual.diff())
    
    mean = individual.mean()
    mr_mean = moving_range.mean()
    
    ucl_i = mean + 2.66 * mr_mean
    lcl_i = mean - 2.66 * mr_mean
    ucl_mr = 3.267 * mr_mean
    lcl_mr = 0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Individual chart
    ax1.plot(individual.index, individual, marker='o', linestyle='-', color='blue')
    ax1.axhline(y=mean, color='green', linestyle='-', label='Mean')
    ax1.axhline(y=ucl_i, color='red', linestyle='--', label='UCL')
    ax1.axhline(y=lcl_i, color='red', linestyle='--', label='LCL')
    ax1.set_title(f'Individual Chart for {variable}')
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('Value')
    ax1.legend()
    
    # Moving Range chart
    ax2.plot(moving_range.index, moving_range, marker='o', linestyle='-', color='blue')
    ax2.axhline(y=mr_mean, color='green', linestyle='-', label='MR Mean')
    ax2.axhline(y=ucl_mr, color='red', linestyle='--', label='UCL')
    ax2.axhline(y=lcl_mr, color='red', linestyle='--', label='LCL')
    ax2.set_title(f'Moving Range Chart for {variable}')
    ax2.set_xlabel('Sample Number')
    ax2.set_ylabel('Moving Range')
    ax2.legend()
    
    st.pyplot(fig)
    
    # Process Capability Analysis
    st.subheader("Process Capability Analysis")
    usl = st.number_input("Enter Upper Specification Limit (USL)", value=float(data.mean() + 3*data.std()))
    lsl = st.number_input("Enter Lower Specification Limit (LSL)", value=float(data.mean() - 3*data.std()))
    
    cp = (usl - lsl) / (6 * data.std())
    cpu = (usl - data.mean()) / (3 * data.std())
    cpl = (data.mean() - lsl) / (3 * data.std())
    cpk = min(cpu, cpl)
    
    st.write(f"Process Capability (Cp): {cp:.3f}")
    st.write(f"Process Capability Index (Cpk): {cpk:.3f}")
    st.write(f"Upper Process Capability (CPU): {cpu:.3f}")
    st.write(f"Lower Process Capability (CPL): {cpl:.3f}")

def generate_xbar_chart(df):
    # Calculate X-bar (mean) and R (range) for each sample
    x_bar = df.mean(axis=0)
    r = df.apply(np.ptp, axis=0)

    # Constants for control limits, for n = 5
    d2 = 2.326  # for calculating UCL and LCL for X-bar chart
    D3 = 0  # for calculating LCL for R chart
    D4 = 2.114  # for calculating UCL for R chart

    # Calculate control limits for R chart
    r_mean = r.mean()
    ucl_r = D4 * r_mean
    lcl_r = D3 * r_mean

    # Calculate control limits for X-bar chart
    x_bar_mean = x_bar.mean()
    ucl_x_bar = x_bar_mean + 3 * r_mean / (d2 * np.sqrt(len(df.index)))
    lcl_x_bar = x_bar_mean - 3 * r_mean / (d2 * np.sqrt(len(df.index)))

    # Streamlit app
    st.title('X-bar and R Control Charts')

    # Plot R chart
    st.subheader('R Chart')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(y=r.mean(), color='g', linestyle='-', label='Overall Range Mean')
    ax.axhline(y=ucl_r, color='r', linestyle='--', label='UCL')
    ax.axhline(y=lcl_r, color='r', linestyle='--', label='LCL')
    ax.plot(r, marker='o')
    ax.set_title('R Chart', fontdict={'fontsize': 16, 'fontweight': 'bold'})
    ax.set_xlabel('Samples', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_ylabel('Range', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    ax.text(9.5, ucl_r + 0.5, f'UCL={round(ucl_r, 3)}')
    ax.text(9.5, lcl_r - 0.5, f'LCL={round(lcl_r, 3)}')
    ax.text(9.5, r_mean + 0.5, f'CL={round(r_mean, 3)}')
    ax.legend()
    st.pyplot(fig)

    # Plot X-bar chart
    st.subheader('X-bar Chart')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(y=x_bar_mean, color='g', linestyle='-', label='Overall Mean')
    ax.axhline(y=ucl_x_bar, color='r', linestyle='--', label='UCL')
    ax.axhline(y=lcl_x_bar, color='r', linestyle='--', label='LCL')
    ax.plot(x_bar, marker='o')
    ax.set_title('X-bar Chart', fontdict={'fontsize': 16, 'fontweight': 'bold'})
    ax.set_xlabel('Samples', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_ylabel('Mean', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    ax.text(9.5, ucl_x_bar + 0.5, f'UCL={round(ucl_x_bar, 3)}')
    ax.text(9.5, lcl_x_bar - 0.5, f'LCL={round(lcl_x_bar, 3)}')
    ax.text(9.5, x_bar_mean, f'CL={round(x_bar_mean, 3)}')
    ax.legend()
    st.pyplot(fig)


def generate_p_chart(df, variable):
    # P Chart for attribute data
    st.write("Note: P Chart is for attribute (binary) data. Ensure your data is in the correct format.")
    
    sample_size = st.number_input("Enter sample size", min_value=1, value=100)
    data = df[variable]
    p = data.mean()
    n = len(data)
    
    ucl = p + 3 * np.sqrt(p * (1 - p) / sample_size)
    lcl = max(0, p - 3 * np.sqrt(p * (1 - p) / sample_size))
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ax.plot(data.index, data, marker='o', linestyle='-', color='blue')
    ax.axhline(y=p, color='green', linestyle='-', label='Mean')
    ax.axhline(y=ucl, color='red', linestyle='--', label='UCL')
    ax.axhline(y=lcl, color='red', linestyle='--', label='LCL')
    ax.set_title(f'P Chart for {variable}')
    ax.set_xlabel('Sample Number')
    ax.set_ylabel('Proportion')
    ax.legend()
    
    st.pyplot(fig)
    
    st.write(f"Process mean (p): {p:.3f}")
    st.write(f"Upper Control Limit (UCL): {ucl:.3f}")
    st.write(f"Lower Control Limit (LCL): {lcl:.3f}")

def perform_normality_test(df, variable):
    st.subheader(f"Normality Test for {variable}")
    df = df.dropna()
    data = df[variable]
    
    # Shapiro-Wilk test
    stat, p = shapiro(data)
    
    st.write(f"Shapiro-Wilk test statistic: {stat:.4f}")
    st.write(f"p-value: {p:.4f}")
    
    if p > 0.05:
        st.write("The data appears to be normally distributed (fail to reject H0)")
    else:
        st.write("The data does not appear to be normally distributed (reject H0)")
    
    # Q-Q plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Q-Q plot
    stats.probplot(data, dist="norm", plot=ax1)
    ax1.set_title(f"Q-Q plot for {variable}")
    
    # Histogram with normal distribution overlay
    sns.histplot(data=data, kde=True, ax=ax2)
    ax2.set_title(f"Histogram with Normal Distribution for {variable}")
    
    # Add a normal distribution curve
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, data.mean(), data.std())
    ax2.plot(x, p, 'k', linewidth=2)
    
    st.pyplot(fig)
    
    # Additional normality tests
    st.subheader("Additional Normality Tests")
    
    # Anderson-Darling test
    result = stats.anderson(data)
    st.write("Anderson-Darling test:")
    st.write(f"Statistic: {result.statistic:.4f}")
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < cv:
            st.write(f"At {sl}% level: The data is normal (fail to reject H0)")
        else:
            st.write(f"At {sl}% level: The data is not normal (reject H0)")
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.kstest(data, 'norm')
    st.write("\nKolmogorov-Smirnov test:")
    st.write(f"Statistic: {ks_stat:.4f}")
    st.write(f"p-value: {ks_p:.4f}")
    if ks_p > 0.05:
        st.write("The data appears to be normally distributed (fail to reject H0)")
    else:
        st.write("The data does not appear to be normally distributed (reject H0)")

def perform_linear_regression(df, x_var, y_var):
    st.subheader(f"Linear Regression: {y_var} vs {x_var}")
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
    ax.set_title(f"Linear Regression: {y_var} vs {x_var}")
    st.pyplot(fig)

    # Add error and MSE columns
    df['Predicted'] = y_pred
    df['Error'] = y - y_pred
    df['MSE'] = (df['Error'] ** 2)
    
    # Residual plot
    residuals = y - model.predict(X)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=model.predict(X), y=residuals, ax=ax)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel("Predicted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot")
    st.pyplot(fig)

    return df

def perform_multiple_linear_regression(df, y_var, x_vars):
    st.subheader(f"Multiple Linear Regression: {y_var} vs {', '.join(x_vars)}")
    df = df.dropna()
    
    X = df[x_vars]
    y = df[y_var]
    
    try:
        # Add constant term to the features
        X = sm.add_constant(X)
        
        # Fit the model
        model = sm.OLS(y, X).fit()
        
        # VIF calculation
        st.subheader("Variance Inflation Factor (VIF)")
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        vif_data = vif_data[vif_data["Variable"] != "const"]
        st.write(vif_data)

        # Display the summary
        st.write(model.summary())

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
        df['Absolue_Error'] = abs(y - df['Predicted'])
        df['MSE'] = round((df['Absolue_Error'] ** 2),8)
        
        return df
        
    except Exception as e:
        st.error(f"An error occurred during multiple linear regression: {str(e)}")
        st.write("Possible reasons for failure:")
        st.write("1. Multicollinearity among independent variables")
        st.write("2. Non-numeric data in selected columns")
        st.write("3. Insufficient data points")
        st.write("4. Perfect collinearity (one variable is a perfect linear combination of others)")

def perform_logistic_regression(df, y_var, x_vars):
    st.subheader(f"Logistic Regression: {y_var} vs {', '.join(x_vars)}")
    df = df.dropna()
    
    X = df[x_vars]
    y = df[y_var]
    
    try:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit the model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Model evaluation
        st.write("Model Coefficients:")
        for feature, coef in zip(x_vars, model.coef_[0]):
            st.write(f"{feature}: {coef:.4f}")
        
        st.write(f"Intercept: {model.intercept_[0]:.4f}")
        st.write(f"Accuracy: {model.score(X_test, y_test):.4f}")
        
        # ROC curve
        #fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        #roc_auc = auc(fpr, tpr)
        
        #fig, ax = plt.subplots(figsize=(10, 6))
        #ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        #ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        #ax.set_xlim([0.0, 1.0])
        #ax.set_ylim([0.0, 1.05])
        #ax.set_xlabel('False Positive Rate')
        # ax.set_ylabel('True Positive Rate')
        #ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        #ax.legend(loc="lower right")
        #st.pyplot(fig)
        
    except Exception as e:
        st.error(f"An error occurred during logistic regression: {str(e)}")

def perform_one_sample_ttest(df, variable, hypothesized_mean):
    st.subheader(f"One-Sample T-Test for {variable}")
    data = df[variable].dropna()
    t_stat, p_value = stats.ttest_1samp(data, hypothesized_mean)
    
    st.write(f"Hypothesized mean: {hypothesized_mean}")
    st.write(f"Sample mean: {data.mean():.4f}")
    st.write(f"t-statistic: {t_stat:.4f}")
    st.write(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        st.write("Reject the null hypothesis. There is a significant difference between the sample mean and the hypothesized mean.")
    else:
        st.write("Fail to reject the null hypothesis. There is no significant difference between the sample mean and the hypothesized mean.")

def perform_two_sample_ttest(df, var1, var2):
    st.subheader(f"Two-Sample T-Test: {var1} vs {var2}")
    data1 = df[var1].dropna()
    data2 = df[var2].dropna()
    
    t_stat, p_value = stats.ttest_ind(data1, data2)
    
    st.write(f"Mean of {var1}: {data1.mean():.4f}")
    st.write(f"Mean of {var2}: {data2.mean():.4f}")
    st.write(f"t-statistic: {t_stat:.4f}")
    st.write(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        st.write("Reject the null hypothesis. There is a significant difference between the two groups.")
    else:
        st.write("Fail to reject the null hypothesis. There is no significant difference between the two groups.")

def perform_paired_ttest(df, var1, var2):
    st.subheader(f"Paired T-Test: {var1} vs {var2}")
    data1 = df[var1].dropna()
    data2 = df[var2].dropna()
    
    t_stat, p_value = stats.ttest_rel(data1, data2)
    
    st.write(f"Mean difference: {(data1 - data2).mean():.4f}")
    st.write(f"t-statistic: {t_stat:.4f}")
    st.write(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        st.write("Reject the null hypothesis. There is a significant difference between the paired observations.")
    else:
        st.write("Fail to reject the null hypothesis. There is no significant difference between the paired observations.")

def perform_ztest(df, variable, population_mean, population_std):
    st.subheader(f"Z-Test for {variable}")
    data = df[variable].dropna()
    z_stat, p_value = ztest(data, value=population_mean)
    
    st.write(f"Population mean: {population_mean}")
    st.write(f"Population standard deviation: {population_std}")
    st.write(f"Sample mean: {data.mean():.4f}")
    st.write(f"z-statistic: {z_stat:.4f}")
    st.write(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        st.write("Reject the null hypothesis. There is a significant difference between the sample mean and the population mean.")
    else:
        st.write("Fail to reject the null hypothesis. There is no significant difference between the sample mean and the population mean.")

def perform_one_way_anova(df, selected_columns):
    st.subheader("One-Way ANOVA:")
    
    # Check if we have at least two columns selected
    if len(selected_columns) < 2:
        st.error("Please select at least two columns for one-way ANOVA.")
        return
    
    # Prepare data for ANOVA
    data = [df[col].dropna() for col in selected_columns]
    
    # Perform one-way ANOVA
    f_val, p_val = stats.f_oneway(*data)
    
    st.write(f"F-value: {f_val:.4f}")
    st.write(f"p-value: {p_val:.4f}")
    
    if p_val < 0.05:
        st.write("Reject the null hypothesis. There are significant differences between group means.")
    else:
        st.write("Fail to reject the null hypothesis. There are no significant differences between group means.")
    
    # Prepare data for Tukey's test
    all_data = []
    group_labels = []
    for col in selected_columns:
        all_data.extend(df[col].dropna())
        group_labels.extend([col] * len(df[col].dropna()))
    
    # Perform Tukey's HSD test
    tukey = pairwise_tukeyhsd(endog=all_data, groups=group_labels, alpha=0.05)
    st.write("\nTukey's HSD test results:")
    st.write(tukey)
    
    # Visualize the data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([df[col].dropna() for col in selected_columns])
    ax.set_xticklabels(selected_columns)
    ax.set_ylabel('Values')
    ax.set_title('Boxplot of Groups')
    st.pyplot(fig)

def perform_two_way_anova(df, value_col, factor1, factor2):
    df = df.dropna()
    st.subheader(f"Two-Way ANOVA: {value_col} by {factor1} and {factor2}")
    
    # Perform two-way ANOVA
    formula = f"{value_col} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})"
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    st.write(anova_table)
    
    # Create interaction plot
    fig, ax = plt.subplots(figsize=(10, 6))
    interaction_plot(df[factor1], df[factor2], df[value_col], ax=ax)
    ax.set_title(f"Interaction Plot: {factor1} and {factor2}")
    st.pyplot(fig)
def perform_gage_rr(df, part_col, operator_col, measurement_col):
    st.subheader("Gage R&R Analysis")
    
    # Calculate mean and range for each part-operator combination
    grouped = df.groupby([part_col, operator_col])
    means = grouped[measurement_col].mean().unstack()
    ranges = grouped[measurement_col].max() - grouped[measurement_col].min()
    
    # Calculate overall mean and average range
    overall_mean = means.mean().mean()
    average_range = ranges.mean()
    
    # Calculate variance components
    parts_var = means.var(axis=1).mean()
    operators_var = means.var(axis=0).mean()
    equipment_var = (average_range / 1.128) ** 2  # d2 = 1.128 for 2 measurements
    total_var = parts_var + operators_var + equipment_var
    
    # Calculate %Contribution and %Study Variation
    contribution = pd.DataFrame({
        'Variance': [parts_var, operators_var, equipment_var, total_var],
        '%Contribution': [parts_var/total_var*100, operators_var/total_var*100, equipment_var/total_var*100, 100],
        '%Study Variation': [np.sqrt(parts_var/total_var)*100, np.sqrt(operators_var/total_var)*100, np.sqrt(equipment_var/total_var)*100, 100]
    }, index=['Part-to-Part', 'Operator', 'Equipment', 'Total'])
    
    st.write("Variance Components:")
    st.write(contribution)
    
    # Calculate Gage R&R metrics
    gage_rr = np.sqrt(operators_var + equipment_var)
    prec_to_tol_ratio = 6 * gage_rr / (df[measurement_col].max() - df[measurement_col].min()) * 100
    
    st.write(f"\nGage R&R: {gage_rr:.4f}")
    st.write(f"Precision to Tolerance Ratio: {prec_to_tol_ratio:.2f}%")
    
    if prec_to_tol_ratio < 10:
        st.write("The measurement system is generally considered acceptable.")
    elif prec_to_tol_ratio < 30:
        st.write("The measurement system may be acceptable depending on the application, cost of measurement devices, cost of rework, or repairs.")
    else:
        st.write("The measurement system needs improvement.")

    
def main():
    st.set_page_config(page_title="Statistical Analysis Tool", layout="wide")
    st.sidebar.title("Select task")
    selection = st.sidebar.radio("Go to", ("Box Plot", "Control Chart", "Normality Test", 
                                           "Regression Analysis", "Hypothesis Testing", "ANOVA", "Gage R&R" ))

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df = convert_data_types(df)

            if selection == "Data Manipulation":
                st.subheader("Data Manipulation")
                st.write(df)
                
                # Option to delete rows
                if st.button("Delete selected rows"):
                    selected_indices = st.multiselect("Select rows to delete", df.index)
                    df = df.drop(selected_indices)
                    st.write("Updated dataframe:")
                    st.write(df)
                
                # Option to delete columns
                if st.button("Delete selected columns"):
                    selected_columns = st.multiselect("Select columns to delete", df.columns)
                    df = df.drop(columns=selected_columns)
                    st.write("Updated dataframe:")
                    st.write(df)

            elif selection == "Box Plot":
                input_var = st.sidebar.selectbox('Select Variable', df.columns)
                generate_boxplot(df, input_var)


            elif selection == "Control Chart":
                chart_type = st.sidebar.selectbox('Select chart Type', ['IMR', 'X-bar', 'P Chart'])
                
                if chart_type == 'IMR':
                    variable = st.sidebar.selectbox('Select Variable for Normality Test', df.columns)
                    generate_imr_chart(df, variable)
                elif chart_type == 'X-bar':
                   generate_xbar_chart(df) 
                elif chart_type == 'P Chart':
                   variable = st.sidebar.selectbox('Select Variable for Normality Test', df.columns)
                   generate_p_chart(df, variable)
            elif selection == "Normality Test":
                variable_for_normality = st.sidebar.selectbox('Select Variable for Normality Test', df.columns)
                perform_normality_test(df, variable_for_normality)
            
            elif selection == "Regression Analysis":
                regression_type = st.sidebar.selectbox('Select Regression Type', ['Simple Linear', 'Multiple Linear'])
                
                if regression_type == 'Simple Linear':
                    x_var = st.sidebar.selectbox('Select Independent Variable (X)', df.columns)
                    y_var = st.sidebar.selectbox('Select Dependent Variable (Y)', [col for col in df.columns if col != x_var])
                    perform_linear_regression(df, x_var, y_var)
                elif regression_type == 'Multiple Linear':
                    y_var = st.sidebar.selectbox('Select Dependent Variable (Y)', df.columns)
                    x_vars = st.sidebar.multiselect('Select Independent Variables (X)', [col for col in df.columns if col != y_var])
                    if len(x_vars) > 1:
                        perform_multiple_linear_regression(df, y_var, x_vars)
                    else:
                        st.warning("Please select at least two independent variables for multiple linear regression.")
                #else:  # Logistic Regression
                #    y_var = st.sidebar.selectbox('Select Dependent Variable (Y)', df.columns)
                #    x_vars = st.sidebar.multiselect('Select Independent Variables (X)', [col for col in df.columns if col != y_var])
                #    if len(x_vars) > 0:
                #        perform_logistic_regression(df, y_var, x_vars)
                 #   else:
                #        st.warning("Please select at least one independent variable for logistic regression.")
            
            elif selection == "Hypothesis Testing":
                test_type = st.sidebar.selectbox('Select Test Type', ['One-Sample T-Test', 'Two-Sample T-Test', 'Paired T-Test', 'Z-Test'])
                
                if test_type == 'One-Sample T-Test':
                    variable = st.sidebar.selectbox('Select Variable', df.columns)
                    hypothesized_mean = st.sidebar.number_input('Hypothesized Mean', value=0.0)
                    perform_one_sample_ttest(df, variable, hypothesized_mean)
                elif test_type == 'Two-Sample T-Test':
                    var1 = st.sidebar.selectbox('Select First Variable', df.columns)
                    var2 = st.sidebar.selectbox('Select Second Variable', [col for col in df.columns if col != var1])
                    perform_two_sample_ttest(df, var1, var2)
                elif test_type == 'Paired T-Test':
                    var1 = st.sidebar.selectbox('Select First Variable', df.columns)
                    var2 = st.sidebar.selectbox('Select Second Variable', [col for col in df.columns if col != var1])
                    perform_paired_ttest(df, var1, var2)
                else:  # Z-Test
                    variable = st.sidebar.selectbox('Select Variable', df.columns)
                    population_mean = st.sidebar.number_input('Population Mean', value=0.0)
                    population_std = st.sidebar.number_input('Population Standard Deviation', value=1.0, min_value=0.0)
                    perform_ztest(df, variable, population_mean, population_std)
            
            elif selection == "ANOVA":
                anova_type = st.sidebar.selectbox('Select ANOVA Type', ['One-Way ANOVA', 'Two-Way ANOVA'])
                
                if anova_type == 'One-Way ANOVA':
                   # group_col = st.sidebar.selectbox('Select Grouping Variable', df.columns)
                    selected_columns = st.sidebar.multiselect('Select Value Variable', [col for col in df.columns])
                    perform_one_way_anova(df, selected_columns)
                else:  # Two-Way ANOVA
                    value_col = st.sidebar.selectbox('Select Value Variable', df.columns)
                    factor1 = st.sidebar.selectbox('Select First Factor', [col for col in df.columns if col != value_col])
                    factor2 = st.sidebar.selectbox('Select Second Factor', [col for col in df.columns if col not in [value_col, factor1]])
                    perform_two_way_anova(df, value_col, factor1, factor2)
            
            elif selection == "Gage R&R":
                part_col = st.sidebar.selectbox('Select Part Column', df.columns)
                operator_col = st.sidebar.selectbox('Select Operator Column', [col for col in df.columns if col != part_col])
                measurement_col = st.sidebar.selectbox('Select Measurement Column', [col for col in df.columns if col not in [part_col, operator_col]])
                perform_gage_rr(df, part_col, operator_col, measurement_col)

        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
            st.write("Possible reasons for failure:")
            st.write("1. The file is not a valid CSV")
            st.write("2. The file is empty or corrupted")
            st.write("3. The file contains non-numeric data in columns expected to be numeric")

if __name__ == "__main__":
    main()    