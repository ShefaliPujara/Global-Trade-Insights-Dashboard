import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Upload CSV via Streamlit
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    
    # Sample the dataset
    dtsample = data.sample(n=3001, random_state=55044)

    # Ensure 'Date' column is in datetime format
    if 'Date' in dtsample.columns:
        dtsample['Date'] = pd.to_datetime(dtsample['Date'], errors='coerce')

    # Monthly Transaction Value Trend
    monthly_value = dtsample.groupby(dtsample['Date'].dt.to_period('M'))['Value'].sum()
    st.write("Monthly Transaction Value:")
    st.write(monthly_value)

    # Plot the monthly transaction value
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_value.index.astype(str), monthly_value.values, marker='o', color='orange')
    plt.title('Monthly Transaction Value Trends')
    plt.xlabel('Month-Year')
    plt.ylabel('Total Transaction Value')
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)  # Use Streamlit to display the plot

    # Scatter plot of Weight vs. Value colored by Shipping Method
    if 'Shipping_Method' in dtsample.columns and 'Weight' in dtsample.columns:
        plt.figure(figsize=(10, 6))
        for method in dtsample['Shipping_Method'].unique():
            subset = dtsample[dtsample['Shipping_Method'] == method]
            plt.scatter(subset['Weight'], subset['Value'], label=method, alpha=0.7)
        plt.title('Weight vs. Value by Shipping Method')
        plt.xlabel('Weight')
        plt.ylabel('Value')
        plt.legend()
        st.pyplot(plt)

    # Bar plot of Value by Category and Payment Terms
    if 'Category' in dtsample.columns and 'Payment_Terms' in dtsample.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Category', y='Value', hue='Payment_Terms', data=dtsample, palette='muted')
        plt.title('Transaction Value by Category and Payment Terms')
        plt.xlabel('Category')
        plt.ylabel('Average Transaction Value')
        plt.grid(True)
        st.pyplot(plt)
        
    # Correlation heatmap
    numeric_cols = dtsample.select_dtypes(include='number')
    if not numeric_cols.empty:
        corr_matrix = numeric_cols.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix Heatmap')
        st.pyplot(plt)
