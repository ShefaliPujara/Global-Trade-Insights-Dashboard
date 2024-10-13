import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Function to load data
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Ensure Date column is in datetime format
    return data

# File uploader for user to upload a CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    data = load_data(uploaded_file)

    # Taking a sample of the dataset
    dtsample = data.sample(n=3001, random_state=55044)

    # Streamlit dashboard layout
    st.title('Trade Value Dashboard')
    st.write("### Imports, Exports, and Supply Chain Efficiency")

    # Data preview
    st.write("#### Sample Data")
    st.dataframe(dtsample.head(10))

    # Interactive Filter: Select Shipping Method
    shipping_methods = dtsample['Shipping_Method'].unique()
    selected_method = st.selectbox('Select Shipping Method', shipping_methods)

    # Filter dataset based on user selection
    filtered_data = dtsample[dtsample['Shipping_Method'] == selected_method]

    # Monthly Transaction Value Trends
    st.write("### Monthly Transaction Value Trends")
    monthly_value = dtsample.groupby(dtsample['Date'].dt.to_period('M'))['Value'].sum()

    # Plotting the monthly transaction trends
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_value.index.astype(str), monthly_value.values, marker='o', color='orange')
    plt.title('Monthly Transaction Value Trends')
    plt.xlabel('Month-Year')
    plt.ylabel('Total Transaction Value')
    plt.xticks(rotation=45)
    plt.grid(True)

    # Display the plot in Streamlit
    st.pyplot(plt)
    plt.close()  # Close the plot to prevent overlapping

    # Scatter plot of Weight vs. Value for the selected Shipping Method
    st.write(f"### Weight vs. Value for {selected_method}")

    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_data['Weight'], filtered_data['Value'], alpha=0.7)
    plt.title(f'Weight vs. Value for {selected_method}')
    plt.xlabel('Weight')
    plt.ylabel('Value')
    plt.grid(True)

    # Display scatter plot in Streamlit
    st.pyplot(plt)
    plt.close()  # Close the plot to prevent overlapping
