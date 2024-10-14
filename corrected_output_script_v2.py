# Importing necessary libraries
import os
import pandas as pd
import numpy as np

# Ensure that matplotlib and seaborn are installed
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Error importing libraries: {e}. Try installing required packages by running:")
    print("pip install matplotlib seaborn")

import altair as al
import streamlit as st

# Load the data
try:
    data = pd.read_csv('C:\Users\ASUS\OneDrive\Desktop\import export.csv')
    data.head(5)
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Make sure the CSV file path is correct.")

# Taking the sample dataset
dtsample = data.sample(n=3001, random_state=55044)
dtsample.head(10)

# Generate monthly transaction value trends
monthly_value = dtsample.groupby(dtsample['Date'].dt.to_period('M'))['Value'].sum()

# Plotting monthly transaction value trends
plt.figure(figsize=(12, 6))
plt.plot(monthly_value.index.astype(str), monthly_value.values, marker='o', color='orange')
plt.title('Monthly Transaction Value Trends')
plt.xlabel('Month-Year')
plt.ylabel('Total Transaction Value')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Scatter plot of Weight vs. Value colored by Shipping Method
plt.figure(figsize=(10, 6))
for method in dtsample['Shipping_Method'].unique():
    subset = dtsample[dtsample['Shipping_Method'] == method]
    plt.scatter(subset['Weight'], subset['Value'], label=method, alpha=0.7)

plt.title('Weight vs. Value by Shipping Method')
plt.xlabel('Weight')
plt.ylabel('Value')
plt.legend(title='Shipping Method')
plt.grid(True)
plt.show()

# Grouping data by Date and Category, summing the Quantity
dtsample['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
date_category_quantity = dtsample.groupby([dtsample['Date'].dt.to_period('M'), 'Category'])['Quantity'].sum().reset_index()
date_category_quantity['Date'] = date_category_quantity['Date'].dt.to_timestamp()

# Pivot data for plotting
pivot_df = date_category_quantity.pivot(index='Date', columns='Category', values='Quantity')

# Plot time series data
plt.figure(figsize=(12, 6))
for category in pivot_df.columns:
    plt.plot(pivot_df.index, pivot_df[category], label=category)
plt.title('Time Series of Quantity by Category')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.legend()
plt.grid(True)
plt.show()

# Bar plot of Transaction Value by Category and Payment Terms
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Value', hue='Payment_Terms', data=dtsample, palette='muted')
plt.title('Transaction Value by Category and Payment Terms', fontsize=14)
plt.xlabel('Category')
plt.ylabel('Average Transaction Value')
plt.grid(True)
plt.show()

# Plotting correlation matrix heatmap
numeric_cols = dtsample.select_dtypes(include='number')
corr_matrix = numeric_cols.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Create waterfall chart for Category and Value
waterfall_data = dtsample[['Category', 'Value']].groupby('Category').sum().reset_index()
waterfall_data['Cumulative'] = waterfall_data['Value'].cumsum()

colors = ['blue' if val >= 0 else 'red' for val in waterfall_data['Value']]
fig, ax = plt.subplots(figsize=(12, 8))
bars = plt.bar(waterfall_data['Category'], waterfall_data['Value'], color=colors)

for bar, value in zip(bars, waterfall_data['Cumulative']):
    height = bar.get_height()
    ax.annotate(f'{value:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 5),
                textcoords="offset points", ha='center', va='bottom')

plt.title('Waterfall Chart: Category-wise Cumulative Value', fontsize=16)
plt.xlabel('Category')
plt.ylabel('Value')
plt.show()
