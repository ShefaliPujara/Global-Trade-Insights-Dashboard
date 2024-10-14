#importing the libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as al
import streamlit as st


# Load the data
data = pd.read_csv('C:\\Users\\ASUS\\OneDrive\\Desktop\\import export.csv')
data.head(5)

#Taking the sample dataset
dtsample=data.sample(n=3001 , random_state=55044)
dtsample.head(10)

monthly_value = dtsample.groupby(dtsample['Date'].dt.to_period('M'))['Value'].sum()

plt.figure(figsize=(12, 6))
plt.plot(monthly_value.index.astype(str), monthly_value.values, marker='o', color='orange')
plt.title('Monthly Transaction Value Trends')
plt.xlabel('Month-Year')
plt.ylabel('Total Transaction Value')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt

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


import pandas as pd
import matplotlib.pyplot as plt

# Convert the 'Date' column to datetime format
dtsample['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')

# Group data by Date and Category, summing the Quantity
date_category_quantity = dtsample.groupby([dtsample['Date'].dt.to_period('M'), 'Category'])['Quantity'].sum().reset_index()
date_category_quantity['Date'] = date_category_quantity['Date'].dt.to_timestamp()

# Pivot the data for plotting
pivot_df = date_category_quantity.pivot(index='Date', columns='Category', values='Quantity')

# Plot the time series
plt.figure(figsize=(12, 6))
for category in pivot_df.columns:
    plt.plot(pivot_df.index, pivot_df[category], label=category)

plt.title('Quantity Trend Over Time by Category')
plt.xlabel('Date')
plt.ylabel('Total Quantity')
plt.legend(title='Category')
plt.grid(True)
plt.show()


# Calculate average transaction value by Payment Terms
payment_terms_value = dtsample.groupby('Payment_Terms')['Value'].mean().reset_index()

# Bar plot
plt.figure(figsize=(8, 6))
plt.bar(payment_terms_value['Payment_Terms'], payment_terms_value['Value'], color='teal')
plt.title('Average Transaction Value by Payment Terms')
plt.xlabel('Payment Terms')
plt.ylabel('Average Transaction Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Group data by Port and Product, summing the Quantity
port_product_quantity = dtsample.groupby(['Port', 'Product'])['Quantity'].sum().reset_index()

# Sort and select top entries
top_port_product = port_product_quantity.sort_values('Quantity', ascending=False).head(10)

# Horizontal bar plot
plt.figure(figsize=(10, 6))
plt.barh(top_port_product['Port'] + ' - ' + top_port_product['Product'], top_port_product['Quantity'], color='purple')
plt.title('Top Ports by Quantity and Product')
plt.xlabel('Total Quantity')
plt.ylabel('Port and Product')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# Group data by Supplier and Payment Terms, summing the Value
supplier_payment_value = dtsample.groupby(['Supplier', 'Payment_Terms'])['Value'].sum().reset_index()

# Sort and select top entries
top_suppliers = supplier_payment_value.sort_values('Value', ascending=False).head(10)

# Bar plot
plt.figure(figsize=(12, 6))
plt.bar(top_suppliers['Supplier'], top_suppliers['Value'], color='coral')
plt.title('Top Suppliers by Transaction Value and Payment Terms')
plt.xlabel('Supplier')
plt.ylabel('Total Transaction Value')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Display payment terms
print(top_suppliers[['Supplier', 'Payment_Terms']])


# Group data by Country, summing Weight and Value
country_weight_value = dtsample.groupby('Country')[['Weight', 'Value']].sum().reset_index()

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(country_weight_value['Weight'], country_weight_value['Value'], color='slateblue', alpha=0.7)
for i, txt in enumerate(country_weight_value['Country']):
    plt.annotate(txt, (country_weight_value['Weight'][i], country_weight_value['Value'][i]), fontsize=8)

plt.title('Total Weight vs. Total Value by Country')
plt.xlabel('Total Weight')
plt.ylabel('Total Value')
plt.grid(True)
plt.show()


# Group data by Customs Code and Product, summing the Value
customs_product_value = dtsample.groupby(['Customs_Code', 'Product'])['Value'].sum().reset_index()

# Sort and select top entries
top_customs_products = customs_product_value.sort_values('Value', ascending=False).head(10)

# Bar plot
plt.figure(figsize=(12, 6))
plt.bar(top_customs_products['Customs_Code'].astype(str) + ' - ' + top_customs_products['Product'], top_customs_products['Value'], color='olive')
plt.title('Top Products by Customs Code and Transaction Value')
plt.xlabel('Customs Code and Product')
plt.ylabel('Total Transaction Value')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Group data by Category, summing Quantity and Value
category_quantity_value = dtsample.groupby('Category')[['Quantity', 'Value']].sum().reset_index()

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(category_quantity_value['Quantity'], category_quantity_value['Value'], color='darkcyan', s=100)
for i, txt in enumerate(category_quantity_value['Category']):
    plt.annotate(txt, (category_quantity_value['Quantity'][i], category_quantity_value['Value'][i]), fontsize=9)

plt.title('Quantity vs. Value by Category')
plt.xlabel('Total Quantity')
plt.ylabel('Total Value')
plt.grid(True)
plt.show()


payment_terms_value = dtsample.groupby('Payment_Terms')['Value'].sum()

plt.figure(figsize=(8, 8))
plt.pie(payment_terms_value, labels=payment_terms_value.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title('Transaction Value Distribution by Payment Terms')
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
plt.show()


# Create a new column for the month
dtsample['Month'] = dtsample['Date'].dt.month

# Group by Month and Category, summing the Value
seasonal_trends = dtsample.groupby(['Month', 'Category'])['Value'].sum().reset_index()

# Pivot for plotting
seasonal_trends_pivot = seasonal_trends.pivot(index='Month', columns='Category', values='Value')

# Plotting
plt.figure(figsize=(12, 6))
for category in seasonal_trends_pivot.columns:
    plt.plot(seasonal_trends_pivot.index, seasonal_trends_pivot[category], label=category)

plt.title('Seasonal Trends in Transactions by Category')
plt.xlabel('Month')
plt.ylabel('Total Transaction Value')
plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(title='Category')
plt.grid(True)
plt.tight_layout()
plt.show()


# Group by Supplier and calculate total Quantity and Value
supplier_qty_value = dtsample.groupby('Supplier')[['Quantity', 'Value']].sum().reset_index()

# Scatter plot for Quantity vs. Value
plt.figure(figsize=(10, 6))
plt.scatter(supplier_qty_value['Quantity'], supplier_qty_value['Value'], color='forestgreen', s=100)
for i, txt in enumerate(supplier_qty_value['Supplier']):
    plt.annotate(txt, (supplier_qty_value['Quantity'][i], supplier_qty_value['Value'][i]), fontsize=8)

plt.title('Quantity vs. Transaction Value by Supplier')
plt.xlabel('Total Quantity')
plt.ylabel('Total Transaction Value')
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Sample data (you can replace this with your dataset)
dtsample = pd.DataFrame({
    'Quantity': [1, 2, 3, 4, 5, 6, 7, 8],
    'Value': [100, 200, 250, 300, 350, 400, 500, 600],
    'Shipping_Cost': [15, 20, 25, 30, 35, 40, 45, 50]
})

plt.figure(figsize=(10, 6))

# Scatter plot with color map based on Shipping Cost
scatter = plt.scatter(dtsample['Quantity'], dtsample['Value'], c=dtsample['Shipping_Cost'], cmap='coolwarm', s=100, edgecolor='k')

# Color bar
cbar = plt.colorbar(scatter)
cbar.set_label('Shipping Cost')

plt.title('Scatter Plot of Quantity vs. Value with Shipping Cost')
plt.xlabel('Quantity Sold')
plt.ylabel('Transaction Value')
plt.grid(True)
plt.show()


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Sample data (replace with your dataset)
dtsample = pd.DataFrame({
    'Category': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
    'Shipping_Cost': [50, 60, 55, 65, 45, 55, 70, 80],
    'Payment_Terms': ['Net30', 'Net60', 'Net30', 'Net60', 'Net30', 'Net60', 'Net30', 'Net60']
})

plt.figure(figsize=(10, 6))

# Box plot with category and payment terms
sns.boxplot(x='Category', y='Shipping_Cost', hue='Payment_Terms', data=dtsample, palette='Set2')

plt.title('Shipping Cost by Category and Payment Terms', fontsize=14)
plt.xlabel('Category')
plt.ylabel('Shipping Cost')
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt
from math import pi

# Select a subset of data for the radar chart
subset = dtsample[['Quantity', 'Value', 'Weight', 'Customs_Code']].iloc[0]

# Normalize data for better visualization
subset_normalized = (subset - subset.min()) / (subset.max() - subset.min())

# Define the categories
categories = ['Quantity', 'Value', 'Weight', 'Customs_Code']
values = subset_normalized.tolist()
values += values[:1]  # Repeat first value to close the radar chart

# Create radar chart
plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)

# Calculate angle of each axis
angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
angles += angles[:1]

# Set up the radar chart
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Add labels to each axis
plt.xticks(angles[:-1], categories)

# Plot the data
ax.plot(angles, values, linewidth=2, linestyle='solid')
ax.fill(angles, values, 'b', alpha=0.1)

# Set the title and show the plot
plt.title('Product Performance Radar Chart')
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data (replace with your dataset)
dtsample = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D'],
    'Value': [1000, 2000, 1500, 1800, 1100, 2200, 1600, 1900],
    'Payment_Terms': ['Net30', 'Net30', 'Net60', 'Net60', 'Net30', 'Net30', 'Net60', 'Net60']
})

plt.figure(figsize=(10, 6))

# Bar plot with category and payment terms
sns.barplot(x='Category', y='Value', hue='Payment_Terms', data=dtsample, palette='muted')

plt.title('Transaction Value by Category and Payment Terms', fontsize=14)
plt.xlabel('Category')
plt.ylabel('Average Transaction Value')
plt.grid(True)
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Filter numeric columns only
numeric_cols = dtsample.select_dtypes(include='number')

# Calculate the correlation matrix
corr_matrix = numeric_cols.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt


# Create a sample for 'Category' and 'Value' columns, assuming 'Category' exists
waterfall_data = dtsample[['Category', 'Value']].groupby('Category').sum().reset_index()

# Calculate cumulative values for the waterfall chart
waterfall_data['Cumulative'] = waterfall_data['Value'].cumsum()

# Assign colors: increases as blue and decreases as red
colors = ['blue' if val >= 0 else 'red' for val in waterfall_data['Value']]

# Create the waterfall chart
fig, ax = plt.subplots(figsize=(12, 8))
bars = plt.bar(waterfall_data['Category'], waterfall_data['Value'], color=colors)

# Annotating cumulative value on top of the bars
for bar, value in zip(bars, waterfall_data['Cumulative']):
    height = bar.get_height()
    ax.annotate(f'{value:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points", ha='center', va='bottom')

plt.title('Waterfall Chart: Category-wise Cumulative Value', fontsize=16)
plt.xlabel('Category')
plt.ylabel('Value')

plt.show()




