# Importing libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as al
import streamlit as st

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('import export.csv')
    return data

data = load_data()

# Show raw data (optional)
st.title("Import-Export Analysis")
st.write("Preview of the dataset:")
st.dataframe(data.head(5))

# Sampling the dataset
dtsample = data.sample(n=3001, random_state=55044)

# Convert 'Date' column to datetime
dtsample['Date'] = pd.to_datetime(dtsample['Date'], dayfirst=True, errors='coerce')

# Sidebar to select analysis type
st.sidebar.title("Select Analysis Type")
analysis_type = st.sidebar.selectbox("Choose an analysis to display:", 
    ["Monthly Transaction Value Trends", 
     "Weight vs Value by Shipping Method", 
     "Quantity Trends by Category", 
     "Average Transaction Value by Payment Terms", 
     "Top Ports and Products by Quantity", 
     "Top Suppliers by Value and Payment Terms", 
     "Weight vs Value by Country", 
     "Top Products by Customs Code and Value", 
     "Quantity vs Value by Category", 
     "Transaction Value by Payment Terms", 
     "Seasonal Trends by Category", 
     "Quantity vs Value by Supplier", 
     "Scatter Plot of Quantity vs Value with Shipping Cost", 
     "Shipping Cost by Category and Payment Terms", 
     "Product Performance Radar Chart", 
     "Correlation Matrix", 
     "Waterfall Chart of Cumulative Category Value"]
)

# 1. Monthly Transaction Value Trends
if analysis_type == "Monthly Transaction Value Trends":
    st.subheader("Monthly Transaction Value Trends")
    monthly_value = dtsample.groupby(dtsample['Date'].dt.to_period('M'))['Value'].sum()

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_value.index.astype(str), monthly_value.values, marker='o', color='orange')
    plt.title('Monthly Transaction Value Trends')
    plt.xlabel('Month-Year')
    plt.ylabel('Total Transaction Value')
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)

# 2. Weight vs Value by Shipping Method
elif analysis_type == "Weight vs Value by Shipping Method":
    st.subheader("Weight vs Value by Shipping Method")
    
    plt.figure(figsize=(10, 6))
    for method in dtsample['Shipping_Method'].unique():
        subset = dtsample[dtsample['Shipping_Method'] == method]
        plt.scatter(subset['Weight'], subset['Value'], label=method, alpha=0.7)

    plt.title('Weight vs. Value by Shipping Method')
    plt.xlabel('Weight')
    plt.ylabel('Value')
    plt.legend(title='Shipping Method')
    plt.grid(True)
    st.pyplot(plt)

# 3. Quantity Trends by Category
elif analysis_type == "Quantity Trends by Category":
    st.subheader("Quantity Trends by Category")
    date_category_quantity = dtsample.groupby([dtsample['Date'].dt.to_period('M'), 'Category'])['Quantity'].sum().reset_index()
    date_category_quantity['Date'] = date_category_quantity['Date'].dt.to_timestamp()

    pivot_df = date_category_quantity.pivot(index='Date', columns='Category', values='Quantity')

    plt.figure(figsize=(12, 6))
    for category in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[category], label=category)

    plt.title('Quantity Trend Over Time by Category')
    plt.xlabel('Date')
    plt.ylabel('Total Quantity')
    plt.legend(title='Category')
    plt.grid(True)
    st.pyplot(plt)

# 4. Average Transaction Value by Payment Terms
elif analysis_type == "Average Transaction Value by Payment Terms":
    st.subheader("Average Transaction Value by Payment Terms")
    payment_terms_value = dtsample.groupby('Payment_Terms')['Value'].mean().reset_index()

    plt.figure(figsize=(8, 6))
    plt.bar(payment_terms_value['Payment_Terms'], payment_terms_value['Value'], color='teal')
    plt.title('Average Transaction Value by Payment Terms')
    plt.xlabel('Payment Terms')
    plt.ylabel('Average Transaction Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

# 5. Top Ports and Products by Quantity
elif analysis_type == "Top Ports and Products by Quantity":
    st.subheader("Top Ports and Products by Quantity")
    port_product_quantity = dtsample.groupby(['Port', 'Product'])['Quantity'].sum().reset_index()
    top_port_product = port_product_quantity.sort_values('Quantity', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(top_port_product['Port'] + ' - ' + top_port_product['Product'], top_port_product['Quantity'], color='purple')
    plt.title('Top Ports by Quantity and Product')
    plt.xlabel('Total Quantity')
    plt.ylabel('Port and Product')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    st.pyplot(plt)

# 6. Top Suppliers by Value and Payment Terms
elif analysis_type == "Top Suppliers by Value and Payment Terms":
    st.subheader("Top Suppliers by Value and Payment Terms")
    supplier_payment_value = dtsample.groupby(['Supplier', 'Payment_Terms'])['Value'].sum().reset_index()
    top_suppliers = supplier_payment_value.sort_values('Value', ascending=False).head(10)

    plt.figure(figsize=(12, 6))
    plt.bar(top_suppliers['Supplier'], top_suppliers['Value'], color='coral')
    plt.title('Top Suppliers by Transaction Value and Payment Terms')
    plt.xlabel('Supplier')
    plt.ylabel('Total Transaction Value')
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(plt)

# 7. Weight vs Value by Country
elif analysis_type == "Weight vs Value by Country":
    st.subheader("Weight vs Value by Country")
    country_weight_value = dtsample.groupby('Country')[['Weight', 'Value']].sum().reset_index()

    plt.figure(figsize=(10, 6))
    plt.scatter(country_weight_value['Weight'], country_weight_value['Value'], color='slateblue', alpha=0.7)
    for i, txt in enumerate(country_weight_value['Country']):
        plt.annotate(txt, (country_weight_value['Weight'][i], country_weight_value['Value'][i]), fontsize=8)

    plt.title('Total Weight vs. Total Value by Country')
    plt.xlabel('Total Weight')
    plt.ylabel('Total Value')
    plt.grid(True)
    st.pyplot(plt)

# 8. Top Products by Customs Code and Value
elif analysis_type == "Top Products by Customs Code and Value":
    st.subheader("Top Products by Customs Code and Transaction Value")
    customs_product_value = dtsample.groupby(['Customs_Code', 'Product'])['Value'].sum().reset_index()
    top_customs_products = customs_product_value.sort_values('Value', ascending=False).head(10)

    plt.figure(figsize=(12, 6))
    plt.bar(top_customs_products['Customs_Code'].astype(str) + ' - ' + top_customs_products['Product'], 
            top_customs_products['Value'], color='olive')
    plt.title('Top Products by Customs Code and Transaction Value')
    plt.xlabel('Customs Code and Product')
    plt.ylabel('Total Transaction Value')
    plt.xticks(rotation=90)
    plt.tight_layout()
    st.pyplot(plt)

# 9. Quantity vs Value by Category
elif analysis_type == "Quantity vs Value by Category":
    st.subheader("Quantity vs Value by Category")
    category_quantity_value = dtsample.groupby('Category')[['Quantity', 'Value']].sum().reset_index()

    plt.figure(figsize=(8, 6))
    plt.scatter(category_quantity_value['Quantity'], category_quantity_value['Value'], color='darkcyan', s=100)
    for i, txt in enumerate(category_quantity_value['Category']):
        plt.annotate(txt, (category_quantity_value['Quantity'][i], category_quantity_value['Value'][i]), fontsize=9)

    plt.title('Quantity vs. Value by Category')
    plt.xlabel('Total Quantity')
    plt.ylabel('Total Value')
    plt.grid(True)
    st.pyplot(plt)

# 10. Transaction Value by Payment Terms
elif analysis_type == "Transaction Value by Payment Terms":
    st.subheader("Transaction Value Distribution by Payment Terms")
    payment_terms_value = dtsample.groupby('Payment_Terms')['Value'].sum()

    plt.figure(figsize=(8, 8))
    plt.pie(payment_terms_value, labels=payment_terms_value.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title('Transaction Value Distribution by Payment Terms')
    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
    st.pyplot(plt)

# 11. Seasonal Trends by Category
elif analysis_type == "Seasonal Trends by Category":
    st.subheader("Seasonal Trends in Transactions by Category")
    dtsample['Month'] = dtsample['Date'].dt.month
    seasonal_trends = dtsample.groupby(['Month', 'Category'])['Value'].sum().reset_index()
    seasonal_trends_pivot = seasonal_trends.pivot(index='Month', columns='Category', values='Value')

    plt.figure(figsize=(12, 6))
    for category in seasonal_trends_pivot.columns:
        plt.plot(seasonal_trends_pivot.index, seasonal_trends_pivot[category], marker='o', label=category)

    plt.title('Seasonal Trends in Transaction Value by Category')
    plt.xlabel('Month')
    plt.ylabel('Total Transaction Value')
    plt.xticks(seasonal_trends_pivot.index, rotation=0)
    plt.legend(title='Category')
    plt.grid(True)
    st.pyplot(plt)

# 12. Quantity vs Value by Supplier
elif analysis_type == "Quantity vs Value by Supplier":
    st.subheader("Quantity vs Value by Supplier")
    supplier_quantity_value = dtsample.groupby('Supplier')[['Quantity', 'Value']].sum().reset_index()

    plt.figure(figsize=(10, 6))
    plt.scatter(supplier_quantity_value['Quantity'], supplier_quantity_value['Value'], color='brown', alpha=0.6)
    for i, txt in enumerate(supplier_quantity_value['Supplier']):
        plt.annotate(txt, (supplier_quantity_value['Quantity'][i], supplier_quantity_value['Value'][i]), fontsize=8)

    plt.title('Quantity vs. Value by Supplier')
    plt.xlabel('Total Quantity')
    plt.ylabel('Total Value')
    plt.grid(True)
    st.pyplot(plt)

# 13. Scatter Plot of Quantity vs Value with Shipping Cost
elif analysis_type == "Scatter Plot of Quantity vs Value with Shipping Cost":
    st.subheader("Scatter Plot of Quantity vs Value with Shipping Cost")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(dtsample['Quantity'], dtsample['Value'], c=dtsample['Shipping_Cost'], cmap='viridis', alpha=0.5)
    plt.colorbar(label='Shipping Cost')
    plt.title('Quantity vs. Value with Shipping Cost as Color')
    plt.xlabel('Total Quantity')
    plt.ylabel('Total Value')
    plt.grid(True)
    st.pyplot(plt)

# 14. Shipping Cost by Category and Payment Terms
elif analysis_type == "Shipping Cost by Category and Payment Terms":
    st.subheader("Shipping Cost by Category and Payment Terms")
    category_payment_cost = dtsample.groupby(['Category', 'Payment_Terms'])['Shipping_Cost'].sum().unstack()

    plt.figure(figsize=(12, 6))
    category_payment_cost.plot(kind='bar', stacked=True)
    plt.title('Shipping Cost by Category and Payment Terms')
    plt.xlabel('Category')
    plt.ylabel('Total Shipping Cost')
    plt.xticks(rotation=45)
    plt.legend(title='Payment Terms')
    plt.grid(True)
    st.pyplot(plt)

# 15. Product Performance Radar Chart
elif analysis_type == "Product Performance Radar Chart":
    st.subheader("Product Performance Radar Chart")
    product_performance = dtsample.groupby('Product').agg({'Quantity': 'sum', 'Value': 'sum', 'Weight': 'sum'}).reset_index()

    categories = list(product_performance.columns[1:])
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    values = product_performance.loc[0, categories].values.flatten().tolist()
    values += values[:1]

    plt.figure(figsize=(8, 8), dpi=150)
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories)
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)
    plt.title(f'Performance of Product: {product_performance.loc[0, "Product"]}')
    st.pyplot(plt)

# 16. Correlation Matrix
elif analysis_type == "Correlation Matrix":
    st.subheader("Correlation Matrix")
    correlation_data = dtsample[['Value', 'Quantity', 'Weight', 'Shipping_Cost']]
    corr = correlation_data.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    st.pyplot(plt)

# 17. Waterfall Chart of Cumulative Category Value
elif analysis_type == "Waterfall Chart of Cumulative Category Value":
    st.subheader("Waterfall Chart of Cumulative Category Value")
    category_value = dtsample.groupby('Category')['Value'].sum().reset_index()
    category_value = category_value.sort_values('Value', ascending=False)
    
    cumulative_value = category_value['Value'].cumsum()
    
    plt.figure(figsize=(10, 6))
    plt.bar(category_value['Category'], category_value['Value'], color='green')
    plt.plot(category_value['Category'], cumulative_value, color='orange', marker='o', label='Cumulative Value')
    plt.title('Cumulative Value by Category')
    plt.xlabel('Category')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Add any other custom analyses you would like from your list, structured similarly.
