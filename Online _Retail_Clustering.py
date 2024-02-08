import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

retail = pd.read_csv('OnlineRetail.csv', encoding = 'ISO-8859-1', sep=',')


### Data Cleaning
# Calculate the number of null values in the DataFrame
retail_null = retail.isnull().sum()

# Remove rows with missing values
retail  = retail.dropna()

# Change the type of the CustomerID column to str
retail['CustomerID'] = retail['CustomerID'].astype(str)

# Remove rows with negative values in the Quantity column
retail = retail[retail['Quantity']>0]

# Remove rows with negative or zero values in the UnitPrice column
retail = retail[retail['UnitPrice']>0]
# New shape is (397884, 8)



### Data Preprocessing
# Creatte a new column for total price and it will be one of the feature that can be used for clustering
retail['Total Price'] = retail['Quantity']*retail['UnitPrice']
# Aggregate each customer's total purchase
customer_total_purchase = retail.groupby('CustomerID').agg({'Total Price': lambda x: x.sum()})
customer_total_purchase = customer_total_purchase.reset_index()


# Doing thhe same for frequency of purchase by defining new attribute called Frequency
customer_frequency = retail.groupby('CustomerID').agg({'InvoiceNo': lambda x: x.nunique()})
customer_frequency = customer_frequency.reset_index()

# Merge the two DataFrames
merge_df = pd.merge(customer_total_purchase, customer_frequency, on='CustomerID', how='inner')

# Convert the InvoiceDate column to datetime and desired format day month year
retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'])
# Calculate the last transaction date to find the recency of the purchase
last_date = retail['InvoiceDate'].max()
# Calculate the difference between the last transaction date and the transaction date
retail['Difference'] = last_date - retail['InvoiceDate']

# Calculate the recency of the customer
customer_recency = retail.groupby('CustomerID').agg({'Difference': lambda x: x.min()})
customer_recency = customer_recency.reset_index()
# Extract the number of days from the Difference column
customer_recency['Difference'] = customer_recency['Difference'].dt.days

# Merge the recency DataFrame with the previous merged DataFrame
merge_df = pd.merge(merge_df, customer_recency, on='CustomerID', how='inner')
merge_df.columns = ['CustomerID', 'Total Purchase', 'Frequency', 'Recency']



### Detecting Outliers
# Detect outliers in three columns unsing seaborn boxplot
plt.figure(figsize=(15,10))
plt.subplot(3, 1, 1)
sns.boxplot(merge_df['Total Purchase'])
plt.subplot(3, 1, 2)
sns.boxplot(merge_df['Frequency'])
plt.subplot(3, 1, 3)
sns.boxplot(merge_df['Recency'])
# plt.show()
# Remove outliers from the Total Purchase column
Q1 = merge_df['Total Purchase'].quantile(0.05)
Q3 = merge_df['Total Purchase'].quantile(0.95)
IQR = Q3 - Q1
merge_df = merge_df[(merge_df['Total Purchase'] >= Q1 - 1.5*IQR) & (merge_df['Total Purchase'] <= Q3 + 1.5*IQR)]
# Shape of the DataFrame after removing outliers is (4293, 4)

# Remove outliers from the Frequency column
Q1 = merge_df['Frequency'].quantile(0.05)
Q3 = merge_df['Frequency'].quantile(0.95)
IQR = Q3 - Q1
merge_df = merge_df[(merge_df['Frequency'] >= Q1 - 1.5*IQR) & (merge_df['Frequency'] <= Q3 + 1.5*IQR)]
# Shape of the DataFrame after removing outliers is (4261,4)

# Remove outliers from the Recency column
Q1 = merge_df['Recency'].quantile(0.05)
Q3 = merge_df['Recency'].quantile(0.95)
IQR = Q3 - Q1
merge_df = merge_df[(merge_df['Recency'] >= Q1 - 1.5*IQR) & (merge_df['Recency'] <= Q3 + 1.5*IQR)]
# Shape of the DataFrame after removing outliers is (4261, 4)




### Feature Scaling and Standardization
from sklearn.preprocessing import StandardScaler
# Standardize the features
scaler = StandardScaler()
scaler.fit(merge_df.iloc[:,1:])
merge_df_scaled = scaler.transform(merge_df.iloc[:,1:])
merge_df_scaled = pd.DataFrame(merge_df_scaled)
merge_df_scaled.columns = ['Total Purchase', 'Frequency', 'Recency']

### K-Means Clustering
from sklearn.cluster import KMeans
# Create a list to store the inertia values
inertia = []
# Create a range of values for K
K = range(1, 10)
# Create a for loop to find the optimal number of clusters
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(merge_df_scaled)
    inertia.append(kmeans.inertia_)
# Plot the inertia values
plt.plot(K, inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
# plt.show()
# Based on the plot, the optimal number of clusters is 3

# Create a KMeans model with 3 clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(merge_df_scaled)
# Add the cluster labels to the DataFrame
merge_df['Cluster'] = kmeans.labels_

# Plot the clusters
plt.figure(figsize=(10,7))
sns.scatterplot(x='Total Purchase', y='Frequency', hue='Cluster', palette='viridis', data=merge_df)
plt.title('KMeans Clustering')
plt.xlabel('Total Purchase')
plt.ylabel('Frequency')
#plt.show()

# Plot the clusters
plt.figure(figsize=(10,7))
sns.scatterplot(x='Total Purchase', y='Recency', hue='Cluster', palette='viridis', data=merge_df)
plt.title('KMeans Clustering')
plt.xlabel('Total Purchase')
plt.ylabel('Recency')
#plt.show()

# Plot the clusters
plt.figure(figsize=(10,7))
sns.scatterplot(x='Frequency', y='Recency', hue='Cluster', palette='viridis', data=merge_df)
plt.title('KMeans Clustering')
plt.xlabel('Frequency')
plt.ylabel('Recency')
#plt.show()

# Plot boxplots for each cluster
plt.figure(figsize=(10,7))
plt.subplot(3, 1, 1)
sns.boxplot(x=merge_df['Cluster'], y=merge_df['Total Purchase'])
plt.subplot(3, 1, 2)
sns.boxplot(x=merge_df['Cluster'], y=merge_df['Frequency'])
plt.subplot(3, 1, 3)
sns.boxplot(x=merge_df['Cluster'], y=merge_df['Recency'])
#plt.show()

# Based on the boxplots, we can see that the clusters are well-separated and the features are good for clustering

# Calculate the mean of each feature for each cluster
cluster_mean = merge_df.groupby('Cluster').mean()
# The mean of the Total Purchase, Frequency, and Recency for each cluster is shown below
#            Total Purchase  Frequency     Recency
# Cluster
#2            5468.123007   12.498834   28.060606
#1            1035.192275   3.142699    53.663216
#0            439.317605    1.439219    264.530612

### Analysis of the Clusters
# Based on the mean values of the clusters and Figure 5, we can see that:
# Cluster 2 has the highest mean Total Purchase and Frequency, and the lowest mean Recency
# Cluster 0 has the lowest mean Total Purchase and Frequency, and the highest mean Recency
# Cluster 1 has the second lowest mean Total Purchase and Frequency, and the second highest mean Recency

# Therefore, we can label the clusters as follows:
# Cluster 2: High Value Customers who have made the highest Total Purchase and Frequency, and the lowest Recency
# Cluster 0: Low Value Customers who have made the lowest Total Purchase and Frequency, and the highest Recency
# Cluster 1: Medium Value Customers who have made the second lowest Total Purchase and Frequency, and the second highest Recency
# This information can be used to create targeted marketing strategies for each cluster of customers

# Save the DataFrame to a CSV file

cluster_mean.to_csv('/Users/hamid/Documents/VS Code/OnlineRetailClustersMean.csv', index=False)
merge_df.to_csv('/Users/hamid/Documents/VS Code/OnlineRetailClusters.csv', index=False)











