# Online Retail Clustering

This project aims to perform clustering analysis on online retail data to identify patterns and segments within the customer base. By grouping similar customers together, we can gain insights into their purchasing behavior and tailor marketing strategies accordingly.


## Table of Contents

- [Introduction](#introduction)
- [Data Cleaning](#data-cleaning)
- [Data Preprocessing](#data-preprocessing)
- [Detecting Outliers](#detecting-outliers)
- [Feature Scaling and Standardization](#feature-scaling-and-standardization)
- [K-Means Clustering](#k-means-clustering)
- [Analysis of the Clusters](#analysis-of-the-clusters)
- [Results and Visualizations](#results-and-visualizations)
- [Saving the Results](#saving-the-results)
- [Usage](#usage)
- [License](#license)
- [Dataset Citation](#dataset-citation)

## Introduction

This project involves the analysis of an online retail dataset to understand customer behavior and segment customers based on their purchasing patterns. The segmentation is performed using K-Means clustering, and the results provide insights into different customer segments.

## Data Cleaning

The dataset undergoes a thorough cleaning process, handling missing values, changing data types, and removing rows with negative values in the Quantity and UnitPrice columns.

## Data Preprocessing

Features for clustering are created by aggregating each customer's Total Purchase and Frequency of purchases. The Recency of each customer is calculated by determining the time since their last transaction.

## Detecting Outliers

Outliers are identified in the Total Purchase, Frequency, and Recency columns using boxplots, and rows containing outliers are removed from the dataset using IQR. 

## Feature Scaling and Standardization

The features are standardized using the StandardScaler from scikit-learn to ensure that each feature contributes equally to the clustering process.

## K-Means Clustering

K-Means clustering is applied to segment customers into distinct groups based on their Total Purchase, Frequency, and Recency. The optimal number of clusters is determined using the Elbow Method.

## Analysis of the Clusters

Clusters are analyzed by calculating the mean values of Total Purchase, Frequency, and Recency for each cluster. The results provide insights into different customer segments.

## Results and Visualizations

The clusters are visualized using scatter plots, and boxplots highlight the differences in Total Purchase, Frequency, and Recency among the clusters.

## Saving the Results

The final cluster means and the entire segmented dataset are saved to CSV files. The trained K-Means model is also saved for future use.

## Usage

To run the project, ensure you have the required dependencies installed. The project can be executed by running the provided Python script.

## Dataset Citation
The dataset for this project is obtained UC Irvine Machine Learning Repository. 

Chen,Daqing. (2015). Online Retail. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33.


```bash
python Online_Retail_Clustering.py

# Online Retail Clustering Requirements

- pandas==1.3.3
- numpy==1.21.2
- scikit-learn==0.24.2
- matplotlib==3.4.3

