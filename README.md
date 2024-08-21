# K-Means Clustering on Iris Dataset

This repository contains a Python project that performs K-Means clustering on the Iris dataset. The project involves finding the optimal number of clusters and visualizing the results.

## Project Overview

The objective of this project is to apply the K-Means clustering algorithm to the Iris dataset to identify distinct clusters within the data. Key steps include:

1. **Loading the Dataset**: The Iris dataset is loaded from a CSV file.
2. **Determining Optimal Clusters**:
   - **Elbow Method**: Helps to find the optimal number of clusters by plotting inertia.
   - **Silhouette Score**: Evaluates the quality of clustering for different numbers of clusters.
3. **Fitting the Model**: The K-Means model is fitted with the optimal number of clusters.
4. **Visualizing Clusters**: Clusters are visualized to interpret the results.

## Project Structure

- `iris.csv`: The dataset used for clustering.
- `predict.py`: The main script for performing clustering, determining optimal clusters, and visualizing results.

## Installation

To run this project, you need to have Python installed. You also need to install the following packages:

```bash
pip install numpy pandas matplotlib scikit-learn
