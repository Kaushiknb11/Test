#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:06:58 2023

@author: kaushiknarasimha
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

# Set page title
st.set_page_config(page_title="Support Vector Machines")

# Display title
st.title("Support Vector Machines")

# Generate synthetic data with blobs
X, y = make_blobs(n_samples=200, centers=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to plot the SVM decision boundary
def plot_svm(C=1.0, kernel='linear'):
    clf = SVC(C=C, kernel=kernel)
    clf.fit(X_train, y_train)

    # Create a meshgrid of points to plot the decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict the labels for each point in the meshgrid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and the training points
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(f'SVM Decision Boundary (C={C}, kernel={kernel})')
    st.pyplot()

# Define interactive sliders for SVM hyperparameters
C_slider = st.slider('C:', min_value=0.1, max_value=10.0, step=0.1, value=1.0)
kernel_dropdown = st.selectbox('Kernel:', ['linear', 'poly', 'rbf', 'sigmoid'], index=0)

# Create an interactive widget
plot_svm(C_slider, kernel_dropdown)
