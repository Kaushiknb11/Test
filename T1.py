#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:50:59 2023

@author: kaushiknarasimha
"""

import streamlit as st
import datetime
import time
import base64
from pathlib import Path
import plotly.express as px

import plotly.figure_factory as ff

import numpy as np
from scipy.spatial import Delaunay

import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import make_moons, make_circles
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Set page title
st.set_page_config(page_title="Support Vector Machines")

# Display title
st.title("Support Vector Machines")

# Sidebar for choosing kernel
kernel = st.sidebar.selectbox('Choose Kernel', ['Polynomial', 'Radial Bias', 'Linear'], key='kernels')

# Sidebar for data points distribution
n_samples = st.sidebar.slider("Samples", 2, 100, value=40)
noise = st.sidebar.slider("Noise", 0.01, 1.00)

# Create data based on kernel
if kernel == 'Polynomial':
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
    ])
else:
    X, y = make_blobs(n_samples=n_samples, noise=noise, random_state=42)
    if kernel == "Linear":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("linear_svc", SVC(kernel="linear", C=10, random_state=42))
        ])
    else:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=0.1, C=0.1, random_state=42))
        ])

# Train SVM classifier
clf.fit(X, y)

# Prediction and decision function
x0s = np.linspace(-1.5, 2.5, 100)
x1s = np.linspace(-1.0, 1.5, 100)
x0, x1 = np.meshgrid(x0s, x1s)
X_concat = np.c_[x0.ravel(), x1.ravel()]
y_prediction = clf.predict(X_concat).reshape(x0.shape)
y_decision = clf.decision_function(X_concat).reshape(x0.shape)

# Plotting section
fig = plt.figure(figsize=(10, 10))
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bo")
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "rs")
plt.axis([-1.5, 2.5, -1, 1.5])
plt.grid(True, which='both')
plt.xlabel(r"$x_1$", fontsize=20)
plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
plt.contourf(x0, x1, y_prediction, cmap="Blues", alpha=0.5)
plt.contourf(x0, x1, y_decision, cmap="Blues", alpha=0.26)

# Streamlit pyplot show
st.pyplot(fig)
