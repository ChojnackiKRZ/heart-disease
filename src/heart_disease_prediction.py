# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 19:25:43 2023

@author: krzys
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import metrics

heart_data = pd.read_csv('data/raw/heart_disease_data.csv')

heart_data.describe()

x = heart_data.copy().drop(columns = 'target')
y = heart_data.copy()['target']

X_train, X_test, y_train, y_test = train_test_split(x, y, stratify = y)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()
