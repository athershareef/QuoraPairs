Language Used: Python 


C:\Users\ather>python --version
Python 3.6.0
Tool Used: PyCharm

Libraries required

#Imports
import sys
import pandas as pd
import numpy as np
import logging
import spacy
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict

Code is available in code folder

How to run the Code?
1.Navigate to folder where QuoraQuestionPairs.py is available using command prompt and run the command as below. It contains 1 parameter i.e. Data folder with file name
	python QuoraQuestionPairs.py "C:/Users/ather/Desktop/train.csv"
2.Observe the plot and results on console