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

inputpath = sys.argv[1]

# Loading spaCy en version as spaCy (Object)
spaCy = spacy.load('en')

# For Logs purpose
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

# Load Custom rows from Train Data - Say it as Total data
print("*** Reading Data (100,000 rows) ***")
train = pd.read_csv(inputpath, encoding="utf-8")[:1000]
print(train.head())
print("*** Pre processing the Data ***")
# Preparing Scalar for Preprocessing the input

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
# Training the Model Using Spacy

print("*** Spacy Piping in Progress ***")
# Process texts as a stream for Question 1
vecq1 = [document.vector for document in spaCy.pipe(train['question1'], n_threads=40)]
vecq1 = np.array(vecq1)

# train['feature_q1'] = list(vecq1)

# Process texts as a stream for Question 2
vecq2 = [document.vector for document in spaCy.pipe(train['question2'], n_threads=40)]
vecq2 = np.array(vecq2)

# Creating Features
difference = (vecq2 - vecq1)
sums = (vecq2 + vecq1)

# Length of Questions
counts = pd.DataFrame(pd.concat([train['question1'], train['question2']])).apply(lambda temp: len(str(temp).split()))

# Preparing Feed for Deep Network
# Difference vector which contains manhattan distance between two sentences
vect_train = pd.DataFrame(difference, sums)
vect_train.append(counts, ignore_index=True)

# Pre processing the feed for Deep Network
scaler.fit(vect_train)
vect_train = scaler.transform(vect_train)

print("*** Pre processed Data ***")
print(pd.DataFrame(vect_train).head())
print("Buidling Model using Train Data ")
# Creating Deep Neural Network using scikit with 5 deep layers with 11, 15, 29, 30, 21 neurons each and relu

model = MLPClassifier(hidden_layer_sizes=(11, 15, 29, 30, 21), batch_size='auto', alpha=0.0001, beta_1=0.9,
                      beta_2=0.999, early_stopping=False, epsilon=1e-08,
                      learning_rate='constant', learning_rate_init=0.01, max_iter=200, random_state=None,
                      solver='adam', tol=0.0001, validation_fraction=0.001,
                      verbose=False, warm_start=False, activation='relu')

print("Training the Model")
# Training the model
model.fit(vect_train, train['is_duplicate'])

# Testing

# Prediction is made with a cross validation of 10, i.e. 9 sets for training and 1 set for testing
print("Validation of Model in progress")
predictions = cross_val_predict(model, vect_train, train['is_duplicate'], cv=10)

# predictions = model.predict(vect_test)
print("*** Confusion Matrix ***")
print(confusion_matrix(train['is_duplicate'], predictions))
print("*** Classification Report ***")
print(classification_report(train['is_duplicate'], predictions))

accuracy = accuracy_score(train['is_duplicate'], predictions)
print('Accuracy of the Model is: ', accuracy)

# preparing for ROC plot
fpr, tpr, threshold = roc_curve(train['is_duplicate'], predictions)

plt.title('Receiver Operation Characteristic (ROC)')
roc_auc = auc(fpr, tpr)
print('Area under the ROC curve of the Model is: ', roc_auc)

plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc, )
plt.plot([0, 1], [0, 1], 'r--')
plt.ylim([-0.2, 1.1])
plt.xlim([-0.2, 1.1])

plt.legend(loc='upper left')

plt.ylabel('True Positive Rate(TPR)')
plt.xlabel('False Positive Rate(FPR)')

print("***** Completed !!! *****")
plt.show()