#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
 
# Load the dataset
data = pd.read_excel('Custom_CNN_Features1.xlsx')
 
# 'embed_0' and 'embed_1' are the features and 'Label' is the target variable
features = data[['f0', 'f3']]
target = data['Label']
 
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
 
# Initialize and train the Support Vector Machine (SVM) model
clf = SVC()
clf.fit(X_train, y_train)
 
# Get the support vectors
support_vectors = clf.support_vectors_
 
# Print the support vectors
print(f'Support Vectors ={support_vectors}')
 


# In[6]:


#A2
 
# Testing the accuracy of the SVM on the test set
accuracy = clf.score(X_test[['f0', 'f3']], y_test)
print(f"Accuracy of the SVM on the test set: {accuracy}")
 
# Perform classification for the given test vector
test_vector = X_test[['f0', 'f3']].iloc[0]
predicted_class = clf.predict([test_vector])
print(f"The predicted class for the test vector: {predicted_class}")


# In[8]:


decision_values = clf.decision_function(X_test[['f0', 'f3']])
 
# Relate the decision values to the class values
predictions = clf.predict(X_test[['f0', 'f3']])
 
# Test the accuracy using your own logic for class determination
# Here, we'll simply compare decision values against zero for binary classification
# Adjust this logic based on the specifics of your classification problem
 
correct_predictions = 0
for i in range(len(predictions)):
    if (predictions[i] == 1 and decision_values[i] > 0) or (predictions[i] == 0 and decision_values[i] < 0):
        correct_predictions += 1
 
accuracy = correct_predictions / len(y_test)
print(f"Accuracy using decision values: {accuracy}")


# In[ ]:




