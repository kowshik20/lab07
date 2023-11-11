#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#for dataset1


# In[2]:


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
 


# In[3]:


#A2
 
# Testing the accuracy of the SVM on the test set
accuracy = clf.score(X_test[['f0', 'f3']], y_test)
print(f"Accuracy of the SVM on the test set: {accuracy}")
 
# Perform classification for the given test vector
test_vector = X_test[['f0', 'f3']].iloc[0]
predicted_class = clf.predict([test_vector])
print(f"The predicted class for the test vector: {predicted_class}")


# In[10]:


# Assuming you have already trained your SVC classifier 'clf' and you have a test set 'X_test'
# X_test is the feature matrix for your test set
features = data[['f0', 'f3']]
target = data['Label']
 
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# Make predictions on the test set
predictions = clf.predict(X_test)

# Now you can study the output values of the classifier
print("Predictions:", predictions)

# If you want to relate the output value to the class value predicted, you can print the corresponding class labels
for i, prediction in enumerate(predictions):
    print(f"Sample {i + 1}: Predicted class {prediction}")

# To test the accuracy, you'll need the true class labels for the test set, assuming 'y_test' contains the true labels
accuracy = sum(predictions == y_test) / len(y_test)
print("Accuracy:", accuracy)


# In[6]:


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assume you have your features in X and labels in y
# Split the data into training and testing sets
features = data[['f0', 'f3']]
target = data['Label']
 
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# List of kernel functions to experiment with
kernel_functions = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel in kernel_functions:
    # Create and train the Support Vector Classifier with the current kernel
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train, y_train)

    # Use the trained classifier to predict the labels of the test set
    y_pred = clf.predict(X_test)

    # Calculate and print the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Kernel: {kernel}, Accuracy: {accuracy * 100:.2f}%")


# In[ ]:


#for dataset 2


# In[16]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
 
# Load the dataset
data = pd.read_excel('modified_dataset.xlsx')
 
# 'embed_0' and 'embed_1' are the features and 'Label' is the target variable
features = data[['Theta0_Lambda1_MeanAmplitude', 'Theta0_Lambda1_LocalEnergy']]
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
 


# In[17]:


#A2
 
# Testing the accuracy of the SVM on the test set
accuracy = clf.score(X_test[['Theta0_Lambda1_MeanAmplitude', 'Theta0_Lambda1_LocalEnergy']], y_test)
print(f"Accuracy of the SVM on the test set: {accuracy}")
 
# Perform classification for the given test vector
test_vector = X_test[['Theta0_Lambda1_MeanAmplitude', 'Theta0_Lambda1_LocalEnergy']].iloc[0]
predicted_class = clf.predict([test_vector])
print(f"The predicted class for the test vector: {predicted_class}")


# In[18]:


# Assuming you have already trained your SVC classifier 'clf' and you have a test set 'X_test'
# X_test is the feature matrix for your test set
features = data[['Theta0_Lambda1_MeanAmplitude', 'Theta0_Lambda1_LocalEnergy']]
target = data['Label']
 
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# Make predictions on the test set
predictions = clf.predict(X_test)

# Now you can study the output values of the classifier
print("Predictions:", predictions)

# If you want to relate the output value to the class value predicted, you can print the corresponding class labels
for i, prediction in enumerate(predictions):
    print(f"Sample {i + 1}: Predicted class {prediction}")

# To test the accuracy, you'll need the true class labels for the test set, assuming 'y_test' contains the true labels
accuracy = sum(predictions == y_test) / len(y_test)
print("Accuracy:", accuracy)


# In[ ]:


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assume you have your features in X and labels in y
# Split the data into training and testing sets
features = data[['Theta0_Lambda1_MeanAmplitude', 'Theta0_Lambda1_LocalEnergy']]
target = data['Label']
 
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# List of kernel functions to experiment with
kernel_functions = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel in kernel_functions:
    # Create and train the Support Vector Classifier with the current kernel
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train, y_train)

    # Use the trained classifier to predict the labels of the test set
    y_pred = clf.predict(X_test)

    # Calculate and print the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Kernel: {kernel}, Accuracy: {accuracy * 100:.2f}%")


# In[ ]:




