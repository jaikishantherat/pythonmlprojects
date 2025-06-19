#!/usr/bin/env python
# coding: utf-8

# ## Iris
# 
# New notebook

# In[1]:


# Step 1: Load the Iris dataset
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Step 2: Train/Test Split and Model Training
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 3: Create Results DataFrame with Readable Labels
label_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

results_df_named = pd.DataFrame({
    'Actual': [label_map[i] for i in y_test],
    'Predicted': [label_map[i] for i in y_pred]
})

# Step 4: Save to Microsoft Fabric Lakehouse
import os
import joblib

# Fabric Lakehouse path
lakehouse_path = '/lakehouse/default/Files/iris'
os.makedirs(lakehouse_path, exist_ok=True)

# Save predictions
results_df_named.to_parquet(f'{lakehouse_path}/iris_predictions.parquet', index=False)

# Save trained model too
joblib.dump(model, f'{lakehouse_path}/iris_knn_model.pkl')

