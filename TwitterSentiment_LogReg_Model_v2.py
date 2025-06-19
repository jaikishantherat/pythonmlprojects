#!/usr/bin/env python
# coding: utf-8

# ## TwitterSentiment_LogReg_Model_v2
# 
# New notebook

# In[16]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# In[17]:


# 2. Load your cleaned dataset (already in Lakehouse or Data Wrangler)
df = spark.read.table("cleaned_twitter_table").toPandas()


# In[18]:


# 3. Split the data
X = df[["clean_text"]]
y = df["category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


# 4. Save original test text for later (optional but useful)
X_test_raw = X_test.copy()


# In[20]:


# 5. Vectorize the text
vectorizer = TfidfVectorizer(max_features=500)
X_train_vec = vectorizer.fit_transform(X_train["clean_text"])
X_test_vec = vectorizer.transform(X_test["clean_text"])


# In[21]:


# 6. Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)


# In[22]:


# 7. Make predictions
predictions = model.predict(X_test_vec)


# In[23]:


# 8. Create predictions DataFrame
predictions_df = pd.DataFrame({
    "clean_text": X_test_raw["clean_text"].values,
    "predicted_sentiment": predictions
})


# In[24]:


# 9. Convert to Spark DataFrame
predictions_spark_df = spark.createDataFrame(predictions_df)


# In[25]:


# 10. Save to Lakehouse
predictions_spark_df.write.mode("overwrite").saveAsTable("predictions_sentiment_nlp")


# In[26]:


from sklearn.metrics import classification_report

# Get the report as a dictionary
report = classification_report(y_test, predictions, output_dict=True)

print(f"{'Feeling':<10} | {'Precision (How often right?)':<30} | {'Recall (Caught all correct?)':<30} | {'F1-Score (Balance)':<20}")
print("-" * 100)

for label in ['-1', '1', '10']:
    feeling = label
    precision = round(report[label]['precision'], 2)
    recall = round(report[label]['recall'], 2)
    f1 = round(report[label]['f1-score'], 2)
    print(f"{feeling:<10} | {precision:<30} | {recall:<30} | {f1:<20}")

