#!/usr/bin/env python
# coding: utf-8

# ## TwitterSentimentCleaning
# 
# New notebook

# In[9]:


# Welcome to your new notebook
# Type here in the cell editor to add code!
import pandas as pd

# Step 1: Load the raw CSV file
df = pd.read_csv("/lakehouse/default/Files/Twitter_Data.csv")


# In[10]:


# Step 2: Clean text column — strip whitespace, remove empty rows
df["clean_text"] = df["clean_text"].astype(str).str.strip()
df = df[df["clean_text"].str.len() > 1]


# In[11]:


# Step 3: Drop missing values in sentiment category
df = df.dropna(subset=["category"])


# In[12]:


# Step 4: Convert to integer type
df["category"] = df["category"].astype(int)


# In[13]:


# Step 5: Replace 0 with 10 (for neutral tweets)
df["category"] = df["category"].replace(0, 10)


# In[14]:


# Step 6: Save the cleaned DataFrame to a new Parquet file
df.to_parquet("/lakehouse/default/Files/cleaned_twitter_data.parquet", index=False)


# In[15]:


# Show sample output
df.head()


# In[16]:


# Load from Parquet file in Files folder
df = spark.read.parquet("Files/cleaned_twitter_data.parquet")

# Write to Lakehouse as a managed table
df.write.mode("overwrite").saveAsTable("cleaned_twitter_table")

