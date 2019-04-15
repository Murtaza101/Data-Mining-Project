#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from os import path


# In[3]:


from PIL import Image


# In[4]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[5]:


import matplotlib.pyplot as plt


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


import warnings
warnings.filterwarnings("ignore")


# In[12]:


df = pd.read_csv(r"C:\Users\ZASS\Downloads\wine-reviews\winemag-data-130k-v2.csv")


# In[13]:


df.head()


# In[14]:


print("There are {} observations and {} features in this dataset. \n".format(df.shape[0],df.shape[1]))


# In[15]:


print("There are {} types of wine in this dataset such as {}... \n".format(len(df.variety.unique()),
                                                                           ", ".join(df.variety.unique()[0:5])))

print("There are {} countries producing wine in this dataset such as {}... \n".format(len(df.country.unique()),
                                                                                      ", ".join(df.country.unique()[0:5])))


# In[16]:


df[["country", "description","points"]].head()


# In[17]:


# Groupby by country
country = df.groupby("country")

# Summary statistic of all countries
country.describe().head()


# In[18]:


country.mean().sort_values(by="points",ascending=False).head()


# In[19]:


plt.figure(figsize=(15,10))
country.size().sort_values(ascending=False).plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Country of Origin")
plt.ylabel("Number of Wines")
plt.show()


# In[20]:


plt.figure(figsize=(15,10))
country.max().sort_values(by="points",ascending=False)["points"].plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Country of Origin")
plt.ylabel("Highest point of Wines")
plt.show()


# In[21]:


get_ipython().run_line_magic('pinfo', 'WordCloud')


# In[22]:


# Start with one review:
text = df.description[0]

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[23]:


wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[24]:


df


# In[ ]:




