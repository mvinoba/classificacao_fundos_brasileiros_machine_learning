#!/usr/bin/env python
# coding: utf-8

# In[42]:


from seleniumwire import webdriver
from io import BytesIO
import pandas as pd
import json
import time
import gzip


# In[43]:


# Set up the selenium webdriver
options = webdriver.ChromeOptions()
# options.add_argument('--headless')
# options.add_argument('--disable-gpu')

driver = webdriver.Chrome(options=options, seleniumwire_options={'mitm_http2': False})
driver.implicitly_wait(10)


# In[44]:


driver.get('https://data.anbima.com.br/fundos?page=1&size=100&')

# extract data
def extract_data_from_response(response):
    try:
        # Decompress
        gzip_file = BytesIO(response.body)
        with gzip.GzipFile(fileobj=gzip_file) as decompressed:
            response_data = json.loads(decompressed.read().decode('utf-8'))
        return response_data['content']
    except json.JSONDecodeError:
        print("Error decoding JSON")
        return []

# get number of pages
def get_total_pages():
    for request in driver.requests:
        if request.response and 'https://data-back.anbima.com.br/fundos-bff/fundos?page=0&size=100&' in request.url:
            gzip_file = BytesIO(request.response.body)
            with gzip.GzipFile(fileobj=gzip_file) as decompressed:
                response_data = json.loads(decompressed.read().decode('utf-8'))
            return response_data['total_pages']
    return 0


# In[45]:


# find the total number of pages
total_pages = get_total_pages()
print(f"Total Pages: {total_pages}")


# In[46]:


# Initialize
all_contents = []

# Loop
i = 0  # debug
for page in range(total_pages):
    driver.get(f'https://data.anbima.com.br/fundos?page={page + 1}&size=100&')
    time.sleep(10)
    print(i)
    for request in driver.requests:
        if request.response and f'https://data-back.anbima.com.br/fundos-bff/fundos?page={page}&size=100&' in request.url:
            page_contents = extract_data_from_response(request.response)
            all_contents.extend(page_contents)
            break
    i += 1


# In[ ]:


all_contents


# In[40]:


pd.DataFrame(all_contents).to_csv('all_funds_df.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




