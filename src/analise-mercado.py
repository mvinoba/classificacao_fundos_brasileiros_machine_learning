#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


file = pd.ExcelFile('../data/Anexo-Boletim-FI-202303.xlsx')


# In[3]:


file.sheet_names  


# In[4]:


df_pl = pd.read_excel(
    '../data/Anexo-Boletim-FI-202303.xlsx', 
    sheet_name=['Pág. 2 - PL Total Geral '],
)['Pág. 2 - PL Total Geral ']


# In[5]:


df_totalfundos = pd.read_excel(
    '../data/Anexo-Boletim-FI-202303.xlsx', 
    sheet_name=['Pág. 13 - N° de Fundos'],
)['Pág. 13 - N° de Fundos']


# In[6]:


df_cotistas = pd.read_excel(
    '../data/Anexo-Boletim-FI-202303.xlsx', 
    sheet_name=['Pág. 14 - N° de Contas'],
)['Pág. 14 - N° de Contas']


# In[7]:


df_pl = df_pl.iloc[6:71].rename({'ANBIMA » Fundos de Investimento | Relatórios': 'date', 'Unnamed: 2': 'Patrimônio Líquido'}, axis=1)[[
    'date', 'Patrimônio Líquido'
]]#.set_index('date')


# In[8]:


df_pl['Patrimônio Líquido'] = df_pl['Patrimônio Líquido'].astype(float)
df_pl['date'] = pd.to_datetime(df_pl['date'])


# In[9]:


df_pl = df_pl[~df_pl['date'].isin(['2022-01-01', '2022-02-01',
       '2022-03-01', '2022-04-01',
       '2022-05-01', '2022-06-01',
       '2022-07-01', '2022-08-01',
       '2022-09-01', '2022-10-01',
       '2022-11-01', '2023-01-01', '2023-02-01', '2023-03-01'])]


# In[10]:


df_pl['Patrimônio Líquido'] = df_pl['Patrimônio Líquido'] / 1_000_000


# In[11]:


df_pl['date'] = df_pl['date'].dt.year


# In[12]:


import matplotlib.ticker as mtick


# In[13]:


df_pl[['date', 'Patrimônio Líquido']].iloc[30:]


# In[14]:


fig, ax = plt.subplots(1, 1, figsize=(20, 10))
fmt = 'R$ {x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick) 

plt.bar(df_pl['date'].iloc[30:], df_pl['Patrimônio Líquido'].iloc[30:])
ax.set_xticks(df_pl['date'].iloc[30:])
plt.xticks(fontsize=14, rotation=25)
plt.yticks(fontsize=14)
#plt.show()
plt.savefig('PatrimonioLiquido.eps')


# In[ ]:





# In[15]:


df_totalfundos


# In[16]:


df_totalfundos = df_totalfundos.iloc[5:35].rename({'ANBIMA » Fundos de Investimento | Relatórios': 'date', 'Unnamed: 13': 'Total'}, axis=1)[[
    'date', 'Total'
]]#.set_index('date')


# In[17]:


df_totalfundos['Total'] = df_totalfundos['Total'].astype(float)
df_totalfundos['date'] = pd.to_datetime(df_totalfundos['date'])


# In[18]:


df_totalfundos = df_totalfundos[~df_totalfundos['date'].isin(['2022-01-01', '2022-02-02',
       '2022-03-02', '2022-04-01',
       '2022-05-01', '2022-06-01',
       '2022-07-01', '2022-08-01',
       '2022-09-01', '2022-10-01',
       '2022-11-01', '2023-01-01', '2023-02-01', '2023-03-01'])]


# In[19]:


df_totalfundos['date'] = df_totalfundos['date'].dt.year


# In[20]:


df_totalfundos


# In[21]:


fig, ax = plt.subplots(1, 1, figsize=(20, 10))
#fmt = 'R$ {x:,.0f}'
#tick = mtick.StrMethodFormatter(fmt)
# ax.yaxis.set_major_formatter(tick) 

plt.bar(df_totalfundos['date'], df_totalfundos['Total'])
ax.set_xticks(df_totalfundos['date'])
plt.xticks(fontsize=14, rotation=25)
plt.yticks(fontsize=14)
#plt.show()
plt.savefig('QuantidadeFundos.eps')


# In[22]:


df_cotistas


# In[23]:


df_cotistas = df_cotistas.iloc[5:34].rename({'ANBIMA » Fundos de Investimento | Relatórios': 'date', 'Unnamed: 14': 'Total'}, axis=1)[[
    'date', 'Total'
]]#.set_index('date')


# In[24]:


df_cotistas['Total'] = df_cotistas['Total'].astype(float)
df_cotistas['date'] = pd.to_datetime(df_cotistas['date'])


# In[25]:


df_cotistas = df_cotistas[~df_cotistas['date'].isin(['2022-01-01', '2022-02-01',
       '2022-03-01', '2022-04-01',
       '2022-05-01', '2022-06-01',
       '2022-07-01', '2022-08-01',
       '2022-09-01', '2022-10-01',
       '2022-11-01', '2023-01-01'])]


# In[26]:


df_cotistas['Total'] = df_cotistas['Total'] / 1_000_000


# In[27]:


df_cotistas['date'] = df_cotistas['date'].dt.year


# In[28]:


fig, ax = plt.subplots(1, 1, figsize=(20, 10))
#fmt = 'R$ {x:,.0f}'
#tick = mtick.StrMethodFormatter(fmt)
# ax.yaxis.set_major_formatter(tick) 

plt.bar(df_cotistas['date'], df_cotistas['Total'])
ax.set_xticks(df_cotistas['date'])
plt.xticks(fontsize=16, rotation=25)
plt.yticks(fontsize=16)
#plt.show()
plt.savefig('NumeroCotistas.eps')


# In[ ]:




