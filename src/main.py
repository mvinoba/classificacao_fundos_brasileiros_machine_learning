#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import glob
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from matplotlib.collections import QuadMesh
import itertools
import datetime


# In[77]:


import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from matplotlib.collections import QuadMesh


def get_new_fig(fn, figsize=[9, 9]):
    """Init graphics"""
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()  # Get Current Axis
    ax1.cla()  # clear existing plot
    return fig1, ax1


def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
    """
    config cell text and colors
    and return text elements to add and to delete
    """
    text_add = []
    text_del = []

    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:, col]
    ccl = len(curr_column)

    # last line and/or last column
    if (col == (ccl - 1)) or (lin == (ccl - 1)):
        # totals and percents
        if cell_val != 0:
            if (col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif col == ccl - 1:
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif lin == ccl - 1:
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ["%.2f%%" % (per_ok), "100%"][per_ok == 100]

        # text to delete
        text_del.append(oText)

        # text to add
        font_prop = fm.FontProperties(weight="bold", size=fz)
        text_kwargs = dict(color="w", ha="center", va="center", gid="sum", fontproperties=font_prop)
        lis_txt = ["%d" % (cell_val), per_ok_s, "%.2f%%" % (per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy()
        dic["color"] = "g"
        lis_kwa.append(dic)
        dic = text_kwargs.copy()
        dic["color"] = "r"
        lis_kwa.append(dic)
        lis_pos = [(oText._x, oText._y - 0.3), (oText._x, oText._y), (oText._x, oText._y + 0.3)]
        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
            text_add.append(newText)

        # set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if (col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr

    else:
        if per > 0:
            txt = "%s\n%.2f%%" % (cell_val, per)
        else:
            txt = "" if show_null_values == 0 else "0" if show_null_values == 1 else "0\n0.0%"
        oText.set_text(txt)

        # main diagonal
        if col == lin:
            # set color of the text in the diagonal to black
            oText.set_color("black")
            # set background color in the diagonal based on face color
            clr = facecolors[posi]
            if sum(clr[:3]) > 1.5:  # light background
                facecolors[posi] = [0.95, 0.95, 0.95, 1.0]  # set background color to a light shade of gray
            else:  # dark background
                facecolors[posi] = [0.35, 0.35, 0.35, 1.0]  # set background color to a dark shade of gray
        else:
            oText.set_color("r")

    return text_add, text_del




def insert_totals(df_cm):
    """insert total column and line (the last ones)"""
    sum_col = []
    for c in df_cm.columns:
        sum_col.append(df_cm[c].sum())
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append(item_line[1].sum())
    df_cm["sum_lin"] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc["sum_col"] = sum_col


def pp_matrix_modified(
    df_cm,
    annot=True,
    cmap="Oranges",
    fmt=".2f",
    fz=11,
    lw=0.5,
    cbar=False,
    figsize=[8, 8],
    show_null_values=0,
    pred_val_axis="y",
    save_fig=False
):
    """
    print conf matrix with default layout (like matlab)
    params:
      df_cm          dataframe (pandas) without totals
      annot          print text in each cell
      cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
      fz             fontsize
      lw             linewidth
      pred_val_axis  where to show the prediction values (x or y axis)
                      'col' or 'x': show predicted values in columns (x axis) instead lines
                      'lin' or 'y': show predicted values in lines   (y axis)
    """
    if pred_val_axis in ("col", "x"):
        xlbl = "Previsto"
        ylbl = "Real"
    else:
        xlbl = "Real"
        ylbl = "Previsto"
        df_cm = df_cm.T

    # create "Total" column
    insert_totals(df_cm)

    # this is for print allways in the same window
    fig, ax1 = get_new_fig("Conf matrix default", figsize)

    ax = sn.heatmap(
        df_cm,
        annot=annot,
        annot_kws={"size": fz},
        linewidths=lw,
        ax=ax1,
        cbar=cbar,
        cmap=cmap,
        linecolor="w",
        fmt=fmt,
    )

    # set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    # iter in text elements
    array_df = np.array(df_cm.to_records(index=False).tolist())
    text_add = []
    text_del = []
    posi = -1  # from left to right, bottom to top.
    for t in ax.collections[0].axes.texts:  # ax.texts:
        pos = np.array(t.get_position()) - [0.5, 0.5]
        lin = int(pos[1])
        col = int(pos[0])
        posi += 1

        # set text
        txt_res = configcell_text_and_colors(
            array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values
        )

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    # remove the old ones
    for item in text_del:
        item.remove()
    # append the new ones
    for item in text_add:
        ax.text(item["x"], item["y"], item["text"], **item["kw"])

    # titles and legends
    ax.set_title("")
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    plt.tight_layout()  # set layout slim
    
    if save_fig:
        plt.savefig(save_fig)
    else:
        plt.show()


def pp_matrix_from_data(
    y_test,
    predictions,
    columns=None,
    annot=True,
    cmap="Oranges",
    fmt=".2f",
    fz=11,
    lw=0.5,
    cbar=False,
    figsize=[8, 8],
    show_null_values=0,
    pred_val_axis="lin",
):
    """
    plot confusion matrix function with y_test (actual values) and predictions (predic),
    whitout a confusion matrix yet
    """
    from pandas import DataFrame
    from sklearn.metrics import confusion_matrix

    # data
    if not columns:
        from string import ascii_uppercase

        columns = [
            "class %s" % (i)
            for i in list(ascii_uppercase)[0 : len(np.unique(y_test))]
        ]

    confm = confusion_matrix(y_test, predictions)
    df_cm = DataFrame(confm, index=columns, columns=columns)
    pp_matrix(
        df_cm,
        fz=fz,
        cmap=cmap,
        figsize=figsize,
        show_null_values=show_null_values,
        pred_val_axis=pred_val_axis,
    )


# In[3]:


plt.rcParams["figure.figsize"] = (16,8)


# In[4]:


all_funds_df = pd.read_csv('all_funds_df.csv')
all_funds_df = all_funds_df.drop('Unnamed: 0', axis=1)

arquivos_carteira = glob.glob('../data/cda_fi_202306/cda_fi_BLC*')
df = pd.concat([pd.read_csv(arquivo, sep=';', encoding='latin-1', low_memory=False) for arquivo in arquivos_carteira])
df_cadastro = pd.read_csv("../data/cad_fi.csv", sep=';', encoding='latin-1', low_memory=False)
df_study = df.copy()
a = df_study.groupby(['CNPJ_FUNDO', 'TP_APLIC']).agg({'VL_MERC_POS_FINAL': 'sum'})
soma_pl_total = df_study.groupby('CNPJ_FUNDO').agg({'VL_MERC_POS_FINAL': 'sum'})
soma_pl_total.columns = ['pl_total']
b = pd.merge(a, soma_pl_total, left_index=True, right_index=True)
c = (b['VL_MERC_POS_FINAL']/b['pl_total']).reset_index()
c = c.rename({0: 'values'}, axis=1)
pivot = pd.pivot_table(c, values='values', index=["CNPJ_FUNDO"], columns="TP_APLIC", fill_value=0) 
pivot.index = pivot.reset_index()['CNPJ_FUNDO'].str.replace('\.|/|-', '', regex=True)
all_funds_df['cnpj_fundo'] = all_funds_df['cnpj_fundo'].dropna().apply(round).astype(str).apply(lambda x: x.zfill(14))
data = all_funds_df.merge(pivot, left_on='cnpj_fundo', right_index=True)

df_cadastro = df_cadastro[df_cadastro['TP_FUNDO'] == 'FI']
df_cadastro = df_cadastro.drop_duplicates('CNPJ_FUNDO')
df_cadastro['cnpj_fundo'] = df_cadastro['CNPJ_FUNDO'].str.replace('\.|/|-', '', regex=True)
data = data.merge(df_cadastro[['cnpj_fundo', 'FUNDO_COTAS']], on='cnpj_fundo', how='inner')

# teste tirando os fundos de previdencia
data = data[~data['tipo_anbima'].str.contains('Previdência')]
data['tipo_anbima'].value_counts().head(60)
min_count = 100
contagem_tipos = data['tipo_anbima'].value_counts()
contagem_tipos = contagem_tipos[(contagem_tipos >= min_count) == True].index.values

data = data[data['tipo_anbima'].isin(contagem_tipos)]

data = data[data['FUNDO_COTAS'] == 'N']


# In[5]:


data['tipo_anbima'].unique()


# In[6]:


sorted_tipo_anbima = sorted(data['tipo_anbima'].unique())


# In[7]:


sorted_tipo_anbima


# In[8]:


data['tipo_anbima'].value_counts().plot.barh()


# In[9]:


kv = {k:v for k, v in zip(sorted_tipo_anbima, range(len(data['tipo_anbima'].unique())))}
from pprint import pprint
pprint({v:k for k,v in kv.items()})
data['tipo_anbima'] = data['tipo_anbima'].apply(lambda x: kv[x])


# In[10]:


X = data[['Ações',
       'Ações e outros TVM cedidos em empréstimo',
       'Brazilian Depository Receipt - BDR',
       'Certificado ou recibo de depósito de valores mobiliários',
       'Compras a termo a receber', 'Cotas de Fundos',
       'Cotas de fundos de investimento - Instrução Nº 409',
       'Cotas de fundos de renda fixa', 'Cotas de fundos de renda variável',
       'DIFERENCIAL DE SWAP A PAGAR', 'DIFERENCIAL DE SWAP A RECEBER',
       'DISPONÍVEL DE OURO', 'Debêntures', 
       'Debêntures conversíveis',
       'Debêntures simples', 
       'Depósitos a prazo e outros títulos de IF',
       'Disponibilidades', 'Investimento no Exterior',
       'Mercado Futuro - Posições compradas',
       'Mercado Futuro - Posições vendidas',
       'Obrigações por ações e outros TVM recebidos em empréstimo',
       'Obrigações por compra a termo a pagar',
       'Obrigações por venda a termo a entregar', 'Operações Compromissadas',
       'Opções - Posições lançadas', 'Opções - Posições titulares',
       'Outras aplicações', 'Outras operações passivas e exigibilidades',
       'Outros valores mobiliários ofertados privadamente',
       'Outros valores mobiliários registrados na CVM objeto de oferta pública',
       'Títulos Públicos', 'Títulos de Crédito Privado',
       'Títulos ligados ao agronegócio', 'Valores a pagar',
       'Valores a receber', 'Vendas a termo a receber']].values
y = data['tipo_anbima'].values


# In[12]:


from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold, cross_val_score, GridSearchCV
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


# In[13]:


grid_n_estimators = list(range(10, 300+1, 10))
grid_criterion = ['gini', 'entropy']
param_grid = {
    'n_estimators': grid_n_estimators,
    'criterion': grid_criterion
}


# In[14]:


forest = RandomForestClassifier(
    random_state=1,
    n_jobs=2,
    class_weight='balanced'
)


# In[15]:


print(datetime.datetime.today())


# In[16]:


grid_clf = GridSearchCV(forest, param_grid, cv=5)
grid_clf.fit(X, y)


# In[17]:


print(datetime.datetime.today())


# In[18]:


grid_clf.best_params_


# In[19]:


cv_results = pd.DataFrame(grid_clf.cv_results_)


# In[20]:


cv_results.columns


# In[21]:


cv_results.sort_values('mean_test_score', ascending=False)


# In[22]:


df_gini = cv_results[cv_results['param_criterion'] == 'gini']
df_entropy = cv_results[cv_results['param_criterion'] == 'entropy']

# Plotting the lines
plt.errorbar(df_gini['param_n_estimators'], df_gini['mean_test_score'], 
             #yerr=df_gini['std_test_score'],
             label='gini', marker='o')
plt.errorbar(df_entropy['param_n_estimators'], df_entropy['mean_test_score'], 
             #yerr=df_entropy['std_test_score'],
             label='entropy', marker='o')

#plt.ylim(0.55, 0.61)
# Adding labels and title
plt.xlabel('n_estimators')
plt.ylabel('Acurácia')
# plt.title('Mean Test Score vs. Number of Estimators')
plt.legend()
# plt.grid(True)

# Show the plot
plt.savefig('gridsearch.eps')
plt.show()


# In[ ]:





# In[ ]:





# In[23]:


model = grid_clf.best_estimator_


# In[24]:


# Calculando métricas de score para o melhor modelo


# In[25]:


model


# In[26]:


model = RandomForestClassifier(class_weight='balanced', n_estimators=190, n_jobs=2,
                       random_state=1)


# In[27]:


acc = np.mean(
    cross_val_score(
        model, X, y, cv=5
    )
)


# In[28]:


f1_weighted = np.mean(
        cross_val_score(
        model,
        X, y,
        scoring=metrics.make_scorer(metrics.f1_score, average='weighted'),
        cv=5
    )
)


# In[29]:


balanced_acc = np.mean(
    cross_val_score(
        model, X, y, scoring=metrics.make_scorer(metrics.balanced_accuracy_score), cv=5
    )
)


# In[55]:


balanced_acc_adj = np.mean(
    cross_val_score(
        model, X, y, scoring=metrics.make_scorer(metrics.balanced_accuracy_score, adjusted=True), cv=5
    )
)


# In[56]:


print('Acurácia:',  acc)
print('Acurácia Balanceada:',  balanced_acc)
print('Acurácia Balanceada Ajustada:',  balanced_acc_adj)
print('F1 Weighted:',  f1_weighted)


# In[31]:


model


# In[32]:


model.fit(X, y)


# In[33]:


features = ['Ações',
       'Ações e outros TVM cedidos em empréstimo',
       'Brazilian Depository Receipt - BDR',
       'Certificado ou recibo de depósito de valores mobiliários',
       'Compras a termo a receber', 'Cotas de Fundos',
       'Cotas de fundos de investimento - Instrução Nº 409',
       'Cotas de fundos de renda fixa', 'Cotas de fundos de renda variável',
       'DIFERENCIAL DE SWAP A PAGAR', 'DIFERENCIAL DE SWAP A RECEBER',
       'DISPONÍVEL DE OURO', 'Debêntures', 
       'Debêntures conversíveis',
       'Debêntures simples', 
       'Depósitos a prazo e outros títulos de IF',
       'Disponibilidades', 'Investimento no Exterior',
       'Mercado Futuro - Posições compradas',
       'Mercado Futuro - Posições vendidas',
       'Obrigações por ações e outros TVM recebidos em empréstimo',
       'Obrigações por compra a termo a pagar',
       'Obrigações por venda a termo a entregar', 'Operações Compromissadas',
       'Opções - Posições lançadas', 'Opções - Posições titulares',
       'Outras aplicações', 'Outras operações passivas e exigibilidades',
       'Outros valores mobiliários ofertados privadamente',
       'Outros valores mobiliários registrados na CVM objeto de oferta pública',
       'Títulos Públicos', 'Títulos de Crédito Privado',
       'Títulos ligados ao agronegócio', 'Valores a pagar',
       'Valores a receber', 'Vendas a termo a receber']


# In[34]:


np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)


# In[35]:


forest_importances = pd.Series(model.feature_importances_, index=features)


# In[36]:


df_forest_importances = pd.DataFrame(forest_importances, columns=['feature_importance'])


# In[37]:


df_forest_importances['std'] = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)


# In[38]:


df_forest_importances = df_forest_importances.sort_values('feature_importance', ascending=True)


# In[39]:


fig, ax = plt.subplots(figsize = (20,12))
df_forest_importances['feature_importance'].plot.barh(xerr=df_forest_importances['std'], ax=ax)

plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

fig.tight_layout()

#ax.set_title("Feature importances using MDI")
ax.set_ylabel("MDI")
plt.savefig('FeatureImportances.eps', bbox_inches="tight")
plt.show()


# In[40]:


worst_features_sorted = list(df_forest_importances['feature_importance'].index)


# In[41]:


from tqdm.notebook import tqdm


# In[42]:


# Testar tirando as features uma a uma para ver como muda a acurácia
results = []

for walking in tqdm(range(len(worst_features_sorted))):
    features_to_remove = worst_features_sorted[:walking]
    features_to_use = [x for x in features if x not in features_to_remove]
    print(features_to_remove)
    
    X = data[features_to_use].values
    y = data['tipo_anbima'].values

    forest = grid_clf.best_estimator_
    forest.fit(X, y)

    acc = np.mean(cross_val_score(grid_clf.best_estimator_, X, y, cv=5))
    print('Accuracy: %.4f' % acc)
    results.append([walking, acc])


# In[43]:


res = pd.DataFrame(results).set_index(0)
res.index += 1

plt.plot(res)
plt.ylabel("Acurácia")
plt.xlabel("Nº de Features Removidas")
plt.savefig('feature_removal.eps')


# In[44]:


from sklearn.model_selection import cross_val_predict


# In[45]:


import seaborn as sns
from sklearn.metrics import confusion_matrix


# In[46]:


X = data[['Ações',
       'Ações e outros TVM cedidos em empréstimo',
       'Brazilian Depository Receipt - BDR',
       'Certificado ou recibo de depósito de valores mobiliários',
       'Compras a termo a receber', 'Cotas de Fundos',
       'Cotas de fundos de investimento - Instrução Nº 409',
       'Cotas de fundos de renda fixa', 'Cotas de fundos de renda variável',
       'DIFERENCIAL DE SWAP A PAGAR', 'DIFERENCIAL DE SWAP A RECEBER',
       'DISPONÍVEL DE OURO', 'Debêntures', 
       'Debêntures conversíveis',
       'Debêntures simples', 
       'Depósitos a prazo e outros títulos de IF',
       'Disponibilidades', 'Investimento no Exterior',
       'Mercado Futuro - Posições compradas',
       'Mercado Futuro - Posições vendidas',
       'Obrigações por ações e outros TVM recebidos em empréstimo',
       'Obrigações por compra a termo a pagar',
       'Obrigações por venda a termo a entregar', 'Operações Compromissadas',
       'Opções - Posições lançadas', 'Opções - Posições titulares',
       'Outras aplicações', 'Outras operações passivas e exigibilidades',
       'Outros valores mobiliários ofertados privadamente',
       'Outros valores mobiliários registrados na CVM objeto de oferta pública',
       'Títulos Públicos', 'Títulos de Crédito Privado',
       'Títulos ligados ao agronegócio', 'Valores a pagar',
       'Valores a receber', 'Vendas a termo a receber']].values
y = data['tipo_anbima'].values


# In[47]:


y_pred = cross_val_predict(model, X, y, cv=5)
conf_mat = confusion_matrix(y, y_pred)


# In[48]:


kv = {k:v for k, v in zip(sorted_tipo_anbima, range(len(data['tipo_anbima'].unique())))}


# In[49]:


kv


# In[50]:


# get pandas dataframe
df_cm = pd.DataFrame(conf_mat, 
                     index=kv.keys(),
                     columns=kv.keys())
# colormap: see this and choose your more dear
cmap = 'PuRd'
pp_matrix_modified(df_cm, cmap=cmap, fz=9, figsize=[15, 15]) #, save_fig='matrizconfusaomodificada.eps')


# In[51]:


cmn = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(cmn.T, annot=True, fmt='.2f', xticklabels=kv.keys(), yticklabels=kv.keys())
plt.ylabel('Previsto')
plt.xlabel('Real')
plt.savefig('matrizconfusaopadrao.eps', bbox_inches="tight")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# redoing with nivel 2


# In[ ]:





# In[57]:


all_funds_df = pd.read_csv('all_funds_df.csv')
all_funds_df = all_funds_df.drop('Unnamed: 0', axis=1)

arquivos_carteira = glob.glob('../data/cda_fi_202106/cda_fi_BLC*')
df = pd.concat([pd.read_csv(arquivo, sep=';', encoding='latin-1', low_memory=False) for arquivo in arquivos_carteira])
df_cadastro = pd.read_csv("../data/cad_fi.csv", sep=';', encoding='latin-1', low_memory=False)
df_study = df.copy()
a = df_study.groupby(['CNPJ_FUNDO', 'TP_APLIC']).agg({'VL_MERC_POS_FINAL': 'sum'})
soma_pl_total = df_study.groupby('CNPJ_FUNDO').agg({'VL_MERC_POS_FINAL': 'sum'})
soma_pl_total.columns = ['pl_total']
b = pd.merge(a, soma_pl_total, left_index=True, right_index=True)
c = (b['VL_MERC_POS_FINAL']/b['pl_total']).reset_index()
c = c.rename({0: 'values'}, axis=1)
pivot = pd.pivot_table(c, values='values', index=["CNPJ_FUNDO"], columns="TP_APLIC", fill_value=0) 
pivot.index = pivot.reset_index()['CNPJ_FUNDO'].str.replace('\.|/|-', '', regex=True)
all_funds_df['cnpj_fundo'] = all_funds_df['cnpj_fundo'].dropna().apply(round).astype(str).apply(lambda x: x.zfill(14))
data = all_funds_df.merge(pivot, left_on='cnpj_fundo', right_index=True)

df_cadastro = df_cadastro[df_cadastro['TP_FUNDO'] == 'FI']
df_cadastro = df_cadastro.drop_duplicates('CNPJ_FUNDO')
df_cadastro['cnpj_fundo'] = df_cadastro['CNPJ_FUNDO'].str.replace('\.|/|-', '', regex=True)
data = data.merge(df_cadastro[['cnpj_fundo', 'FUNDO_COTAS']], on='cnpj_fundo', how='inner')

# teste tirando os fundos de previdencia
data = data[~data['tipo_anbima'].str.contains('Previdência')]
data['tipo_anbima'].value_counts().head(60)
min_count = 100
contagem_tipos = data['tipo_anbima'].value_counts()
contagem_tipos = contagem_tipos[(contagem_tipos >= min_count) == True].index.values

data = data[data['tipo_anbima'].isin(contagem_tipos)]

data = data[data['FUNDO_COTAS'] == 'N']


# In[58]:


nivel2translate = {
    'Ações Invest. no Exterior': 'Ações Invest. no Exterior',
    'Ações Livre': 'Ações Ativo',
    'Ações Valor/Crescimento': 'Ações Ativo',
    'Ações Índice Ativo': 'Ações Ativo',
    'Multimercados Dinâmico': 'Multimercados Alocação',
    'Multimercados Estrat. Específica': 'Multimercados Estratégia',
    'Multimercados Invest. no Exterior': 'Multimercados Invest. no Exterior',
    'Multimercados Juros e Moedas': 'Multimercados Estratégia',
    'Multimercados L/S - Direcional': 'Multimercados Estratégia',
    'Multimercados Livre': 'Multimercados Estratégia',
    'Multimercados Macro': 'Multimercados Estratégia',
    'Renda Fixa Duração Baixa Grau de Invest.': 'Renda Fixa Duração Baixa',
    'Renda Fixa Duração Baixa Soberano': 'Renda Fixa Duração Baixa',
    'Renda Fixa Duração Livre Crédito Livre': 'Renda Fixa Duração Livre',
    'Renda Fixa Duração Livre Grau de Invest.': 'Renda Fixa Duração Livre',
    'Renda Fixa Duração Livre Soberano': 'Renda Fixa Duração Livre',
    'Renda Fixa Indexados': 'Renda Fixa Indexados',
}


# In[59]:


data['tipo_anbima'] = data['tipo_anbima'].apply(lambda x: nivel2translate[x])


# In[60]:


sorted_tipo_anbima = sorted(data['tipo_anbima'].unique())


# In[61]:


kv = {k:v for k, v in zip(sorted_tipo_anbima, range(len(data['tipo_anbima'].unique())))}
from pprint import pprint
pprint({v:k for k,v in kv.items()})


# In[62]:


model = RandomForestClassifier(class_weight='balanced', n_estimators=190, n_jobs=2, random_state=1)


# In[63]:


data['tipo_anbima'] = data['tipo_anbima'].apply(lambda x: kv[x])


# In[64]:


X = data[['Ações',
       'Ações e outros TVM cedidos em empréstimo',
       'Brazilian Depository Receipt - BDR',
       'Certificado ou recibo de depósito de valores mobiliários',
       'Compras a termo a receber', 'Cotas de Fundos',
       'Cotas de fundos de investimento - Instrução Nº 409',
       'Cotas de fundos de renda fixa', 'Cotas de fundos de renda variável',
       'DIFERENCIAL DE SWAP A PAGAR', 'DIFERENCIAL DE SWAP A RECEBER',
       'DISPONÍVEL DE OURO', 'Debêntures', 
       'Debêntures conversíveis',
       'Debêntures simples', 
       'Depósitos a prazo e outros títulos de IF',
       'Disponibilidades', 'Investimento no Exterior',
       'Mercado Futuro - Posições compradas',
       'Mercado Futuro - Posições vendidas',
       'Obrigações por ações e outros TVM recebidos em empréstimo',
       'Obrigações por compra a termo a pagar',
       'Obrigações por venda a termo a entregar', 'Operações Compromissadas',
       'Opções - Posições lançadas', 'Opções - Posições titulares',
       'Outras aplicações', 'Outras operações passivas e exigibilidades',
       'Outros valores mobiliários ofertados privadamente',
       'Outros valores mobiliários registrados na CVM objeto de oferta pública',
       'Títulos Públicos', 'Títulos de Crédito Privado',
       'Títulos ligados ao agronegócio', 'Valores a pagar',
       'Valores a receber', 'Vendas a termo a receber']].values
y = data['tipo_anbima'].values


# In[65]:


acc = np.mean(
    cross_val_score(
        model, X, y, cv=5
    )
)


# In[66]:


balanced_acc = np.mean(
    cross_val_score(
        model, X, y, scoring=metrics.make_scorer(metrics.balanced_accuracy_score), cv=5
    )
)


# In[67]:


balanced_acc_adj = np.mean(
    cross_val_score(
        model, X, y, scoring=metrics.make_scorer(metrics.balanced_accuracy_score, adjusted=True), cv=5
    )
)


# In[68]:


f1_weighted = np.mean(
        cross_val_score(
        model,
        X, y,
        scoring=metrics.make_scorer(metrics.f1_score, average='weighted'),
        cv=5
    )
)


# In[69]:


from sklearn.metrics import roc_auc_score


# In[70]:


print('Acurácia:',  acc)
print('Acurácia Balanceada:',  balanced_acc)
print('Acurácia Balanceada Ajustada:',  balanced_acc_adj)
print('F1 Weighted:',  f1_weighted)


# In[71]:


y_pred = cross_val_predict(model, X, y, cv=5)
conf_mat = confusion_matrix(y, y_pred)


# In[72]:


kv = {k:v for k, v in zip(sorted_tipo_anbima, range(len(data['tipo_anbima'].unique())))}


# In[73]:


kv


# In[74]:


df_cm


# In[78]:


# get pandas dataframe'
df_cm = pd.DataFrame(conf_mat, 
                     index=kv.keys(),
                     columns=kv.keys())
# colormap: see this and choose your more dear
cmap = 'PuRd'
pp_matrix_modified(
    df_cm, 
    cmap=cmap, 
    fz=10, 
    figsize=[10, 10]
, save_fig='matrizconfusaomodificadaNIVEL2.eps')


# In[76]:


cmn = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn.T, annot=True, fmt='.2f', xticklabels=kv.keys(), yticklabels=kv.keys())
plt.ylabel('Predicted')
plt.xlabel('Actual')
plt.savefig('matrizconfusaopadraoNIVEL2.eps', bbox_inches="tight")
plt.show(block=False)


# In[147]:


data['tipo_anbima'].value_counts()


# In[ ]:





# In[ ]:





# In[ ]:




