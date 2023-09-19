import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from yellowbrick.regressor import PredictionError
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
asd = pd.read_json('./imoveis.json', orient='columns')
ident = pd.json_normalize(asd.ident)
listing = pd.json_normalize(asd.listing)
data = pd.concat([ident,listing],axis = 1)
'''
for i in data.columns:
    print('---'*10)
    print(data[i].value_counts())
'''
filter = (data['types.usage'] == 'Residencial')*(data['address.city'] == "Rio de Janeiro")
data = data[filter]
data.reset_index(drop=True, inplace=True)
data = data.astype({
    'prices.price':'float64',
    'prices.tax.iptu':'float64',
    'prices.tax.condo':'float64',
    'features.usableAreas':'int64',
    'features.totalAreas':'int64',
})
data['address.zone'] = data['address.zone'].replace('',np.nan)
dict = data[~data['address.zone'].isna()].drop_duplicates(subset=['address.neighborhood']).to_dict('records')
dict_zones = {i['address.neighborhood']:i['address.zone']for i in dict}
for b, z in dict_zones.items():
    data.loc[data['address.neighborhood']==b,'address.zone']=z

data.head()
print(data['address.zone'].isnull().sum())
data['prices.tax.condo'].fillna(0.0,inplace=True)
data['prices.tax.iptu'].fillna(0.0,inplace=True)
print(data["prices.tax.condo"].isnull().sum())
print(data["prices.tax.iptu"].isnull().sum())
data.drop(['customerID','source','types.usage','address.city','address.location.lon','address.location.lat','address.neighborhood'],axis=1,inplace=True)
dict_columns = {
    'types.unit':'unit',
    'address.zone':'zone',
    'prices.price':'price',
    'prices.tax.condo':'tax.condo',
    'prices.tax.iptu':'tax.iptu',
    'features.bedrooms':'bedrooms',
    'features.bathrooms':'bathrooms',
    'features.suites':'suites',
    'features.parkingSpaces':'parkingSpaces',
    'features.usableAreas':'usableAreas',
    'features.totalAreas':'totalAreas',
    'features.floors':'floors',
    'features.unitsOnTheFloor':'unitsOnTheFloor',
    'features.unitFloor':'unitFloor',
}
data = data.rename(dict_columns, axis=1)
column_n = data.select_dtypes(include=['number'])
correlation = column_n.corr()
print(correlation)
colors = sns.color_palette('light:blue', as_cmap=True)
mask = np.zeros_like(correlation)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style('white'):
    f, ax = plt.subplots(figsize=(13,8))
    ax =sns.heatmap(correlation,cmap=colors,mask=mask,square=False,fmt = '.2f',annot=True)

transformer = FunctionTransformer(np.log1p, validate=True)
data_transformed = transformer.transform(data.select_dtypes(exclude=['object']))
data_transformed_columns = data.select_dtypes(exclude=['object']).columns
df_transformed = pd.concat([data.select_dtypes(include=['object']), pd.DataFrame(data_transformed, columns=data_transformed_columns)], axis=1)
ax = plt.figure()
ax = sns.histplot(data=df_transformed, x='price',kde=True)
ax.figure.set_size_inches(20,10)
ax.set_title('hadwbad')
ax.set_xlabel('preço')
plt.show()

category_variable = df_transformed.select_dtypes(include=['object']).columns
df_dummies = pd.get_dummies(df_transformed[category_variable])
data_dummies = pd.concat([df_transformed.drop(category_variable,axis=1), df_dummies], axis=1)
x = data_dummies.drop('price',axis=1)
y = data_dummies['price']
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=42)
lr = LinearRegression()
lr.fit(x_treino,y_treino)
lr_preview = lr.predict(x_teste)
print(x_teste)
print(lr_preview)
print(data_dummies.head(33542))
print(np.expm1(7.496097))

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_squared_error(y_true, y_pred)
dtr = DecisionTreeRegressor(random_state=42,max_depth=5)
dtr.fit(x_treino,y_treino)
fig,ax = plt.subplots(figsize=(10,10))
pev = PredictionError(dtr)
pev.fit(x_treino,y_treino)
pev.score(x_teste,y_teste)
pev.poof()

rf = RandomForestRegressor(random_state=42,max_depth=5,n_estimators=20)
rf.fit(x_treino,y_treino)
asd_rf = rf.predict(x_teste)
fig,ax = plt.subplots(figsize=(10,10))
pev = PredictionError(rf)
pev.fit(x_treino,y_treino)
pev.score(x_teste,y_teste)
pev.poof()