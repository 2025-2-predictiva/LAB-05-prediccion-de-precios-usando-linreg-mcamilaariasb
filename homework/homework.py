#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import numpy as np
import pandas as pd
import os
import json
import gzip
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    median_absolute_error
)

def preprocess_data(path):
    df=pd.read_csv(path,index_col=False,compression='zip')
    df['Age']=2021-df['Year']
    df.drop(columns=['Year','Car_Name'],inplace=True)
    return df

path_train='files/input/train_data.csv.zip'
path_test='files/input/test_data.csv.zip'

train_data=preprocess_data(path_train)
test_data=preprocess_data(path_test)

# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
x_train,y_train=train_data.drop('Present_Price',axis=1),train_data['Present_Price']
x_test,y_test=test_data.drop('Present_Price',axis=1),test_data['Present_Price']

# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
categorical_features=['Fuel_Type','Selling_type','Transmission']
numerical_features= [col for col in x_train.columns if col not in categorical_features]

steps=[('preprocessor',ColumnTransformer(transformers=[('cat',OneHotEncoder(),categorical_features),
                                                       ('num',MinMaxScaler(),numerical_features)],
                                                        remainder='passthrough')),
       ('selector',SelectKBest(f_regression)),
       ('LinearRegression', LinearRegression())    
       ]
pipeline=Pipeline(steps)

# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
param_grid={
    'selector__k':[5,10,15,20],
    'LinearRegression__fit_intercept':[True,False],
}

model=GridSearchCV(pipeline,
                   param_grid=param_grid,
                   cv=10,
                   scoring='neg_mean_absolute_error',
                   n_jobs=-1)

model.fit(x_train,y_train)
    
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
model_filename='files/models'
os.makedirs(model_filename,exist_ok=True)

model_path=os.path.join(model_filename,'model.pkl.gz')
with gzip.open(model_path,'wb') as file:
    pickle.dump(model,file)

# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
def metrics_calculate(model,x_train,x_test,y_train,y_test):
    
    y_train_pred=model.predict(x_train)
    y_test_pred=model.predict(x_test)
    
    metrics=[{'type':'metrics',
           'dataset':'train',
           'r2':r2_score(y_train,y_train_pred),
           'mse':mean_squared_error(y_train,y_train_pred),
           'mad':median_absolute_error(y_train,y_train_pred),},
             
            {'type':'metrics',
           'dataset':'test',
           'r2':r2_score(y_test,y_test_pred),
           'mse':mean_squared_error(y_test,y_test_pred),
           'mad':median_absolute_error(y_test,y_test_pred),}
            ]
            
    os.makedirs('files/output',exist_ok=True)
    with open('files/output/metrics.json','w') as file:
        for metric in metrics:
            file.write(json.dumps(metric)+'\n')
            
        
metrics_calculate(model,x_train,x_test,y_train,y_test)