import numpy as np
import pandas as pd
from random import random, randint
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import pickle 

def create_df(n):

    m = get_random_item()

    for i in range(n):
        v = get_random_item()
        m = np.vstack((m,v))

    df = pd.DataFrame(m, columns=['blob', 'x', 'y', 'last_x', 'lspeed', 'rspeed'])

    return df

def get_random_item():
    blob = randint(0,1)
    x =  random()
    y = random()
    last_x = random()
    lspeed, rspeed = get_speed(blob, x, y, last_x)

    return np.array([blob, x, y, last_x, lspeed, rspeed])

def get_speed(blob, x, y, last_x):

    if blob == 1: # ve la pelota
        if x <= 0.4: # Pelota hacia la izquierda
            lspeed, rspeed = +0.5, +1.3 # Gira izquierda
        elif x >= 0.6: #Pelota hacia la derecha
            lspeed, rspeed = +1.3, +0.5 # Gira derecha
        else:
            lspeed, rspeed = 1.5, 1.5

    else: # no ve la pelota
        if last_x <= 0.5: # Pelota hacia la izquierda
            lspeed, rspeed = -0.5, +0.5 # Gira izquierda
        else: #Pelota hacia la derecha
            lspeed, rspeed = +0.5, -0.5 # Gira derecha


    return lspeed, rspeed

def create_model(df):

    # Separamos la columna a clasificar (y) del resto del conjunto (X)
    col_names = list(df.columns.values)
    X = df[col_names[:len(col_names) - 2]]
    y = df[col_names[-2:]]

    # Sacamos los datos de entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

    clf = MLPRegressor()
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    
    rmse = mean_squared_error(y_test, y_pred)
    print(rmse)

    return clf

def save_model(clf):
    pickle.dump(clf, open('clf.sav', 'wb'))


if __name__ == "__main__":
    
    n_items = 10000

    df = create_df(n_items)
    clf = create_model(df)
    save_model(clf)