# Wykorzystując moduł sklearn.dataset pobierz dataset diabetes
# 1) Wyświetl top 10 wierszy
# 2) Przygotuj funkcję mapującą wartości (age, sex, bmi, bp ...) (https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
#    na wspaznik progresji choroby
# 3) Wykorzystaj funkcję curve_fit aby policzyć parametry funkcji
# 3.1) Podzielic dataset na testowy i trenujcy
# 4) Policz jak usunięcie jednej z kolumn wpłynie na wartość błędu średniokwadratowego dla modelu
# 5) Przedstaw swoją interpretację parametrów (ich ważność, przedział liczbowy) za pomoca krótkiego komentarza pod funkcją main()

import pandas as pd
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import seaborn as sns

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler

matplotlib.rcParams['figure.figsize'] = [10, 10]

PropertyArea = 0
NumberOfRooms = 1
NumberOfBath = 2
NumberOfUnit = 3
NumberOfStories = 4

Weights = [1,1,1,1,1]

picked_column_names = ["Property Area", "Number of Rooms", "Number of Bathrooms",
                        "Number of Units", "Number of Stories"]

def func(x, a0, a1, a2, a3, a4, a):
    return x[PropertyArea]*a0 + x[NumberOfRooms]*a1 
    + x[NumberOfBath]*a2 + x[NumberOfUnit]*a3 + x[NumberOfStories]*a4 + a

class CustomModelWrapper:
    def __init__(self, pred_fun, params):
        self.pred_fun = pred_fun
        self.params = params
    
    def predict(self, x):
        return self.pred_fun(x.ravel(), *self.params)

def main():
    global Weights
    
    print('Read data')
    #dataFr = pd.read_csv("./clean_data.csv")
    dataFr = pd.read_csv("./clean_data_without_nans.csv")
    
    
    print('Train data in: all')
    X = dataFr[picked_column_names].values
    Y = dataFr['Assessed Value'].values

    # normilize X
    X =(X-X.mean())/X.std()
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)    
    
    
    best_hyperparams = []
    best_mistake = float('inf')
    
    #test_vals = [0.01, 0.1, 0.5, 1, 2, 2.5]
    test_vals = [0.1, 0.5, 1, 2]
    #test_vals = [1, 2]
    
    count = 0
    
    for i1 in test_vals:
        for i2 in test_vals:
            for i3 in test_vals:
                for i4 in test_vals:
                    for i5 in test_vals:
                        Weights.clear()
                        Weights = [i1, i2, i3, i4, i5]
                        params, _ = curve_fit(func, xdata=X_train.ravel(), ydata=Y_train)    
                        model_custom = CustomModelWrapper(func, params)
                        Y_pred = list(map(model_custom.predict, X_test))
                        svr_mse = mean_squared_error(Y_test, Y_pred)
                        if(svr_mse < best_mistake):
                            best_mistake = svr_mse
                            best_hyperparams = [i1, i2, i3, i4, i5]
                        count = count + 1
                        print('Progress: ' + str(count) + '/' + str(len(test_vals) ** 5))
                        print('Progress: ' + str(round(count/(len(test_vals) ** 5)*100, 2)) + '%')
                        
                        
    print('Best fit')
    print(best_hyperparams)
    print(best_mistake)


if __name__ == "__main__":
    main()
