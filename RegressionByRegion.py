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

def takeError(elem):
    return elem[4]

def takeErrorPerc(elem):
    return elem[5]

def main():
    print('Read data')
    #dataFr = pd.read_csv("./clean_data.csv")
    dataFr = pd.read_csv("./clean_data_without_nans.csv")

    
    Region_data = []

    if(True):
        for loc in np.unique(dataFr['Assessor Neighborhood'].dropna().values.astype('str')):
            print('Train data in: ' + loc)
            selectDataFr = dataFr[dataFr['Assessor Neighborhood'] == loc]
            X = selectDataFr[picked_column_names].values
            Y = selectDataFr['Assessed Value'].values

            # normilize X
            X =(X-X.mean())/X.std()
            
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)   
            
            
            params, _ = curve_fit(func, xdata=X_train.ravel(), ydata=Y_train)    
            model_custom = CustomModelWrapper(func, params)
            Y_pred = list(map(model_custom.predict, X_test))

            svr_mse = mean_squared_error(Y_test, Y_pred, squared=False)
            
            Region_data.append([loc, np.average(Y), np.average(Y_test), np.average(Y_pred), svr_mse, round(svr_mse/np.average(Y)*100,0)])
            if(False):
                print('AVG value: ' + str(np.average(Y)))
                print('AVG traint value: ' + str(np.average(Y_test)))
                print('AVG pred value: ' + str(np.average(Y_pred)))
                print('Mean error: ' + svr_mse.astype('str'))
                print('Error % from avg: ' + (round(svr_mse/np.average(Y)*100,0)).astype('str') + '%')
                print()
            
    
    if(True):
        Region_data.sort(key=takeError)
        print('Top 5 predictions by error\n----------------------')
        for i in Region_data[0:5]:
            print('Location: ' + i[0].astype('str'))
            print('AVG value: ' + i[1].astype('str'))
            print('AVG traint value: ' + i[2].astype('str'))
            print('AVG pred value: ' + i[3].astype('str'))
            print('Mean error: ' + i[4].astype('str'))
            print('Error % from avg: ' + i[5].astype('str') + '%')
            print()
    
    if(True):
        Region_data.sort(key=takeErrorPerc)
        print('Top 5 predictions by error%\n----------------------')
        for i in Region_data[0:5]:
            print('Location: ' + i[0].astype('str'))
            print('AVG value: ' + i[1].astype('str'))
            print('AVG traint value: ' + i[2].astype('str'))
            print('AVG pred value: ' + i[3].astype('str'))
            print('Mean error: ' + i[4].astype('str'))
            print('Error % from avg: ' + i[5].astype('str') + '%')
            print()

if __name__ == "__main__":
    main()
