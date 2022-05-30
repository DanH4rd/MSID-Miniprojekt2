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

matplotlib.rcParams['figure.figsize'] = [10, 10]

PropertyArea = 0
NumberOfRooms = 1
NumberOfBath = 2
NumberOfBed = 3
NumberOfUnit = 4
NumberOfStories = 5
BasementArea = 6

picked_column_names = ["Property Area", "Number of Rooms", "Number of Bathrooms",
                       "Number of Bedrooms", "Number of Units",
                     "Number of Stories",  "Basement Area"]

def func(x, a0, a1, a2, a3, a4, a5, a):
    return x[PropertyArea]*a0 + x[NumberOfRooms]*a1 + x[NumberOfBath]*a2 + x[NumberOfUnit]*a3 + x[NumberOfStories]*a4 + x[BasementArea]*a5 + a

class CustomModelWrapper:
    def __init__(self, pred_fun, params):
        self.pred_fun = pred_fun
        self.params = params
    
    def predict(self, x):
        return self.pred_fun(x.ravel(), *self.params)

def main():
    dataFr = pd.read_csv("./clean_data.csv")
    X = dataFr['Assessed Value'].values
    Y = dataFr[picked_column_names].values

    print(X)
    print(Y)
    X =(X-X.mean())/X.std()
    Y =(Y-Y.mean())/Y.std()

    print(X)
    print(Y)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    params, _ = curve_fit(func, xdata=X_train.ravel(), ydata=Y_train)
    
    print('Params: ' + str(params))
    print()
    
    model_custom = CustomModelWrapper(func, params)
    Y_pred = list(map(model_custom.predict, X_test))

    svr_mse = mean_squared_error(Y_test, Y_pred)
    print('Mean error: ' + svr_mse.astype('str'))
    print()


if __name__ == "__main__":
    main()
