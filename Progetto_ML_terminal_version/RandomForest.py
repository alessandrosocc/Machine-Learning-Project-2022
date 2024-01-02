from libraries import *
from csv_open import csv_open
from metrics import metrics
from clear_terminal import clear_terminal


def RandomForest():
    X, y, dataType, features = csv_open() #Utilizzando la funzione csv_open andiamo a prendere i dati del dataset per utilizzarli in questo classificatore
    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=0, test_size=0.25, stratify=y)#Facciamo lo split dei dati in training e test set
    
    clear_terminal()
    print("Non chiudere! Sto elaborando ...")

    rf = RandomForestClassifier() #Inizializziamo l'oggetto dTree che sarà il nostro classificatore
    params={#Definiamo un dizionario con tutti i parametri che varieranno durante il tuning
            "n_estimators":list(range(10,80,10)), #numero degli alberi decisionali da usare
            "criterion":["gini","entropy","log_loss"], #funzioni per misurare la qualità di uno split
            "max_depth":[None]+list(range(4,34,10)), #massima profondità dell'albero
            "max_features":["sqrt","log2"] #numero di attrubuti da considerare quando si cerca il miglior split
            }

    #grid search
    rf_grid = GridSearchCV(estimator=rf, cv=10,param_grid=params,n_jobs=-1)# estimator indica il classificatore da usare, param_grid gli iperparametri su cui fare il tuning, in forma di dizionario. cv = 10 indica di fare una ten fold cross validation, se specificato un valore intero allora esegue una stratified cross validation, in questo caso stratified ten-fold cross validation. n_jobs = -1 indica di usare tutti i processori.
    results = rf_grid.fit(train_x, train_y)#Addestriamo il classificatore generato utilizzando GridSearchCV

    #best model
    rf_best = rf.set_params(**results.best_params_) #Assegnamo i migliori parametri ottenuti con il precedente classificatore a un nuovo classificatore
    rf_best.fit(train_x, train_y)#Addestriamo il nuovo classificatore
    pred_y = rf_best.predict(test_x)#Gli facciamo eseguire una predizione

    clear_terminal()
    print("---------------------------")
    print("BEST PARAMETERS:")
    print("---------------------------")
    print(results.best_params_)# best_params_ è un attributo che restituisce un dizionario con i parametri migliori trovati
    print("---------------------------")
    print("PERFORMANCES RandomForest:")
    print("---------------------------")
    metrics(test_y, pred_y, np.unique(test_y))#Stampiamo a video le metriche della predizione ottenute da una matrice di confusione

