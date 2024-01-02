from libraries import *
from csv_open import csv_open
from metrics import metrics
def dTree(tra,mod,bil):
    X, y, dataType, features = csv_open(tra,mod,bil) #Utilizzando la funzione csv_open andiamo a prendere i dati del dataset per utilizzarli in questo classificatore
    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=0, test_size=0.25, stratify=y) #Facciamo lo split dei dati in training e test set
    dTree = DecisionTreeClassifier() #Inizializziamo l'oggetto dTree che sarà il nostro classificatore
    params_dTree = { #Definiamo un dizionario con tutti i parametri che varieranno durante il tuning
        'criterion': ["entropy", "gini"], # le funzioni da usare per misurare la qualità di uno split
        'max_depth': [None]+list(range(2,30,2)), # Indica la massima profondità dell'albero, "None" indica che espande l'albero finchè tutte le foglie sono pure oppure fino a quando tutte le foglie contengono meno di min_samples_split campioni. 
        'max_features': ["sqrt", "log2"], #indica il numero di attributi da considerare quando si cerca il miglior split
        'min_samples_leaf': list(range(1,10,1)), # indica il numero minimo di elementi richiesto per essere in un nodo foglia
        'min_samples_split': list(range(2,10,1)) # indica il numero minimo di elementi richiesti per fare uno split in un nodo interno
    } 
    # grid search
    dTree_grid = GridSearchCV(estimator = dTree, param_grid = params_dTree, cv=10, n_jobs=-1) # estimator indica il classificatore da usare, param_grid gli iperparametri su cui fare il tuning, in forma di dizionario. cv = 10 indica di fare una ten fold cross validation, se specificato un valore intero allora esegue una stratified cross validation, in questo caso stratified ten-fold cross validation. n_jobs = -1 indica di usare tutti i processori.
    results_dTree = dTree_grid.fit(train_x, train_y) #Addestriamo il classificatore generato utilizzando GridSearchCV

    # best model
    dTree_best = dTree.set_params(**results_dTree.best_params_) #Assegnamo i migliori parametri ottenuti con il precedente classificatore a un nuovo classificatore
    dTree_best.fit(train_x, train_y) #Addestriamo il nuovo classificatore
    pred_y = dTree_best.predict(test_x) #Gli facciamo eseguire una predizione

    print("-----------------------------")
    print("BEST PARAMETERS:")
    print("-----------------------------")
    print(results_dTree.best_params_)
    print("-----------------------------")
    print("PERFORMANCES decisionalTree:")
    print("-----------------------------")
    return metrics(test_y, pred_y, np.unique(test_y)), results_dTree.best_params_ #Stampiamo a video le metriche della predizione ottenute da una matrice di confusione

