from libraries import *
from csv_open import csv_open
from metrics import metrics
from clear_terminal import clear_terminal

def kNN():
    X, y, dataType, features = csv_open() #Utilizzando la funzione csv_open andiamo a prendere i dati del dataset per utilizzarli in questo classificatore
    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=0, test_size=0.25, stratify=y) #Facciamo lo split dei dati in training e test set

    #clear_terminal()
    print("loading...")

    kNN = KNeighborsClassifier() #Inizializziamo l'oggetto kNN che sarà il nostro classificatore

    params={ #Definiamo un dizionario con tutti i parametri che varieranno durante il tuning
            'n_neighbors':[1,100], #numero di elementi da prendere in considerazione per l'assegnazione della classe (k)
            'weights':('uniform','distance'), #pesi da assegnare ai k vicini, se "uniform" tutti gli oggetti sono pesati allo stesso modo, se "distance", il peso dei singoli oggetti è inversamente proporzionale alla distanza (favorendo l'influenza degli oggetti più vicini)
            'metric':('cityblock', 'euclidean', 'cosine') #tipo di metrica da utilizzare per calolare la distanza
            }

    #grid search
    kNN_grid = GridSearchCV(kNN, params, cv=10, n_jobs=-1) #La funzione GridSearchCV, effettua una cross validation e cerca i parametri migliori tra le varie iterazioni, estimator indica il classificatore da usare, params gli iperparametri su cui fare il tuning, in forma di dizionario. cv = 10 indica di fare una ten fold cross validation, se specificato un valore intero allora esegue una stratified cross validation, in questo caso stratified ten-fold cross validation. n_jobs = -1 indica di usare tutti i processori.
    results = kNN_grid.fit(train_x, train_y) #Addestriamo il classificatore generato utilizzando GridSearchCV

    #best model
    kNN_best = kNN.set_params(**results.best_params_) #Assegnamo i migliori parametri ottenuti con il precedente classificatore a un nuovo classificatore
    kNN_best.fit(train_x, train_y) #Addestriamo il nuovo classificatore
    pred_y = kNN_best.predict(test_x) #Gli facciamo eseguire una predizione

    clear_terminal()
    print("------------------")
    print("BEST PARAMETERS:")
    print("------------------")
    print(results.best_params_)
    print("------------------")
    print("PERFORMANCES kNN:")
    print("------------------")
    metrics(test_y,pred_y,np.unique(test_y)) #Stampiamo a video le metriche della predizione ottenute da una matrice di confusione
