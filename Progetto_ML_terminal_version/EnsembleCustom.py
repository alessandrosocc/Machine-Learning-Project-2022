from libraries import *
from csv_open import csv_open
from metrics import metrics
from clear_terminal import clear_terminal

#funzione per il majority voting
def aggregate_function(labels, probability, voting_mode, weights, predictions):
    
    maggiore = [] 
    count_0 = 1 #variabile che tiene il conto dei classificatori he hanno votato per la label 0
    count_1 = 0 #variabile che tiene il conto dei classificatori he hanno votato per la label 1

    if voting_mode == 1: #hard voting
        for i in range(len(predictions)): #controllo la previsione di ogni classificatore e scelgo la classe votata dal maggior numero di essi
            if predictions[i] == 0:
                count_0 = count_0 + (1 * weights[i])
            else:
                count_1 = count_1 + (1 * weights[i])

        if count_0 > count_1:
            return 0
        else:
            return 1

    if voting_mode == 2: #soft voting #controllo le probabilità delle classi di ogni classificatore
        maggiore.append(sum(probability[0][:]*weights))
        maggiore.append(sum(probability[1][:]*weights))
        #verrà restituito 0 se la classe 0 avrà avuto la probabilità più alta, in caso contrario verrà restituita la classe 1
        if (maggiore[0] > maggiore[1]):
            return 0
        else: 
            return 1

class Ensemble:
    #fase di inizializzazione
    def __init__(self, estimators, voting, w):
        self.estimators = estimators 
        self.voting = voting
        if w == None: #se non vengono inseriti pesi il tutti avranno lo stesso (1)
            self.w = [1,1,1]
        else:
            self.w = w #altrimenti ad ogni classificatore sarà assegnato il proprio peso

        self.fitted = False

    #fase di allenamento
    def fit(self, x, y, labels):
        self.labels = labels
        #per ogni classificatore verrà fatta una fit con il data set partizionato in modo diverso
        for estimator in self.estimators:
            sub_train_x, _, sub_train_y, _ = train_test_split(x, y, test_size=0.20, stratify=y, random_state = 42)
            estimator.fit(sub_train_x, sub_train_y)
        self.fitted = True #flag che indica che il classificatore multiplo è stato correttamente allenato 

    #fase di predizione
    def predict(self, test_x):
        if self.fitted: #posso continuare con la funzione solo se i classificatori sono già stati allenati
            proba = [] 
            predictions = []
            for estimator in self.estimators: #si calcolano le probabilità di predizione per ogni classificatore
                proba.append(estimator.predict_proba(test_x))
                predictions.append(estimator.predict(test_x))
            proba = np.array(proba) #vengono trasformate in np.array per poter svolgere delle operazioni numpy
            predictions = np.array(predictions).T
            pred_y = []
            for i in range(0, len(test_x)): #per ogni riga del test set viene trovata la classe prevalente tra i classificatori
                pred_y.append(aggregate_function(self.labels, proba[:, i, :].T, self.voting, self.w, predictions[i]))
            return pred_y #è ritornata infine la lista con le predizioni sul test_set 
                          # andrà confrontata con le classi reali per valurare l'accuracy del classificatore
        else:
            print("Il classificatore non è ancora stato addestrato")

#funzione di ensamble 
def myEnsemble():

    clear_terminal()

    X, y, dataType, features = csv_open() #Utilizzando la funzione csv_open andiamo a prendere i dati del dataset per utilizzarli in questo classificatore
    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=0, test_size=0.25, stratify=y) #Facciamo lo split dei dati in training e test set

    clear_terminal()
    print("loading...")

    #TUNING kNN
    kNN = KNeighborsClassifier() #Inizializziamo l'oggetto KNN che sarà il nostro classificatore

    params_kNN = { #Definiamo un dizionario con tutti i parametri che varieranno durante il tuning
              'n_neighbors': [1, 100], #numero degli elementi da prendere in considerazione per la scelta della classe k
              'weights': ('uniform', 'distance'), #pesi usati nella predizione, possono essere uniformi oppure cambiano in base alla distanza
              'metric': ('cityblock', 'euclidean', 'cosine') #tipologie di misure per il calcolo della distanza 
              }
    # grid search
    kNN_grid = GridSearchCV(kNN, params_kNN, cv=10, n_jobs=-1)# estimator indica il classificatore da usare, param_grid gli iperparametri su cui fare il tuning, in forma di dizionario. cv = 10 indica di fare una ten fold cross validation, se specificato un valore intero allora esegue una stratified cross validation, in questo caso stratified ten-fold cross validation. n_jobs = -1 indica di usare tutti i processori.
    results_kNN = kNN_grid.fit(train_x, train_y)#Addestriamo il classificatore generato utilizzando GridSearchCV
    # best model
    kNN_best = kNN.set_params(**results_kNN.best_params_)#Assegnamo i migliori parametri ottenuti con il precedente classificatore a un nuovo classificatore

    #TUNING dTree
    dTree = DecisionTreeClassifier(random_state=0) #Inizializziamo l'oggetto dTree che sarà il nostro classificatore

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

    #TUNING gNB
    gNB = GaussianNB() #Inizializziamo l'oggetto gNB che sarà il nostro classificatore

    params_Bayes = {#Definiamo un dizionario con tutti i parametri che varieranno durante il tuning
        'var_smoothing': np.logspace(0, -9, num=100) #porzione della varianza più grande di tutte le features che viene aggiunta alle varianze di tutte le features per migliorare la stabilità numerica durante il calcolo
    }
    # grid search
    gNB_grid = GridSearchCV(estimator=gNB, param_grid=params_Bayes, n_jobs=-1, cv=10)# estimator indica il classificatore da usare, param_grid gli iperparametri su cui fare il tuning, in forma di dizionario. cv = 10 indica di fare una ten fold cross validation, se specificato un valore intero allora esegue una stratified cross validation, in questo caso stratified ten-fold cross validation. n_jobs = -1 indica di usare tutti i processori.
    results_gNB = gNB_grid.fit(train_x, train_y)#Addestriamo il classificatore generato utilizzando GridSearchCV
    # final model
    gNB_best = gNB.set_params(**results_gNB.best_params_)#Assegnamo i migliori parametri ottenuti con il precedente classificatore a un nuovo classificatore

    clear_terminal()
    print("---------")
    print("1 | Hard")
    print("---------")
    print("2 | Soft")
    print("---------")
    print("Vuoi fare un Hard Voting o un Soft Voting?") #l'utente può selezionare il tipo di voting da utlizzare (hard/soft)
    voting_type = int(input())
    print("---------------")
    print("1 | Pesato")
    print("---------------")
    print("2 | Non pesato")
    print("---------------")
    print("Vuoi utilizzare dei pesi?") #l'utente può scegliere di assegnare dei pesi ad ogni classificatore
    scelta = int(input())
    if(scelta == 1): #nel caso si inserisca 1 devono essere inseriti i pesi per ogni classificatore
        weights = []
        print("Inserisci 3 pesi:")
        weights.append(int(input())) 
        weights.append(int(input())) 
        weights.append(int(input())) 
        #il primo peso sarà assegnato al primo classificatore inserito, il secondo al secondo e il terzo al terzo 
        e2_clf = Ensemble(estimators=[kNN_best, dTree_best, gNB_best], voting=voting_type, w = weights)
        e2_clf.fit(train_x, train_y, y)
        pred_y = e2_clf.predict(test_x)
    else: #altrimenti il classificatore verrà inizializzato senza pesi
        e2_clf = Ensemble(estimators=[kNN_best, dTree_best, gNB_best], voting=voting_type, w = None)
        e2_clf.fit(train_x, train_y, y)
        pred_y = e2_clf.predict(test_x)
    
    
    print("-----------------------")
    print("PERFORMANCES Ensemble:")
    print("-----------------------")
    #calcolo delle metriche per la valutazione del classificatore
    metrics(test_y, pred_y, np.unique(test_y))
