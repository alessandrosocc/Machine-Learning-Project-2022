import numpy
from libraries import *
from csv_open import csv_open
from metrics import metrics

class NBayes:
    def __init__(self):
        self.classes = []
        self.prior = {}
        self.post = {}
        self.a_names = []
        self.fitted = False

    def fit(self, train_x, train_y, a_names):
        self.a_names = a_names
        self.get_class_proba(train_y)  # calcolo le probabilità delle classi
        for col in range(0, np.shape(train_x)[1]):  # itero per il numero degli attributi (57)
            self.get_att_proba(train_x[:, col], a_names[col], train_y)
            # print(train_x[:,col])
        self.fitted = True

    def predict(self, test_x, test_y):
        if self.fitted: # posso fare una predizione solo se prima è stato addestrato il modello
            # per ogni record del test set
            pred_y = []
            for row in test_x:  # itero per tutte le righe preseni nel test_x
                i_max = 0
                prob_max = 0
                for i, y_val in enumerate(self.classes):
                    #print(i,y_val)
                    prod = self.prior[y_val]  # inizialmente la probabilità è quella a priori per 0 o per 1
                    y_val = str(y_val)
                    for j, a_val in enumerate(row):
                        # si prendono sia media che varianza calcolate precedentemente per ogni cella del dizionario
                        mu = self.post[y_val + self.a_names[j]]["mu"]
                        var = self.post[y_val + self.a_names[j]]["var"]
                        denominator = np.sqrt(2 * np.pi * var)
                        numerator = np.exp(-np.power(a_val - mu, 2) / (2 * var))
                        #print(numerator,denominator)
                        prod = (prod * numerator) / denominator
                    if prod > prob_max:
                        prob_max = prod
                        i_max = i
                    # vengono calcolate le probabilità per tutti i possibili valori della classe
                    # e poi ci si salva solo la classe che aveva la probabilità più alta

                pred_y = pred_y + [self.classes[i_max]]
            return pred_y
        else:
            print("Il classificatore non è ancora stato addestrato")

    # calcolo delle probabilità a priori delle classi
    def get_class_proba(self, y):  
        n = len(y) # numero classi
        y_vals, y_counts = np.unique(y, return_counts=True)  # unique restituisce i possibili valori delle classi
        self.classes = y_vals  # e il conto dei diversi valori delle classi
        for i, val in enumerate(y_vals):  # calcolo delle probabilità per una classe e per l'altra
            self.prior[val] = y_counts[i] / n # calcolo probabilità

    # funzione che ha il compito di calcolare le probabilità di ogni attributo tra le due possibili classi
    def get_att_proba(self, attribute, a_name, y):
        y_vals, y_counts = np.unique(y, return_counts=True)  # fa la stessa cosa di prima, soltanto che i valori
        # cambiano da attributo ad attributo,
        # vengono analizzati tutti
        for i, y_val in enumerate(y_vals):
            y_val = str(y_val)
            # crea un dizionario formato da 114 chiavi 57 con classe0 e 57 con classe 1
            # ogni chiave del dizionario è data dal nome dell'attributo più il valore della classe corrispondente
            self.post[y_val + a_name] = {}

            # per ogni chiave calcolo anche
            a = attribute[y == int(y_val)].astype(float)  # i valori assunti dall'attributo
            mu = np.mean(a)  # la media
            var = np.var(a)+0.00000000001 # la varianza, aggiungo un valore minuscolo perchè usando le tecniche di undersampling la varianza diventa uguale a 0 e mi da problemi. Con tutte le altre tecniche di preprocessing non avviene.

            # inserisco nel dizionario sia la media che la varianza per ogni cella presente
            self.post[y_val + a_name]["mu"] = mu
            self.post[y_val + a_name]["var"] = var


def myNBayes(tra,mod,bil):
    X, y, dataType, features = csv_open(tra,mod,bil)
    if type(X) != numpy.ndarray:
        X = X.to_numpy()
    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=0, test_size=0.25, stratify=y)
    NB = NBayes()
    NB.fit(train_x, train_y, features)
    pred_y = NB.predict(test_x, test_y)

    # print("-------------------------")
    # print("PERFORMANCES NaiveBayes:")
    # print("-------------------------")
    return metrics(test_y,pred_y,np.unique(test_y))

