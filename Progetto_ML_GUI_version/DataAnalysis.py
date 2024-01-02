import pandas as pd
from libraries import *
import sys
from metrics import metrics
from clear_terminal import clear_terminal
from tabulate import tabulate


def DataAnalysis(id):
    clear_terminal()
    pd.options.display.max_columns = None

    with open("spambase/features.txt","r") as f:
        lines=f.readlines()
    raw=[x.strip().split(":") for x in lines]
    features=[x[0] for x in raw]+["spam"]
    dataType=[x[1].strip()[:-1] for x in raw]

    df=pd.read_csv("spambase/spambase.csv",names=features,index_col=False)

    X=df.iloc[:,:-1] # drop class
    y=df.iloc[:,-1]
    y=y.to_numpy()[:,]

    #Mostriamo le statistiche generali per ogni attributo del dataset
    data_describe = X.describe()

    #Mostriamo il numero dei valori NON nulli per ogni attributo
    info_dataset =df.info(verbose = True)

    #Controlliamo la presenza o meno di valori nulli per ogni attributo
    null_values = X.isnull().any(axis=0)

    #Controlliamo inoltre la presenza di outliers, guardando per ogni attributo i valori
    #che si trovano oltre il 90esimo e prima del decimo perentile
    outliers = pd.DataFrame(X[(X<X.quantile(0.10)) | (X>X.quantile(0.90))])

    #Selezione anche degli outliers che si trovano oltre i baffi di un boxplot
    Q1=X.quantile(0.25)
    Q3=X.quantile(0.75)
    IQR=Q3-Q1
    outliersMatrix = pd.DataFrame(X[(X<Q1-1.5*IQR) | (X>Q3+1.5*IQR) & (X<X.quantile(0.10)) | (X>X.quantile(0.90))])

    #Contiamo quanti outliers ci sono per ogni attributo
    outliersCount=X.shape[0]-outliersMatrix.isnull().sum()

    #Mostriamo la correlazione tra gli attributi tramite una matrice di correlazione
    #Viene usato il colore 'coolwarm' per mostrare i valori dei coefficienti
    corr_df = df.corr()
    matrix_corr = corr_df.style.background_gradient(cmap='coolwarm')
    clear_terminal()
    if id == 1:
        original_stdout = sys.stdout
        with open('dataDescribe.txt', 'w') as f:
            sys.stdout = f # Change the standard output to the file we created.
            print(data_describe)
            sys.stdout = original_stdout
            name = f.name
            return name
    elif id == 2:
        print ('\nNumero dei valori non nulli per ogni attributo:\n', info_dataset)
    elif id == 3:
        print ('\nValori nulli per ogni attributo:\n', null_values)
    elif id == 4:
        print ('\nOutliers oltre il 90esimo e prima del decimo percentile:\n', outliers)
    elif id == 5:
        print ('\nMatrice degli outliers:\n', outliersMatrix)
    elif id == 6:
        print ('\nNumero degli outliers per ogni attributo:\n', outliersCount)
    elif id == 7:
        print(matrix_corr)

