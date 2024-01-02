import pandas as pd
from libraries import *
from metrics import metrics
from clear_terminal import clear_terminal

def DataAnalysis():
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
    Q1=X.quantile(0.25) # calcolo primo quartile
    Q3=X.quantile(0.75) # calcolo terzo quartile
    IQR=Q3-Q1 # calcolo distanza interquartile
    outliersMatrix = pd.DataFrame(X[(X<Q1-1.5*IQR) | (X>Q3+1.5*IQR) & (X<X.quantile(0.10)) | (X>X.quantile(0.90))])

    #Contiamo quanti outliers ci sono per ogni attributo
    outliersCount=X.shape[0]-outliersMatrix.isnull().sum()

    #Mostriamo la correlazione tra gli attributi tramite una matrice di correlazione
    #Viene usato il colore 'coolwarm' per mostrare i valori dei coefficienti
    corr_df = df.corr()
    matrix_corr = corr_df.style.background_gradient(cmap='coolwarm')
    clear_terminal()
    print("----------------------")
    print("1 | Statistiche generali")
    print("----------------------")
    print("2 | Numero di valori non nulli per ogni attriubto")
    print("----------------------")
    print("3 | valori nulli per ogni attributo")
    print("----------------------")
    print("4 | Outliers oltre il 90esimo e prima del decimo percentile")
    print("----------------------")
    print("5 | Matrice degli outliers")
    print("----------------------")
    print("6 | Numero degli outliers per ogni attributo")
    print("----------------------")
    print("Che operazione vuoi fare?")

    choiche = int(input())
    if choiche == 1:
        print ('\nStatistiche generali:\n', data_describe)
    elif choiche == 2:
        print ('\nNumero dei valori non nulli per ogni attributo:\n', info_dataset)
    elif choiche == 3:
        print ('\nValori nulli per ogni attributo:\n', null_values)
    elif choiche == 4:
        print ('\nOutliers oltre il 90esimo e prima del decimo percentile:\n', outliers)
    elif choiche == 5:
        print ('\nMatrice degli outliers:\n', outliersMatrix)
    elif choiche == 6:
        print ('\nNumero degli outliers per ogni attributo:\n', outliersCount)

