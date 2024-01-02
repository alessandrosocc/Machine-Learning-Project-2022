from libraries import *
from preprocessing import preprocessing
from clear_terminal import clear_terminal

def csv_open():
    with open("spambase/features.txt","r") as f: # apriamo il file
        lines=f.readlines() # leggiamo le righe
    raw=[x.strip().split(":") for x in lines] # creiamo una lista eliminando i new lines e separando il nome della feature dal tipo
        # con il delimitatore ":"

    features=[x[0] for x in raw]+["spam"] # prendiamo dalla lista "raw" il nome delle features e aggiungiamo la features che indica
        # l'etichetta di classe, ossia "spam"
    dataType=[x[1].strip()[:-1] for x in raw] # questa lista conterrà i tipi delle nostre features

    df=pd.read_csv("spambase/spambase.csv",names=features,index_col=False) # usiamo la libreria pandas per aprire il dataset in un
        # formato csv

    X=df.iloc[:,:-1] # non consideriamo l'ultimo attributo, ossia l'etichetta di classe corrispondente ad una email
    y=df.iloc[:,-1] # consideriamo solo l'etichetta di classe
    y=y.to_numpy()[:,] # trasformiamo in un array numpy
    
    clear_terminal()
    print("-------------------------")
    print("1 | Preprocessing")
    print("-------------------------")
    print("2 | Nessun preprocessing")
    print("-------------------------")
    print("Vuoi eseguire del preprocessing sui dati?")
    prepro = int(input())
    if prepro == 1:
        X, y = preprocessing(X,y,df,features)
        return X, y, dataType, features
    else:
        return X, y, dataType, features

