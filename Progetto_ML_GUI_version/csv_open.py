from libraries import *
from preprocessing import preprocessing
from clear_terminal import clear_terminal


def csv_open(trasformazione, modifica, bilanciamento):
    with open("spambase/features.txt","r") as f:
        lines=f.readlines()
    raw=[x.strip().split(":") for x in lines]
    features=[x[0] for x in raw]+["spam"]
    dataType=[x[1].strip()[:-1] for x in raw]

    df=pd.read_csv("spambase/spambase.csv",names=features,index_col=False)

    X=df.iloc[:,:-1] # drop class
    y=df.iloc[:,-1]
    y=y.to_numpy()[:,]
    
    clear_terminal()
    if (trasformazione == "Nessuna trasformazione") and (modifica == "Nessuna modifica degli attributi") and (bilanciamento == "Nessun bilanciamento"):
        return X, y, dataType, features
    else:
        X, y = preprocessing(X,y,df,features,trasformazione, modifica, bilanciamento)
        return X, y, dataType, features
    


