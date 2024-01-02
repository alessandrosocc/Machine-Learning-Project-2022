from libraries import *
from clear_terminal import clear_terminal

def preprocessing(X, y, df, features):
    corr_X = X.corr()
    X = X.to_numpy()

    clear_terminal()

    #TRASFORMAZIONE DEI DATI
    clear_terminal()
    print("---------------------------")
    print("1 | MinMaxScaler")
    print("---------------------------")
    print("2 | StandardScaler")
    print("---------------------------")
    print("3 | Normalizzazione")
    print("---------------------------")
    print("4 | Nessuna trasformazione")
    print("---------------------------")
    print("Scegli un tipo di trasformazione:")
    transf = int(input())
    if transf == 1:
        minMaxScaler = MinMaxScaler()
        minMaxScaler.fit(X)
        minMaxScaler.transform(X)
    elif transf == 2:
        zScore = StandardScaler()
        zScore.fit(X)
        zScore.transform(X)
    elif transf == 3:
        normalize(X, norm = 'l1')

    #FEATURE SELECTION/AGGREAGATION
    clear_terminal()
    print("-------------------------------------")
    print("1 | Feature selection")
    print("-------------------------------------")
    print("2 | Feature aggregation")
    print("-------------------------------------")
    print("3 | Nessuna modifica degli attributi")
    print("-------------------------------------")
    print("Scegli una tecnica di riduzione della dimensionalità:")
    feat = int(input())
    if feat == 1:
        clear_terminal()
        print("--------------------------------")
        print("1 | VarianceThreshold")
        print("--------------------------------")
        print("2 | Scoring con chi2")
        print("--------------------------------")
        print("3 | Scoring con mutual_info_classif")
        print("--------------------------------")
        print("4 | SequentialFeatureSelector")
        print("--------------------------------")
        print("5 | FeatureSelection con matrice di correlazione")
        print("--------------------------------")
        print("Scegli una tecnica di selezione delle features:")
        sel = int(input())
        if sel == 1:
            #VarianceThreshold
            selector = VarianceThreshold(threshold=1)
            X = selector.fit_transform(X)
        elif sel == 2:
            #scoring con chi2
            X = SelectKBest(chi2, k=28).fit_transform(X, y)
        elif sel == 3:
            #scoring con mutual_info_classif
            X = SelectKBest(mutual_info_classif, k=28).fit_transform(X, y)
        elif sel == 4:
            clear_terminal()
            print("loading...")
            #SequentialFeatureSelector
            sfs = SequentialFeatureSelector(KNeighborsClassifier(n_neighbors=1), n_features_to_select=28, n_jobs= -1)
            X = sfs.fit_transform(X, y)
        elif sel == 5:
            featureCorr = np.where(corr_X > 0.80)  # quali sono gli attributi con correlazione >0.8?
            # seleziona, per ogni attributo correlato con un altro e che hanno correlazione >0.8, uno dei due attributi non considerando ovviamente
            # gli attributi che per se stessi hanno correlazione == 1
            indexes = [featureCorr[0][i] for i in range(len(featureCorr[0])) if featureCorr[0][i] != featureCorr[1][i]]
            indexes = np.unique(indexes)  # indici degli attributi che sono molto correlati con altri attributi, togliendoli dal dataset iniziale
            # rimarranno gli attributi che hanno una correlazione <=0.8 con qualsiasi altro attributo e il corrispondente attributo che ha correlazione
            # > 0.8 con gli attributi ottenuti 
            indexesNames = [features[i] for i in indexes]  # prendo i nomi delle features
            X = df.drop(columns=indexesNames, inplace=False, axis=1)  # elimina gli attributi con nome: indexesNames,
            # ritorna una copia senza modificare il dataset originale (inplace=False), elimina le colonne (axis=1)
            X = X.to_numpy() # trasformiamo in una matrice numpy
    elif feat == 2:
        #FEATURE AGGREGATION
        clear_terminal()
        print("--------------------------------")
        print("1 | SparseRandomProjection")
        print("--------------------------------")
        print("2 | GaussianRandomProjection")
        print("--------------------------------")
        print("3 | FeatureAgglomeration")
        print("--------------------------------")
        print("4 | PrincipalComponentsAnalysis")
        print("--------------------------------")
        print("Scegli una tecnica di aggregazione:")
        aggr = int(input())
        #tecniche di aggregazione
        if aggr == 1:
            srp = SparseRandomProjection(57)
            X = srp.fit_transform(X)
        elif aggr == 2:
            grp = GaussianRandomProjection(57)
            X = grp.fit_transform(X)
        elif aggr == 3:
            fa = FeatureAgglomeration()
            X = fa.fit_transform(X)
        elif aggr == 4:
            pca = PCA()
            X = pca.fit_transform(X)

    #BALANCING
    clear_terminal()
    print("-------------------------")
    print("1 | Oversampling")
    print("-------------------------")
    print("2 | Undersampling")
    print("-------------------------")
    print("3 | Combinazione di Undersampling e Oversampling")
    print("-------------------------")
    print("4 | Nessun bilanciamento")
    print("-------------------------")
    #le funzioni per il bilanciamento sono state prese dalla libreria ImbalancedLearn
    print("Scegli un tipo di bilanciamento:")
    samp = int(input())
    if samp == 1:
        #OVERSAMPLING
        clear_terminal()
        print("----------------------")
        print("1 | RandomOverSampler")
        print("----------------------")
        print("2 | SMOTE")
        print("----------------------")
        print("3 | ADASYN")
        print("----------------------")
        print("Scegli una tecnica di oversampling:")
        over = int(input())
        if over == 1:
            ros = RandomOverSampler(random_state=42)
            X, y = ros.fit_resample(X, y)
        elif over == 2:
            sm = SMOTE(random_state=42)
            X, y = sm.fit_resample(X, y)
        elif over == 3:
            ada = ADASYN(random_state=42)
            X, y = ada.fit_resample(X, y)
    elif samp == 2:
        #UNDERSAMPLING
        clear_terminal()
        print("------------------------------")
        print("1 | RandomUnderSampler")
        print("------------------------------")
        print("2 | InstanceHardnessThreshold")
        print("------------------------------")
        print("3 | NearMiss v.1")
        print("------------------------------")
        print("4 | NearMiss v.2")
        print("------------------------------")
        print("5 | ClusterCentroids")
        print("------------------------------")
        print("Scegli una tecnica di undersampling:")
        under = int(input())
        if under == 1:
            rus = RandomUnderSampler(random_state=0)
            X, y = rus.fit_resample(X, y)
        elif under == 2:
            iht = InstanceHardnessThreshold()
            X, y = iht.fit_resample(X, y)
        elif under == 3:
            nm = NearMiss(version=1)
            X, y = nm.fit_resample(X, y)
        elif under == 4:
            nm = NearMiss(version=2)
            X, y = nm.fit_resample(X, y)
        elif under == 5:
            cc = ClusterCentroids(random_state=0)
            X, y = cc.fit_resample(X, y)
    elif samp==3:
        clear_terminal()
        tmp=X.copy()

        """
        Faccio undersampling e oversampling separatamente
        Prima applico l'undersampling con il metodo InstanceHardnessThreshold per sottocampionare la classe 0 e portarla a 2300 record
        Poi applico l'oversampling con il metodo SMOTE per sovracampionare la classe 1 e portarla a 2301, partendo dal dataset precedente
        sottocampionato.
        """

        ##########################
        #Non-Spam  2788  Classe: 0
        #Spam	  1813  Classe: 1
        ##########################

        #undersampling
        iht = InstanceHardnessThreshold(sampling_strategy={
            0: 2300 # facciamo undersampling della classe maggioritaria 0 portandola a 2300 record.
        },random_state=42)
        X_resampled, y_resampled = iht.fit_resample(tmp, y)
        #print('Dataset dopo il probabilistico undersampling %s' % sorted(Counter(y_resampled0).items()))  

        #oversampling
        sm = SMOTE(sampling_strategy={
            1: 2301 # facciamo oversampling della classe minoritaria 1, portandola a 2301 record
        },random_state=42)
        X_resampled, y_resampled = sm.fit_resample(X_resampled, y_resampled)
        #print('Dataset dopo SMOTE oversampling %s' % sorted(Counter(y_resampled0).items()))
        X=X_resampled # riassegnamo al posto dataset originale
        y=y_resampled # riassegnamo al posto del dataset originale
    return X,y