from libraries import *
from clear_terminal import clear_terminal

def preprocessing(X, y, df, features, trasformazione, modifica, bilanciamento):
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
    
    if trasformazione == "MinMaxScaler":
        minMaxScaler = MinMaxScaler()
        minMaxScaler.fit(X)
        minMaxScaler.transform(X)
    elif trasformazione == "StandardScaler":
        zScore = StandardScaler()
        zScore.fit(X)
        zScore.transform(X)
    elif trasformazione == "Normalizzazione":
        normalize(X, norm = 'l1')

    #FEATURE SELECTION/AGGREAGATION
    
    if modifica == "VarianceThreshold":
        #VarianceThreshold
        selector = VarianceThreshold(threshold=1)
        X = selector.fit_transform(X)
    elif modifica == "Scoring con chi2":
        #scoring con chi2
        X = SelectKBest(chi2, k=28).fit_transform(X, y)
    elif modifica == "Scoring con mutual_info_classif":
        #scoring con mutual_info_classif
        X = SelectKBest(mutual_info_classif, k=28).fit_transform(X, y)
    elif modifica == "SequentialFeatureSelector":
        clear_terminal()
        print("loading...")
        #SequentialFeatureSelector
        sfs = SequentialFeatureSelector(KNeighborsClassifier(n_neighbors=1), n_features_to_select=28, n_jobs= -1)
        X = sfs.fit_transform(X, y)
    elif modifica == "FeatureSelection con matrice di correlazione":
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
    if modifica == "SparseRandomProjection":
        srp = SparseRandomProjection(57)
        X = srp.fit_transform(X)
    elif modifica == "GaussianRandomProjection":
        grp = GaussianRandomProjection(57)
        X = grp.fit_transform(X)
    elif modifica == "FeatureAgglomeration":
        fa = FeatureAgglomeration()
        X = fa.fit_transform(X)
    elif modifica == "PrincipalComponentsAnalysis":
        pca = PCA()
        X = pca.fit_transform(X)

    #OVERSAMPLING
    clear_terminal()
    if bilanciamento == "RandomOverSampler (Oversampling)":
        ros = RandomOverSampler(random_state=42)
        X, y = ros.fit_resample(X, y)
    elif bilanciamento == "SMOTE (Oversampling)":
        sm = SMOTE(random_state=42)
        X, y = sm.fit_resample(X, y)
    elif bilanciamento == "ADASYN (Oversampling)":
        ada = ADASYN(random_state=42)
        X, y = ada.fit_resample(X, y)

    #UNDERSAMPLING
    elif bilanciamento == "RandomUnderSampler (Undersampling)":
        rus = RandomUnderSampler(random_state=0)
        X, y = rus.fit_resample(X, y)
    elif bilanciamento == "InstanceHardnessThreshold (Undersampling)":
        iht = InstanceHardnessThreshold()
        X, y = iht.fit_resample(X, y)
    elif bilanciamento == "NearMiss v.1 (Undersampling)":
        nm = NearMiss(version=1)
        X, y = nm.fit_resample(X, y)
    elif bilanciamento == "NearMiss v.2 (Undersampling)":
        nm = NearMiss(version=2)
        X, y = nm.fit_resample(X, y)
    elif bilanciamento == "ClusterCentroids (Undersampling)":
        cc = ClusterCentroids(random_state=0)
        X, y = cc.fit_resample(X, y)
    elif bilanciamento == "Combinazione di Undersampling e Oversampling":
        clear_terminal()
        print("Oversampling della classe 0 che da 2788 passa a 1843 oggetti")
        print("Undersampling della classe 1 che da 1813 passa a 2788 oggetti\n\n")
        under=X.copy()
        over=X.copy()


        """
        Faccio undersampling e oversampling separatamente
        Faccio undersampling della classe maggioritaria (ossia classe 0 (non spam) 2788 istanze) e
        Oversampling della classe minoritaria  (ossia classe 1 (spam) 1813 istanze)

        Faccio queste due operazioni separatamente su una copia del dataset originale

        Dopodichè per le due copie tengo solo le classi per cui mi interessava fare undersampling e oversampling

        Le unisco, avrò quindi che la classe maggioritaria sarà ora quella che nel dataset originale era minoritaria
        e la classe minoritaria sarà quella che nel dataset originale era maggioritaria
        """
        #undersampling
        iht = InstanceHardnessThreshold()
        X_resampled1, y_resampled1 = iht.fit_resample(under, y)
        #print('Dataset dopo il probabilistico undersampling %s' % sorted(Counter(y_resampled1).items()))  

        #oversampling
        sm = SMOTE(random_state=42)
        X_resampled0, y_resampled0 = sm.fit_resample(over, y)
        #print('Dataset dopo SMOTE oversampling %s' % sorted(Counter(y_resampled0).items()))
        ##########################
        #Spam	  1813  Classe: 1
        #Non-Spam  2788  Classe: 0
        X_resampled1[y_resampled1==0] # oversampling della classe 0 che da 2788 passa a 1843
        X_resampled0[y_resampled0==1] # undersampling della classe 1 che da 1813 passa a 2788
        dfResampled=pd.concat([pd.DataFrame(X_resampled1[y_resampled1==0]),pd.DataFrame(X_resampled0[y_resampled0==1])])

        XNew=dfResampled.to_numpy()
        yNew=np.concatenate([y_resampled1[y_resampled1==0],y_resampled0[y_resampled0==1]])

        X=XNew
        y=yNew
    return X,y