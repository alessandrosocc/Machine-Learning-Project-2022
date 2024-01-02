
from libraries import * # importo tutte le librerie presenti nel file libraries
import sys
from kNN import kNN # importo dal file corrispondente tutti i calssificatori che sono stati implementati per questo progetto
from dTree import dTree
from NBayesCustom import myNBayes
from RandomForest import RandomForest
from EnsembleCustom import myEnsemble


class Ui_classificationWindow(object): 
    def setupUi(self, classificationWindow): # inizializzo l'oggetto Classification Window
        classificationWindow.setObjectName("classificationWindow")
        classificationWindow.resize(840, 600) # definisco le dimensioni della finestra che andrò a generare
        self.centralwidget = QtWidgets.QWidget(classificationWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget) 
        self.gridLayout.setObjectName("gridLayout")
        self.preProcessingBox = QtWidgets.QGroupBox(self.centralwidget) # inizializzo la group box che andrà a contenere tutti gli input relativi al pre processing, questa poi verrà assegnata al grid layout che divide la finestra in tre parti scalabili
        font = QtGui.QFont()
        font.setPointSize(13) # setto il font che verrà rispettato da tutti gli elementi  all'interno della groupBox
        self.preProcessingBox.setFont(font)
        self.preProcessingBox.setAlignment(QtCore.Qt.AlignCenter)
        self.preProcessingBox.setObjectName("preProcessingBox")
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(self.preProcessingBox)
        self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(20, 30, 211, 81))
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_5)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.trasformazioniLabel = QtWidgets.QLabel(self.verticalLayoutWidget_5)
        self.trasformazioniLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.trasformazioniLabel.setObjectName("trasformazioniLabel")
        self.verticalLayout_5.addWidget(self.trasformazioniLabel)
        self.trasformazioneComboBox = QtWidgets.QComboBox(self.verticalLayoutWidget_5)
        self.trasformazioneComboBox.setObjectName("trasformazioneComboBox") # inizializzo la combo Box per le trasformazioni lasciando liberi 4 spazi nei quali posso inserire delle opzioni di input
        self.trasformazioneComboBox.addItem("")
        self.trasformazioneComboBox.addItem("")
        self.trasformazioneComboBox.addItem("")
        self.trasformazioneComboBox.addItem("")
        self.verticalLayout_5.addWidget(self.trasformazioneComboBox)
        self.verticalLayoutWidget_8 = QtWidgets.QWidget(self.preProcessingBox)
        self.verticalLayoutWidget_8.setGeometry(QtCore.QRect(20, 110, 211, 80))
        self.verticalLayoutWidget_8.setObjectName("verticalLayoutWidget_8")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_8)
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.selezioneLabel = QtWidgets.QLabel(self.verticalLayoutWidget_8)
        self.selezioneLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.selezioneLabel.setObjectName("selezioneLabel")
        self.verticalLayout_8.addWidget(self.selezioneLabel)
        self.modificaComboBox = QtWidgets.QComboBox(self.verticalLayoutWidget_8)
        self.modificaComboBox.setObjectName("modificaComboBox") # inizializzo la combo box per la modifica nella quale lascio liberi 10 spazi che conterranno altrettante opzioni di input
        self.modificaComboBox.addItem("")
        self.modificaComboBox.addItem("")
        self.modificaComboBox.addItem("")
        self.modificaComboBox.addItem("")
        self.modificaComboBox.addItem("")
        self.modificaComboBox.addItem("")
        self.modificaComboBox.addItem("")
        self.modificaComboBox.addItem("")
        self.modificaComboBox.addItem("")
        self.modificaComboBox.addItem("")
        self.verticalLayout_8.addWidget(self.modificaComboBox)
        self.verticalLayoutWidget_13 = QtWidgets.QWidget(self.preProcessingBox)
        self.verticalLayoutWidget_13.setGeometry(QtCore.QRect(20, 190, 211, 80))
        self.verticalLayoutWidget_13.setObjectName("verticalLayoutWidget_13")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_13)
        self.verticalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.bilanciamentoLabel = QtWidgets.QLabel(self.verticalLayoutWidget_13)
        self.bilanciamentoLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.bilanciamentoLabel.setObjectName("bilanciamentoLabel")
        self.verticalLayout_10.addWidget(self.bilanciamentoLabel)
        self.bilanciamentoComboBox = QtWidgets.QComboBox(self.verticalLayoutWidget_13)
        self.bilanciamentoComboBox.setObjectName("bilanciamentoComboBox") # inizializzo la combo box per il bilanciamento nella quale lascio liberi 10 spazi che conterranno altrettante opzioni di input
        self.bilanciamentoComboBox.addItem("")
        self.bilanciamentoComboBox.addItem("")
        self.bilanciamentoComboBox.addItem("")
        self.bilanciamentoComboBox.addItem("")
        self.bilanciamentoComboBox.addItem("")
        self.bilanciamentoComboBox.addItem("")
        self.bilanciamentoComboBox.addItem("")
        self.bilanciamentoComboBox.addItem("")
        self.bilanciamentoComboBox.addItem("")
        self.bilanciamentoComboBox.addItem("")
        self.verticalLayout_10.addWidget(self.bilanciamentoComboBox)
        self.gridLayout.addWidget(self.preProcessingBox, 1, 0, 2, 1)
        self.invioButton = QtWidgets.QPushButton(self.centralwidget) # inizializzo il bottone di invio che avrà il compito di stamparmi graficamente il risultato dei classificatori
        self.invioButton.setObjectName("invioButton")
        self.gridLayout.addWidget(self.invioButton, 3, 1, 1, 1) # gli assegno una posizione relativa nel grid layout
        self.classificatoriBox = QtWidgets.QGroupBox(self.centralwidget) # definisco una nuova group  boc che conterrà tutti gli input che mi permettono di scegliere un classificatore da un'altro
        font = QtGui.QFont()
        font.setPointSize(13) # setto il font che verrà rispettato da tutti gli elementi  all'interno della groupBox
        self.classificatoriBox.setFont(font) # assegno il font appena definito
        self.classificatoriBox.setAlignment(QtCore.Qt.AlignCenter)
        self.classificatoriBox.setObjectName("classificatoriBox") # do il nome all'oggetto
        
        self.gridLayout_3 = QtWidgets.QGridLayout(self.classificatoriBox) # assegno al layout la nuova group box appena definita 
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.inserisciPesiLabel = QtWidgets.QLabel(self.classificatoriBox)
        self.inserisciPesiLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.inserisciPesiLabel.setObjectName("inserisciPesiLabel")
        self.gridLayout_3.addWidget(self.inserisciPesiLabel, 3, 0, 1, 1)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.spinBox_2 = QtWidgets.QSpinBox(self.classificatoriBox) #inizializzo 3 spin box che mi saranno utili per specificare i pesi nel caso in cui l'utente scelga di utilizzare un classificatore multiplo ensamble custom pesato
        self.spinBox_2.setObjectName("spinBox_2") 
        self.gridLayout_2.addWidget(self.spinBox_2, 0, 1, 1, 1)
        self.spinBox = QtWidgets.QSpinBox(self.classificatoriBox)
        self.spinBox.setObjectName("spinBox")
        self.gridLayout_2.addWidget(self.spinBox, 0, 0, 1, 1)
        self.spinBox_3 = QtWidgets.QSpinBox(self.classificatoriBox)
        self.spinBox_3.setObjectName("spinBox_3")
        self.gridLayout_2.addWidget(self.spinBox_3, 0, 2, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 4, 0, 1, 1)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.votingBox = QtWidgets.QComboBox(self.classificatoriBox) # definisco una combo box che mi permetterà di scegliere tra hard e soft voting
        self.votingBox.setObjectName("votingBox")
        self.votingBox.addItem("")
        self.votingBox.addItem("")
        self.gridLayout_4.addWidget(self.votingBox, 1, 0, 1, 1)
        self.pesoBox = QtWidgets.QComboBox(self.classificatoriBox)
        self.pesoBox.setObjectName("pesoBox") # definisco una combo box che mi permetterà di scegliere tra pesare e non pesare il voting scelto
        self.pesoBox.addItem("")
        self.pesoBox.addItem("")
        self.gridLayout_4.addWidget(self.pesoBox, 2, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.classificatoriBox)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout_4.addWidget(self.label, 0, 0, 1, 1) 
        self.gridLayout_3.addLayout(self.gridLayout_4, 2, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout() # vado a definirmi un vertical layout per allinerare tutti i radio button, che mi permetteranno di scegliere il classificatore da utilizzare
        self.verticalLayout.setObjectName("verticalLayout")
        self.kNNButton = QtWidgets.QRadioButton(self.classificatoriBox) # inizializzo il bottone corrispondente al classificatore Knn
        self.kNNButton.setObjectName("kNNButton") 
        self.verticalLayout.addWidget(self.kNNButton)
        self.decisionTreeButton = QtWidgets.QRadioButton(self.classificatoriBox)  # inizializzo il bottone corrispondente al classificatore decision tree
        self.decisionTreeButton.setObjectName("decisionTreeButton")
        self.verticalLayout.addWidget(self.decisionTreeButton)
        self.bayesButton = QtWidgets.QRadioButton(self.classificatoriBox)  # inizializzo il bottone corrispondente al classificatore bayesCustom
        self.bayesButton.setObjectName("bayesButton")
        self.verticalLayout.addWidget(self.bayesButton)
        self.randomForestButton = QtWidgets.QRadioButton(self.classificatoriBox)  # inizializzo il bottone corrispondente al classificatore random forest
        self.randomForestButton.setObjectName("randomForestButton")
        self.verticalLayout.addWidget(self.randomForestButton)
        self.ensambleButton = QtWidgets.QRadioButton(self.classificatoriBox)  # inizializzo il bottone corrispondente al classificatore ensamble custom
        self.ensambleButton.setObjectName("ensambleButton")
        self.verticalLayout.addWidget(self.ensambleButton)
        self.gridLayout_3.addLayout(self.verticalLayout, 1, 0, 1, 1)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.selezioneLabel_2 = QtWidgets.QLabel(self.classificatoriBox)
        self.selezioneLabel_2.setAlignment(QtCore.Qt.AlignCenter)
        self.selezioneLabel_2.setObjectName("selezioneLabel_2")
        self.verticalLayout_6.addWidget(self.selezioneLabel_2)
        self.label_5 = QtWidgets.QLabel(self.classificatoriBox)
        self.label_5.setText("")
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_6.addWidget(self.label_5)
        self.gridLayout_3.addLayout(self.verticalLayout_6, 0, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.classificatoriBox)
        self.label_9.setGeometry(QtCore.QRect(40, 350, 161, 20))
        self.label_9.setText("")
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.classificatoriBox, 0, 1, 3, 1) # dopo aver concluso l'inizializzazione a della classificatori group box la assegno al grid layout che è gerarchicamente superiore
        
        self.confrontoBox = QtWidgets.QGroupBox(self.centralwidget) # inizializzo l'ultima gropu box, ovvero quella per il confronto
        font = QtGui.QFont()
        font.setPointSize(13)
        self.confrontoBox.setFont(font) # assegno il font appena definito 
        self.confrontoBox.setAlignment(QtCore.Qt.AlignCenter)
        self.confrontoBox.setObjectName("confrontoBox")
        self.rocLabel = QtWidgets.QLabel(self.confrontoBox)
        self.rocLabel.setGeometry(QtCore.QRect(10, 130, 231, 201)) # definisco la geometria della label all'interno della quale inserirò l'immagine della curva roc
        self.rocLabel.setText("")
        self.rocLabel.setScaledContents(True)
        self.rocLabel.setObjectName("rocLabel")
        self.verticalLayoutWidget_7 = QtWidgets.QWidget(self.confrontoBox)
        self.verticalLayoutWidget_7.setGeometry(QtCore.QRect(20, 30, 211, 93))
        self.verticalLayoutWidget_7.setObjectName("verticalLayoutWidget_7")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_7)
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.label_7 = QtWidgets.QLabel(self.verticalLayoutWidget_7)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_9.addWidget(self.label_7)
        self.noPreButton = QtWidgets.QPushButton(self.verticalLayoutWidget_7) # definisco due bottoni che mi permetteranno di scegliere quale delle due curve roc visualizzare una senza pre processing e una con il preprocessing 
        self.noPreButton.setObjectName("noPreButton") 
        self.verticalLayout_9.addWidget(self.noPreButton)
        self.siPreButton = QtWidgets.QPushButton(self.verticalLayoutWidget_7)
        self.siPreButton.setObjectName("siPreButton")
        self.verticalLayout_9.addWidget(self.siPreButton)
        self.gridLayout.addWidget(self.confrontoBox, 1, 2, 2, 1)
        classificationWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(classificationWindow)
        QtCore.QMetaObject.connectSlotsByName(classificationWindow)
        
        # definisco gli event listeners per controllare quale dei radio button è triggerato in un certo momento
        self.kNNButton.clicked.connect(lambda:self.modifyLabel(1)) 
        self.decisionTreeButton.clicked.connect(lambda:self.modifyLabel(2))
        self.bayesButton.clicked.connect(lambda:self.modifyLabel(3))
        self.randomForestButton.clicked.connect(lambda:self.modifyLabel(4))
        self.ensambleButton.clicked.connect(lambda:self.modifyLabel(5))
        
        # definisco gli event listeners per i rimanenti bottoni, invio e per i bottoni di visualizzazione della curva roc
        self.invioButton.clicked.connect(self.invioClicked)
        self.noPreButton.clicked.connect(self.rocNoPre)
        self.siPreButton.clicked.connect(self.rocPre)

    def retranslateUi(self, classificationWindow): # in questa funzione definisco in testi che verranno visualizzati sulla finestra, per ogni elemento inserito: titolo della window, group box, combo box bottoni e label
        _translate = QtCore.QCoreApplication.translate
        classificationWindow.setWindowTitle(_translate("classificationWindow", "Finestra per la classificazione"))
        self.preProcessingBox.setTitle(_translate("classificationWindow", "Pre-processing"))
        self.trasformazioniLabel.setText(_translate("classificationWindow", "Trasformazioni"))
        self.trasformazioneComboBox.setItemText(0, _translate("classificationWindow", "Nessuna trasformazione"))
        self.trasformazioneComboBox.setItemText(1, _translate("classificationWindow", "MinMaxScaler"))
        self.trasformazioneComboBox.setItemText(2, _translate("classificationWindow", "StandardScaler"))
        self.trasformazioneComboBox.setItemText(3, _translate("classificationWindow", "Normalizzazione"))
        self.selezioneLabel.setText(_translate("classificationWindow", "Selezione / Aggregazione"))
        self.modificaComboBox.setItemText(0, _translate("classificationWindow", "Nessuna modifica degli attributi"))
        self.modificaComboBox.setItemText(1, _translate("classificationWindow", "SparseRandomProjection"))
        self.modificaComboBox.setItemText(2, _translate("classificationWindow", "GaussianRandomProjection"))
        self.modificaComboBox.setItemText(3, _translate("classificationWindow", "FeatureAgglomeration"))
        self.modificaComboBox.setItemText(4, _translate("classificationWindow", "PrincipalComponentsAnalysis"))
        self.modificaComboBox.setItemText(5, _translate("classificationWindow", "VarianceThreshold"))
        self.modificaComboBox.setItemText(6, _translate("classificationWindow", "Scoring con chi2"))
        self.modificaComboBox.setItemText(7, _translate("classificationWindow", "Scoring con mutual_info_classif"))
        self.modificaComboBox.setItemText(8, _translate("classificationWindow", "SequentialFeatureSelector"))
        self.modificaComboBox.setItemText(9, _translate("classificationWindow", "FeatureSelection con matrice di correlazione"))
        self.bilanciamentoLabel.setText(_translate("classificationWindow", "Bilanciamento"))
        self.bilanciamentoComboBox.setItemText(0, _translate("classificationWindow", "Nessun bilanciamento"))
        self.bilanciamentoComboBox.setItemText(1, _translate("classificationWindow", "Combinazione di Undersampling e Oversampling"))
        self.bilanciamentoComboBox.setItemText(2, _translate("classificationWindow", "RandomUnderSampler (Undersampling)"))
        self.bilanciamentoComboBox.setItemText(3, _translate("classificationWindow", "InstanceHardnessThreshold (Undersampling)"))
        self.bilanciamentoComboBox.setItemText(4, _translate("classificationWindow", "NearMiss v.1 (Undersampling)"))
        self.bilanciamentoComboBox.setItemText(5, _translate("classificationWindow", "NearMiss v.2 (Undersampling)"))
        self.bilanciamentoComboBox.setItemText(6, _translate("classificationWindow", "ClusterCentroids (Undersampling)"))
        self.bilanciamentoComboBox.setItemText(7, _translate("classificationWindow", "RandomOverSampler (Oversampling)"))
        self.bilanciamentoComboBox.setItemText(8, _translate("classificationWindow", "SMOTE (Oversampling)"))
        self.bilanciamentoComboBox.setItemText(9, _translate("classificationWindow", "ADASYN (Oversampling)"))
        self.invioButton.setText(_translate("classificationWindow", "Invio"))
        self.classificatoriBox.setTitle(_translate("classificationWindow", "Classificatori"))
        self.inserisciPesiLabel.setText(_translate("classificationWindow", "Inserisci i pesi "))
        self.votingBox.setItemText(0, _translate("classificationWindow", "Hard Voting"))
        self.votingBox.setItemText(1, _translate("classificationWindow", "Soft Voting"))
        self.pesoBox.setItemText(0, _translate("classificationWindow", "Non pesato"))
        self.pesoBox.setItemText(1, _translate("classificationWindow", "Pesato"))
        self.label.setText(_translate("classificationWindow", "Se utilizzi Ensamble (custom) seleziona:"))
        self.kNNButton.setText(_translate("classificationWindow", "kNN"))
        self.decisionTreeButton.setText(_translate("classificationWindow", "Decision Tree"))
        self.bayesButton.setText(_translate("classificationWindow", "Bayes (custom)"))
        self.randomForestButton.setText(_translate("classificationWindow", "Random Forest"))
        self.ensambleButton.setText(_translate("classificationWindow", "Ensamble (custom)"))
        
        self.selezioneLabel_2.setText(_translate("classificationWindow", "Hai selezionato:"))
        self.confrontoBox.setTitle(_translate("classificationWindow", "Confronto"))
        self.label_7.setText(_translate("classificationWindow", "Visualizzazione curva ROC"))
        self.noPreButton.setText(_translate("classificationWindow", "Senza pre-processing"))
        self.siPreButton.setText(_translate("classificationWindow", "Con pre-processing"))
        
        
        
    def modifyLabel(self,id): # tramite questa funzione vado a modificare la label in testa alla group box di classificazione aggiungendo in tempo reale il radio button selezionato
        title = self.obtainTitle(id) # seleziono il titolo del bottone triggerato in queto momento
        self.selezioneLabel_2.setText("\nYou have Selected: \n\n{"+str(title)+"}") # aggiungo il titolo del bottone alla label già presente
        self.label.show() # mostro la label
        
    def rocPre(self): # setto la pix map della curva roc corrispondente, in questo caso quella di una curva roc che contiene tutti i classificatori, che hanno classificato un dataset modificato con le tecniche di pre-processing che sono risultate più efficienti
        self.rocLabel.setPixmap(QtGui.QPixmap("WPreprocess.png"))
        
    def rocNoPre(self): # setto la pix map della curva roc corrispondente a una un'insieme di curve (una per ogni classificatore) in modo tale da confrontarne le prestazioni senza che nessuna tecnica di pre-processing sia stata utilizzata sul dataset.
        self.rocLabel.setPixmap(QtGui.QPixmap("noPreprocess.png"))
        
    def idClassifierCheck(self): # con questa funzione vado a controllare quale dei bottoni dei classificatori era triggerato al momento della pressione del tasto invio
        id = 0
        if self.randomForestButton.isChecked() == 1:
            id = 4
        elif self.decisionTreeButton.isChecked() == 1:
            id = 2
        elif self.kNNButton.isChecked() == 1:
            id = 1
        elif self.ensambleButton.isChecked() == 1:
            id = 5
        elif self.bayesButton.isChecked() == 1:
            id = 3
        return id
    
    def obtainMetrics(self, id): # tramite questa funzione richiamo le funzioni dei classificatori dagli altri file, i quali verranno addestrati, testati e le loro metriche saranno restituite con una return
        # con le successive 3 righe vado a prendermi l'input dalle combo box di pre processing
        tra = self.trasformazioneComboBox.currentText()
        mod = self.modificaComboBox.currentText()
        bil = self.bilanciamentoComboBox.currentText()
        bestParams = "" 
        if id == 1:
            metrics,bestParams = kNN(tra,mod,bil)
        elif id == 2:
            metrics,bestParams = dTree(tra,mod,bil)
        elif id == 3:
            metrics = myNBayes(tra,mod,bil)
        elif id == 4:
            metrics,bestParams = RandomForest(tra,mod,bil)
        elif id == 5:
            # nel caso in cui venga scelto il classificatore ensamble prendo anche gli input aggiuntivi di voting, tipo di peso e pesi
            voting = self.votingBox.currentText()
            weight = self.pesoBox.currentText()
            peso1 = self.spinBox.value()
            peso2 = self.spinBox_2.value()
            peso3 = self.spinBox_3.value()
            metrics = myEnsemble(tra,mod,bil,voting,weight,peso1,peso2,peso3)
        return metrics, bestParams
    
    def obtainTitle(self, id): # in base all'id selezionato ritorno alla funzione chiamante un titolo da stampare a video
        if id == 1:
            title = "kNearestNeighbour"
        elif id == 2:
            title = "decisionalTree"
        elif id == 3:
            title = "NaiveBayes"
        elif id == 4:
            title = "RandomForest"
        elif id == 5:
            title = "Ensemble"
        return title
    
    def messageBoxCreation(self,id): # questa funzione va a creare la message box nella quale si visualizzerà il risultato del classificatore selezionato
        bestParams = ""
        metrics, bestParams = self.obtainMetrics(id) # richiamo la funzione di classificazione
        title = self.obtainTitle(id) # ottengno il titolo da inserire tramite una funzione appostia
        msg = QMessageBox()  # definisco la message box
        msg.setText("Classificatore utilizzato: "+title) # setto il suo titolo
        if id == 3: # setto l'informative text, che varia un po in base al classificatore scelto 
            msg.setInformativeText("------------------------------------------------------\nPRE-PROCESSING:\n"+str(self.trasformazioneComboBox.currentText())+"\n"+str(self.modificaComboBox.currentText())+"\n"+str(self.bilanciamentoComboBox.currentText())+"\n------------------------------------------------------\nPERFORMANCES: "+title+"\n------------------------------------------------------\n"+metrics)
        elif id == 5:
            if self.pesoBox.currentText() == "Non pesato":
                msg.setInformativeText("------------------------------------------------------\nPRE-PROCESSING:\n"+str(self.trasformazioneComboBox.currentText())+"\n"+str(self.modificaComboBox.currentText())+"\n"+str(self.bilanciamentoComboBox.currentText())+"\n"+"------------------------------------------------------\nSELEZIONI ENSAMBLE:"+"\nVOTING: "+str(self.votingBox.currentText())+"\nPESO: "+str(self.pesoBox.currentText())+"\n------------------------------------------------------\nPERFORMANCES: "+title+"\n------------------------------------------------------\n"+metrics)
            else:
                msg.setInformativeText("------------------------------------------------------\nPRE-PROCESSING:\n"+str(self.trasformazioneComboBox.currentText())+"\n"+str(self.modificaComboBox.currentText())+"\n"+str(self.bilanciamentoComboBox.currentText())+"\n"+"------------------------------------------------------\nSELEZIONI ENSAMBLE:"+"\nVOTING: "+str(self.votingBox.currentText())+"\nPESO: "+str(self.pesoBox.currentText())+"\nPESI: "+str(self.spinBox.value())+ " | " + str(self.spinBox_2.value())+" | "+str(self.spinBox_3.value())+"\n------------------------------------------------------\nPERFORMANCES: "+title+"\n------------------------------------------------------\n"+metrics)
        else:
            msg.setInformativeText("------------------------------------------------------\nPRE-PROCESSING:\n"+str(self.trasformazioneComboBox.currentText())+"\n"+str(self.modificaComboBox.currentText())+"\n"+str(self.bilanciamentoComboBox.currentText())+"\n"+"------------------------------------------------------\nBEST PARAMETERS:\n"+str(bestParams)+"\n------------------------------------------------------\nPERFORMANCES: "+title+"\n------------------------------------------------------\n"+metrics)
        msg.setStyleSheet("#qt_msgbox_informativelabel{font-size: 13px} #qt_msgbox_label{font-size: 15px}") # tramite un richiamo di css setto il font per ogni parte della message box
        x = msg.exec_()

    def invioClicked(self): # questa è la prima funzione richiamata dopo il click del bottone invio e andrà a controllare se un radio button è selezionato
        id = self.idClassifierCheck() # il controllo viene fatto tramite id 
        if id == 0:
            self.erroreInvio()
        else:
            self.messageBoxCreation(id) # nel caso in cui si abbia premuto il pulsante invio senza selezionare nessun classificatore viene stampata una finestra di errore
            
    def erroreInvio(self): # generazione della finestra di errore
        msg = QMessageBox() # inizializzo la message box di errore
        msg.setWindowTitle("ERRORE") # setto il titolo
        msg.setText("ERRORE L'INVIO NON É ANDATO A BUON FINE") # setto il testo
        msg.setIcon(QMessageBox.Critical)
        msg.setDefaultButton(QMessageBox.Ignore)
        msg.setInformativeText("É obbligatorio scegliere un classificatore\n quando si preme il tasto invio") # setto del testo informativo con qualche dettaglio in più   
        x = msg.exec_()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    classificationWindow = QtWidgets.QMainWindow() # vado a inzializzare un oggetto main window
    ui = Ui_classificationWindow()
    ui.setupUi(classificationWindow)
    classificationWindow.show()
    sys.exit(app.exec_()) # queste righe mi permettono di chiudere senza problemi la finestra premendo la x 
