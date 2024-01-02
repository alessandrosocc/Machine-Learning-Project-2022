from kNN import kNN
from dTree import dTree
from NBayesCustom import myNBayes
from RandomForest import RandomForest
from EnsembleCustom import myEnsemble
from clear_terminal import clear_terminal
from DataAnalysis import DataAnalysis

def main():
    clear_terminal()
    print("----------------------")
    print("1 | kNearestNeighbour")
    print("----------------------")
    print("2 | decisionalTree")
    print("----------------------")
    print("3 | NaiveBayes")
    print("----------------------")
    print("4 | RandomForest")
    print("----------------------")
    print("5 | Ensemble")
    print("----------------------")
    print("6 | Analisi dei dati")
    print("----------------------")
    print("Che operazione vuoi fare?")
    s = int(input())
    if s == 1:
        kNN()
    elif s == 2:
        dTree()
    elif s == 3:
        myNBayes()
    elif s == 4:
        RandomForest()
    elif s == 5:
        myEnsemble()
    elif s == 6:
        DataAnalysis()

if __name__ == "__main__":
    main()
