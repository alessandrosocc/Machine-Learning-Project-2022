from libraries import *
from csv_open import csv_open
from metrics import metrics
from clear_terminal import clear_terminal


def RandomForest(tra,mod,bil):
    X, y, dataType, features = csv_open(tra,mod,bil)
    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=0, test_size=0.25, stratify=y)
    
    clear_terminal()
    print("loading...")

    rf = RandomForestClassifier()
    params={"n_estimators":list(range(10,80,10)),
            "criterion":["gini","entropy","log_loss"],
            "max_depth":[None]+list(range(4,34,10)),
            "max_features":["sqrt","log2"]
            }

    #grid search
    rf_grid = GridSearchCV(estimator=rf, cv=10,param_grid=params,n_jobs=-1)
    results = rf_grid.fit(train_x, train_y)

    #best model
    rf_best = rf.set_params(**results.best_params_)
    rf_best.fit(train_x, train_y)
    pred_y = rf_best.predict(test_x)

    clear_terminal()
#     print("---------------------------")
#     print("BEST PARAMETERS:")
#     print("---------------------------")
#     print(results.best_params_)
#     print("---------------------------")
#     print("PERFORMANCES RandomForest:")
#     print("---------------------------")
    return metrics(test_y, pred_y, np.unique(test_y)), results.best_params_

