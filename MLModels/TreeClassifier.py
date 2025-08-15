from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from MLModels.KClassifier import test_score, get_train_test_data

csvfile = "./Data/WESAD_filtered.csv"

feature_names = ["MeanSCR", "MaxSCR", "MinSCR", "RangeSCR", "SkewenessSCR", "KurtosisSCR"]

def train_random_forest_model(k_folds=10):

    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(class_weight="balanced", random_state=5, n_jobs=-1))])

    X_train, y_train, groups_train, X_test, y_test, groups_test, y = get_train_test_data(csvfile, feature_names)

    param_grid = {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_leaf": [1, 3, 5]
    }

    cross_valid = GroupKFold(k_folds)
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro"
    }
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cross_valid, scoring=scoring, refit="f1_macro", n_jobs=-1)

    grid_search.fit(X_train, y_train, groups=groups_train)

    print("Best parameters:", grid_search.best_params_)

    best_model = grid_search.best_estimator_

    importances = best_model.named_steps['clf'].feature_importances_
    for name, imp in zip(feature_names, importances):
        print(name, imp)

    test_score(best_model, X_test, y_test, y, "file.txt")
    return best_model


def train_xgboost(k_folds=10):
    pipeline = Pipeline([("scaler", StandardScaler()),
                         ("clf", XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False))])

    X_train, y_train, groups_train, X_test, y_test, groups_test, y = get_train_test_data(csvfile, feature_names)

    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [4, 6]
    }

    cross_valid = GroupKFold(k_folds)
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro"
    }
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cross_valid, scoring=scoring,
                               refit="f1_macro", n_jobs=-1)

    grid_search.fit(X_train, y_train, groups=groups_train)

    print("Best parameters:", grid_search.best_params_)

    best_model = grid_search.best_estimator_

    importances = best_model.named_steps['clf'].feature_importances_
    for name, imp in zip(feature_names, importances):
        print(name, imp)

    test_score(best_model, X_test, y_test, y, "file.txt")
    return best_model


train_random_forest_model()
