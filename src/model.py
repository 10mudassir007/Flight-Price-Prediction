#from ydf import RandomForestLearner, DecisionTreeLearner, GradientBoostedTreesLearner, Task
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import os, pickle


# ===================== YDF MODELS =====================

# def train_ydf_random_forest(train_df, test_df):
#     model = RandomForestLearner(label="price", task=Task.REGRESSION).train(train_df)
#     r2 = r2_score(test_df['price'], model.predict(test_df.drop('price', axis=1)))
#     return model, r2

# def train_ydf_gradient_boosted(train_df, test_df):
#     model = GradientBoostedTreesLearner(label="price", task=Task.REGRESSION).train(train_df)
#     r2 = r2_score(test_df['price'], model.predict(test_df.drop('price', axis=1)))
#     return model, r2

# def train_ydf_decision_tree(train_df, test_df):
#     model = DecisionTreeLearner(label="price", task=Task.REGRESSION).train(train_df)
#     r2 = r2_score(test_df['price'], model.predict(test_df.drop('price', axis=1)))
#     return model, r2


# ===================== SKLEARN / OTHER MODELS =====================


def train_xgboost(X_train, y_train, X_test, y_test):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    return model, r2

def train_catboost(X_train, y_train, X_test, y_test):
    model = CatBoostRegressor(verbose=0)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    return model, r2

def train_lightgbm(X_train, y_train, X_test, y_test):
    model = LGBMRegressor(verbose=0)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    return model, r2

def train_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    return model, r2

def train_decision_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    return model, r2

def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    return model, r2


# ===================== SAVE HELPERS =====================

def save_model(model, model_name, folder="models/"):
    """Save trained model to disk."""
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"{model_name}.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Saved model to {filepath}")
