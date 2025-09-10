# import os
# import logging
# from logging import getLogger

# import logging_config  # <- your logging.py at project root

# from src.data import (
#     load_dataset,
#     add_coordinates,
#     prepare_features,
#     train_test_split_data,
#     save_processed_data,
#     encode_data
# )
# from src.model import (
#     train_linear_regression,
#     train_decision_tree,
#     train_random_forest,
#     train_xgboost,
#     train_catboost,
#     train_lightgbm,
#     save_model
# )


# def main():
#     # init logging once
#     logging_config.setup_logging()
#     logger = getLogger(__name__)

#     # 1. Load raw dataset
#     logger.info("Loading dataset...")
#     df = load_dataset()

#     # 2. Add coordinates
#     logger.info("Adding coordinates...")
#     df = add_coordinates(df)

#     # 3. Prepare features
#     logger.info("Preparing features...")
#     df, cols_to_encode = prepare_features(df)

#     # 4. Save processed dataset
#     logger.info("Saving processed data...")
#     save_processed_data(df)

#     # 5. Train/test split
#     logger.info("Splitting into train/test...")
#     train_df, test_df = train_test_split_data(df)

#     # Separate features & target
#     X_train, y_train = train_df.drop("price", axis=1), train_df["price"]
#     X_test, y_test = test_df.drop("price", axis=1), test_df["price"]

#     # Encode categorical variables
#     logger.info("Encoding categorical features...")
#     X_train, X_test = encode_data(X_train, X_test, cols_to_encode)

#     models = {}

#     logger.info("Training Base Models (Sklearn)...")

#     lin_model, lin_r2 = train_linear_regression(X_train, y_train, X_test, y_test)
#     logger.info(f"LinearRegression R²: {lin_r2:.4f}")
#     models["LR"] = lin_r2
#     save_model(lin_model, "linear_regression")

#     dt_model, dt_r2 = train_decision_tree(X_train, y_train, X_test, y_test)
#     logger.info(f"DecisionTree R²: {dt_r2:.4f}")
#     models["DT"] = dt_r2
#     save_model(dt_model, "decision_tree")

#     rf_model, rf_r2 = train_random_forest(X_train, y_train, X_test, y_test)
#     logger.info(f"RandomForest R²: {rf_r2:.4f}")
#     models["RF"] = rf_r2
#     save_model(rf_model, "random_forest")

#     logger.info("Training Boosting Models...")

#     xgb_model, xgb_r2 = train_xgboost(X_train, y_train, X_test, y_test)
#     logger.info(f"XGBoost R²: {xgb_r2:.4f}")
#     models["XGB"] = xgb_r2
#     save_model(xgb_model, "xgboost")

#     cat_model, cat_r2 = train_catboost(X_train, y_train, X_test, y_test)
#     logger.info(f"CatBoost R²: {cat_r2:.4f}")
#     models["CBM"] = cat_r2
#     save_model(cat_model, "catboost")

#     lgbm_model, lgbm_r2 = train_lightgbm(X_train, y_train, X_test, y_test)
#     logger.info(f"LightGBM R²: {lgbm_r2:.4f}")
#     models["LGBM"] = lgbm_r2
#     save_model(lgbm_model, "lightgbm")

#     logger.info("Training pipeline completed successfully!")


# if __name__ == "__main__":
#     main()


import os
import logging
from logging import getLogger

import mlflow
import mlflow.sklearn

import logging_config  # your logging config file in project root

from src.data import (
    load_dataset,
    add_coordinates,
    prepare_features,
    train_test_split_data,
    save_processed_data,
    encode_data
)
from src.model import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    train_xgboost,
    train_catboost,
    train_lightgbm,
    save_model
)


def main():
    logging_config.setup_logging()
    logger = getLogger(__name__)

    mlflow.set_tracking_uri("http://127.0.0.1:5000") 
    mlflow.set_experiment("flight_price_prediction")

    logger.info("Loading dataset...")
    df = load_dataset()

    logger.info("Adding coordinates...")
    df = add_coordinates(df)

    logger.info("Preparing features...")
    df, cols_to_encode = prepare_features(df)

    logger.info("Saving processed data...")
    save_processed_data(df)

    logger.info("Splitting into train/test...")
    train_df, test_df = train_test_split_data(df)

    X_train, y_train = train_df.drop("price", axis=1), train_df["price"]
    X_test, y_test = test_df.drop("price", axis=1), test_df["price"]

    logger.info("Encoding categorical features...")
    X_train, X_test = encode_data(X_train, X_test, cols_to_encode)

    logger.info("Training Base Models...")
    logger.info("Training Linear Regression...")
    
    with mlflow.start_run(run_name="LinearRegression"):
        lin_model, lin_r2 = train_linear_regression(X_train, y_train, X_test, y_test)
        logger.info(f"LinearRegression R²: {lin_r2:.4f}")
        mlflow.log_metric("r2", lin_r2)
        mlflow.sklearn.log_model(lin_model, "model")
        save_model(lin_model, "linear_regression")

    logger.info("Training Decision Tree...")
    with mlflow.start_run(run_name="DecisionTree"):
        dt_model, dt_r2 = train_decision_tree(X_train, y_train, X_test, y_test)
        logger.info(f"DecisionTree R²: {dt_r2:.4f}")
        mlflow.log_metric("r2", dt_r2)
        mlflow.sklearn.log_model(dt_model, "model")
        save_model(dt_model, "decision_tree")

    logger.info("Training Random Forest...")
    with mlflow.start_run(run_name="RandomForest"):
        rf_model, rf_r2 = train_random_forest(X_train, y_train, X_test, y_test)
        logger.info(f"RandomForest R²: {rf_r2:.4f}")
        mlflow.log_metric("r2", rf_r2)
        mlflow.sklearn.log_model(rf_model, "model")
        save_model(rf_model, "random_forest")

    logger.info("Training Boosting Models...")
    logger.info("Training XGBoost...")
    with mlflow.start_run(run_name="XGBoost"):
        xgb_model, xgb_r2 = train_xgboost(X_train, y_train, X_test, y_test)
        logger.info(f"XGBoost R²: {xgb_r2:.4f}")
        mlflow.log_metric("r2", xgb_r2)
        mlflow.sklearn.log_model(xgb_model, "model")
        save_model(xgb_model, "xgboost")

    logger.info("Training CatBoost...")
    with mlflow.start_run(run_name="CatBoost"):
        cat_model, cat_r2 = train_catboost(X_train, y_train, X_test, y_test)
        logger.info(f"CatBoost R²: {cat_r2:.4f}")
        mlflow.log_metric("r2", cat_r2)
        mlflow.sklearn.log_model(cat_model, "model")
        save_model(cat_model, "catboost")

    logger.info("Training LightGBM...")
    with mlflow.start_run(run_name="LightGBM"):
        lgbm_model, lgbm_r2 = train_lightgbm(X_train, y_train, X_test, y_test)
        logger.info(f"LightGBM R²: {lgbm_r2:.4f}")
        mlflow.log_metric("r2", lgbm_r2)
        mlflow.sklearn.log_model(lgbm_model, "model")
        save_model(lgbm_model, "lightgbm")

    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
