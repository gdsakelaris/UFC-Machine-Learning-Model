# UFC Predictor Models - Parameter Comparison

## **📊 MODEL PARAMETERS COMPARISON TABLE**

| **Parameter**                               | **exe Base** | **Two Runs** | **Class Weighting** | **Data Augmentation** |
| ------------------------------------------------- | ------------------ | ------------------ | ------------------------- | --------------------------- |
| **🎯 XGBOOST PARAMETERS**                   |                    |                    |                           |                             |
| **n_estimators**                            | 600                | 600                | 800                       | 500                         |
| **max_depth**                               | 8                  | 8                  | 10                        | 8                           |
| **learning_rate**                           | 0.025              | 0.02               | 0.015                     | 0.025                       |
| **subsample**                               | 0.85               | 0.8                | 0.85                      | 0.85                        |
| **colsample_bytree**                        | 0.8                | 0.8                | 0.85                      | 0.85                        |
| **reg_alpha**                               | 0.2                | 0.2                | 0.1                       | 0.2                         |
| **reg_lambda**                              | 1                  | 1                  | 0.8                       | 1                           |
| **min_child_weight**                        | 4                  | 4                  | 3                         | 4                           |
| **gamma**                                   | 0.15               | 0.15               | 0.15                      | 0.15                        |
| **early_stopping_rounds**                   | -                  | -                  | -                         | 50                          |
| **🎯 LIGHTGBM PARAMETERS**                  |                    |                    |                           |                             |
| **n_estimators**                            | 600                | 600                | 800                       | 500                         |
| **max_depth**                               | 8                  | 8                  | 10                        | 8                           |
| **learning_rate**                           | 0.025              | 0.02               | 0.015                     | 0.025                       |
| **num_leaves**                              | 60                 | 60                 | 80                        | 50                          |
| **subsample**                               | 0.85               | 0.8                | 0.85                      | 0.85                        |
| **colsample_bytree**                        | 0.8                | 0.8                | 0.85                      | 0.85                        |
| **reg_alpha**                               | 0.2                | 0.2                | 0.1                       | 0.2                         |
| **reg_lambda**                              | 1                  | 1                  | 0.8                       | 1                           |
| **min_child_weight**                        | 4                  | 4                  | 3                         | 4                           |
| **early_stopping_rounds**                   | -                  | -                  | -                         | 50                          |
| **🎯 CATBOOST PARAMETERS**                  |                    |                    |                           |                             |
| **iterations**                              | 600                | 600                | 800                       | 500                         |
| **depth**                                   | 8                  | 8                  | 10                        | 8                           |
| **learning_rate**                           | 0.025              | 0.02               | 0.015                     | 0.025                       |
| **l2_leaf_reg**                             | 1                  | 1                  | 0.5                       | 1                           |
| **early_stopping_rounds**                   | -                  | -                  | -                         | 50                          |
| **🎯 RANDOM FOREST PARAMETERS**             |                    |                    |                           |                             |
| **n_estimators**                            | 600                | 600                | 800                       | 500                         |
| **max_depth**                               | 20                 | 20                 | 25                        | 15                          |
| **min_samples_split**                       | 8                  | 8                  | 6                         | 6                           |
| **min_samples_leaf**                        | 2                  | 2                  | 2                         | 2                           |
| **🎯 NEURAL NETWORK PARAMETERS**            |                    |                    |                           |                             |
| **hidden_layers**                           | (256, 128, 64)     | (256, 128, 64)     | (256, 128, 64)            | (256, 128)                  |
| **learning_rate**                           | adaptive           | adaptive           | adaptive                  | adaptive                    |
| **max_iter**                                | 500                | 500                | 500                       | 300                         |
| **batch_size**                              | 32                 | 32                 | 32                        | 32                          |
| **alpha**                                   | 0.001              | 0.001              | 0.001                     | 0.001                       |
| **early_stopping**                          | True               | True               | True                      | True                        |
| **🎯 CROSS-VALIDATION & FEATURE SELECTION** |                    |                    |                           |                             |
| **TimeSeriesSplit folds**                   | 5                  | 5                  | 7                         | 5                           |
| **Feature Selection (Main)**                | 75%                | 75%                | 85%                       | 60%                         |
| **Feature Selection (RF)**                  | 75%                | 75%                | 85%                       | 60%                         |
| **Feature Selection (Method)**              | 75%                | 75%                | 85%                       | 60%                         |
| **Stacking CV folds**                       | 5                  | 5                  | 5                         | 5                           |
| **Calibration CV folds**                    | 3                  | 3                  | 3                         | 3                           |
| **🎯 META-LEARNER PARAMETERS**              |                    |                    |                           |                             |
| **XGBoost Meta n_estimators**               | 200                | 200                | 200                       | 200                         |
| **XGBoost Meta max_depth**                  | 4                  | 4                  | 4                         | 4                           |
| **XGBoost Meta learning_rate**              | 0.05               | 0.05               | 0.05                      | 0.05                        |
| **LightGBM Meta n_estimators**              | 200                | 200                | 200                       | 200                         |
| **LightGBM Meta max_depth**                 | 4                  | 4                  | 4                         | 4                           |
| **LightGBM Meta learning_rate**             | 0.05               | 0.05               | 0.05                      | 0.05                        |
| **Neural Network Meta layers**              | (64, 32)           | (64, 32)           | (64, 32)                  | (64, 32)                    |
| **Neural Network Meta max_iter**            | 300                | 300                | 300                       | 300                         |
| **🎯 METHOD PREDICTION PARAMETERS**         |                    |                    |                           |                             |
| **XGBoost Method n_estimators**             | 600                | 600                | 800                       | 500                         |
| **XGBoost Method max_depth**                | 8                  | 8                  | 10                        | 8                           |
| **XGBoost Method learning_rate**            | 0.025              | 0.02               | 0.015                     | 0.025                       |
| **LightGBM Method n_estimators**            | 600                | 600                | 800                       | 500                         |
| **LightGBM Method max_depth**               | 8                  | 8                  | 10                        | 8                           |
| **LightGBM Method learning_rate**           | 0.025              | 0.02               | 0.015                     | 0.025                       |
| **Random Forest Method n_estimators**       | 600                | 600                | 800                       | 500                         |
| **Random Forest Method max_depth**          | 20                 | 20                 | 25                        | 15                          |
| **Neural Network Method layers**            | (128, 64, 32)      | (128, 64, 32)      | (128, 64, 32)             | (128, 64)                   |
| **Neural Network Method max_iter**          | 400                | 400                | 400                       | 300                         |

## **DETAILED PARAMETER ANALYSIS**

### **PERFORMANCE ENHANCEMENT**

| **Model**             | **Key Improvements**                                  | **Performance Impact**                   |
| --------------------------- | ----------------------------------------------------------- | ---------------------------------------------- |
| **exe Base**          | Enhanced parameters                                         | High accuracy, no bias correction              |
| **Two Runs**          | Enhanced parameters, bias correction                        | High accuracy, better generalization           |
| **Class Weighting**   | Enhanced parameters, class weighting                        | High accuracy with bias correction             |
| **Data Augmentation** | **OPTIMIZED + BUFFED parameters, smart augmentation** | **High accuracy, 8-12x faster training** |

### **REGULARIZATION COMPARISON**

| **Model**             | **XGBoost reg_alpha** | **XGBoost reg_lambda** | **LightGBM reg_alpha** | **LightGBM reg_lambda** | **CatBoost l2_leaf_reg** |
| --------------------------- | --------------------------- | ---------------------------- | ---------------------------- | ----------------------------- | ------------------------------ |
| **exe Base**          | 0.2                         | 1                            | 0.2                          | 1                             | 1                              |
| **Two Runs**          | 0.2                         | 1                            | 0.2                          | 1                             | 1                              |
| **Class Weighting**   | 0.1                         | 0.8                          | 0.1                          | 0.8                           | 0.5                            |
| **Data Augmentation** | 0.2                         | 1                            | 0.2                          | 1                             | 1                              |

### **FEATURE SELECTION EVOLUTION**

| **Model**             | **Main Models** | **Random Forest** | **Method Models** | **Cross-Validation** |
| --------------------------- | --------------------- | ----------------------- | ----------------------- | -------------------------- |
| **exe Base**          | 75%                   | 75%                     | 75%                     | 5 folds                    |
| **Two Runs**          | 75%                   | 75%                     | 75%                     | 5 folds                    |
| **Class Weighting**   | 85%                   | 85%                     | 85%                     | 7 folds                    |
| **Data Augmentation** | 60%                   | 60%                     | 60%                     | 5 folds                    |
