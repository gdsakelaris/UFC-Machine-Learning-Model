# UFC Predictor Models - Parameter Comparison

## **MODEL PARAMETERS COMPARISON TABLE**

| **Parameter**                            | **exe Base** | **Two Runs** | **Class Weighting** | **Data Augmentation** |
| ---------------------------------------------- | ------------------ | ------------------ | ------------------------- | --------------------------- |
| **XGBOOST PARAMETERS**                   |                    |                    |                           |                             |
| **n_estimators**                         | 600                | 500                | 800                       | 500                         |
| **max_depth**                            | 8                  | 8                  | 10                        | 8                           |
| **learning_rate**                        | 0.025              | 0.025              | 0.015                     | 0.025                       |
| **subsample**                            | 0.85               | 0.85               | 0.85                      | 0.85                        |
| **colsample_bytree**                     | 0.8                | 0.85               | 0.85                      | 0.85                        |
| **colsample_bylevel**                    | 0.8                | 0.85               | 0.85                      | 0.85                        |
| **reg_alpha**                            | 0.2                | 0.2                | 0.1                       | 0.2                         |
| **reg_lambda**                           | 1                  | 1                  | 0.8                       | 1                           |
| **min_child_weight**                     | 4                  | 4                  | 3                         | 4                           |
| **gamma**                                | 0.15               | 0.15               | 0.15                      | 0.15                        |
| **early_stopping_rounds**                | -                  | -                  | -                         | -                           |
| **LIGHTGBM PARAMETERS**                  |                    |                    |                           |                             |
| **n_estimators**                         | 600                | 500                | 800                       | 500                         |
| **max_depth**                            | 8                  | 8                  | 10                        | 8                           |
| **learning_rate**                        | 0.025              | 0.025              | 0.015                     | 0.025                       |
| **num_leaves**                           | 60                 | 50                 | 80                        | 50                          |
| **subsample**                            | 0.85               | 0.85               | 0.85                      | 0.85                        |
| **colsample_bytree**                     | 0.8                | 0.85               | 0.85                      | 0.85                        |
| **colsample_bylevel**                    | 0.8                | 0.85               | 0.85                      | 0.85                        |
| **reg_alpha**                            | 0.2                | 0.2                | 0.1                       | 0.2                         |
| **reg_lambda**                           | 1                  | 1                  | 0.8                       | 1                           |
| **min_child_weight**                     | 4                  | 4                  | 3                         | 4                           |
| **early_stopping_rounds**                | -                  | -                  | -                         | -                           |
| **CATBOOST PARAMETERS**                  |                    |                    |                           |                             |
| **iterations**                           | 600                | 500                | 800                       | 500                         |
| **depth**                                | 8                  | 8                  | 10                        | 8                           |
| **learning_rate**                        | 0.025              | 0.025              | 0.015                     | 0.025                       |
| **l2_leaf_reg**                          | 1                  | 1                  | 0.5                       | 1                           |
| **early_stopping_rounds**                | -                  | -                  | -                         | -                           |
| **RANDOM FOREST PARAMETERS**             |                    |                    |                           |                             |
| **n_estimators**                         | 600                | 500                | 800                       | 500                         |
| **max_depth**                            | 20                 | 15                 | 25                        | 15                          |
| **min_samples_split**                    | 8                  | 6                  | 6                         | 6                           |
| **min_samples_leaf**                     | 2                  | 2                  | 2                         | 2                           |
| **NEURAL NETWORK PARAMETERS**            |                    |                    |                           |                             |
| **hidden_layers**                        | (256, 128, 64)     | (256, 128)         | (256, 128, 64)            | (256, 128)                  |
| **learning_rate**                        | adaptive           | adaptive           | adaptive                  | adaptive                    |
| **max_iter**                             | 500                | 300                | 500                       | 300                         |
| **batch_size**                           | 32                 | 32                 | 32                        | 32                          |
| **alpha**                                | 0.001              | 0.001              | 0.001                     | 0.001                       |
| **early_stopping**                       | True               | True               | True                      | True                        |
| **CROSS-VALIDATION & FEATURE SELECTION** |                    |                    |                           |                             |
| **TimeSeriesSplit folds**                | 5                  | 5                  | 7                         | 5                           |
| **Feature Selection (Main)**             | 75%                | 75%                | 85%                       | 60%                         |
| **Feature Selection (RF)**               | 75%                | 75%                | 85%                       | 75%                         |
| **Feature Selection (Method)**           | 75%                | 75%                | 85%                       | 60%                         |
| **Stacking CV folds**                    | 5                  | 5                  | 5                         | 5                           |
| **Calibration CV folds**                 | 3                  | 3                  | 3                         | 3                           |
| **META-LEARNER PARAMETERS**              |                    |                    |                           |                             |
| **XGBoost Meta n_estimators**            | 200                | 200                | 200                       | 200                         |
| **XGBoost Meta max_depth**               | 4                  | 4                  | 4                         | 4                           |
| **XGBoost Meta learning_rate**           | 0.05               | 0.05               | 0.05                      | 0.05                        |
| **LightGBM Meta n_estimators**           | 200                | 200                | 200                       | 200                         |
| **LightGBM Meta max_depth**              | 4                  | 4                  | 4                         | 4                           |
| **LightGBM Meta learning_rate**          | 0.05               | 0.05               | 0.05                      | 0.05                        |
| **Neural Network Meta layers**           | (64, 32)           | (64, 32)           | (64, 32)                  | (64, 32)                    |
| **Neural Network Meta max_iter**         | 300                | 300                | 300                       | 300                         |
| **METHOD PREDICTION PARAMETERS**         |                    |                    |                           |                             |
| **XGBoost Method n_estimators**          | 600                | 500                | 800                       | 500                         |
| **XGBoost Method max_depth**             | 8                  | 8                  | 10                        | 8                           |
| **XGBoost Method learning_rate**         | 0.025              | 0.025              | 0.015                     | 0.025                       |
| **LightGBM Method n_estimators**         | 600                | 500                | 800                       | 500                         |
| **LightGBM Method max_depth**            | 8                  | 8                  | 10                        | 8                           |
| **LightGBM Method learning_rate**        | 0.025              | 0.025              | 0.015                     | 0.025                       |
| **Random Forest Method n_estimators**    | 600                | 500                | 800                       | 500                         |
| **Random Forest Method max_depth**       | 20                 | 15                 | 25                        | 15                          |
| **Neural Network Method layers**         | (128, 64, 32)      | (128, 64)          | (128, 64, 32)             | (128, 64)                   |
| **Neural Network Method max_iter**       | 400                | 300                | 400                       | 300                         |

## **PERFORMANCE OPTIMIZATIONS**

| **Model**             | **12-Core Processing** | **Feature Caching** | **Parallel Backend** |
| --------------------------- | ---------------------------- | ------------------------- | -------------------------- |
| **exe Base**          | ✅ Yes                       | ✅ Yes                    | loky                       |
| **Two Runs**          | ✅ Yes                       | ✅ Yes                    | loky                       |
| **Class Weighting**   | ✅ Yes                       | ✅ Yes                    | loky                       |
| **Data Augmentation** | ✅ Yes                       | ✅ Yes                    | loky                       |

## **TECHNICAL IMPLEMENTATIONS**

### **MULTIPROCESSING CONFIGURATION**

- **Environment Variables**: `JOBLIB_MULTIPROCESSING=1`, `LOKY_MAX_WORKERS=12`
- **Parallel Processing**: `min(12, mp.cpu_count())` cores
- **Backend**: `loky` (more efficient than threading)
- **Cross-Validation**: Parallel fold processing

### **FEATURE CACHING SYSTEM**

- **Cache Key**: `f"{len(df)}_{hash(str(df.columns.tolist()))}"`
- **Cache Storage**: `(df, feature_columns)` tuples
- **Speed Gain**: ~2x faster on subsequent runs
- **Memory Management**: Automatic cache cleanup

### **AUTOMATIC CLEANUP**

- **Temporary Files**: `catboost_info/` folder, `best_dl_model.h5`
- **Cleanup Triggers**: GUI completion, program exit, atexit handler
- **Implementation**: `shutil.rmtree()` and `os.remove()`

## **PERFORMANCE COMPARISON**

### **ACCURACY CHARACTERISTICS**

| **Model**             | **Bias Correction** | **Method** | **Key Features**              |
| --------------------------- | ------------------------- | ---------------- | ----------------------------------- |
| **exe Base**          | None                      | Standard         | High accuracy, no bias correction   |
| **Two Runs**          | Two-run prediction        | Standard         | Bias correction via corner swapping |
| **Class Weighting**   | Class weight balancing    | Standard         | Handles class imbalance             |
| **Data Augmentation** | Smart augmentation        | Standard         | 1.5x dataset size, bias threshold   |

## **REGULARIZATION COMPARISON**

| **Model**             | **XGBoost reg_alpha** | **XGBoost reg_lambda** | **LightGBM reg_alpha** | **LightGBM reg_lambda** | **CatBoost l2_leaf_reg** |
| --------------------------- | --------------------------- | ---------------------------- | ---------------------------- | ----------------------------- | ------------------------------ |
| **exe Base**          | 0.2                         | 1                            | 0.2                          | 1                             | 1                              |
| **Two Runs**          | 0.2                         | 1                            | 0.2                          | 1                             | 1                              |
| **Class Weighting**   | 0.1                         | 0.8                          | 0.1                          | 0.8                           | 0.5                            |
| **Data Augmentation** | 0.2                         | 1                            | 0.2                          | 1                             | 1                              |

## **FEATURE SELECTION EVOLUTION**

| **Model**             | **Main Models** | **Random Forest** | **Method Models** | **Cross-Validation** |
| --------------------------- | --------------------- | ----------------------- | ----------------------- | -------------------------- |
| **exe Base**          | 75%                   | 75%                     | 75%                     | 5 folds                    |
| **Two Runs**          | 75%                   | 75%                     | 75%                     | 5 folds                    |
| **Class Weighting**   | 85%                   | 85%                     | 85%                     | 7 folds                    |
| **Data Augmentation** | 60%                   | 75%                     | 60%                     | 5 folds                    |

## **OPTIMIZATION SUMMARY**

### **UNIVERSAL IMPROVEMENTS (All Models)**

- ✅ 12-core parallel processing for cross-validation
- ✅ Feature caching for 2x speedup on repeated runs
- ✅ Loky backend for efficient process management
- ✅ Automatic cleanup of temporary files
- ✅ Consistent parameter optimization

### **MODEL-SPECIFIC FEATURES**

- **exe Base**: Standard ensemble
- **Two Runs**: Bias correction via corner swapping
- **Class Weighting**: Handles class imbalance with enhanced parameters
- **Data Augmentation**: Smart augmentation with bias threshold (1.5x dataset)
