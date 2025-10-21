# UFC Predictor Models - Parameter Comparison

## **MODEL PARAMETERS COMPARISON TABLE**

| **Parameter**                            | **exe Base** | **Two Runs** | **Data Augmentation**   |
| ---------------------------------------------- | ------------------ | ------------------ | ----------------------------- |
| **XGBOOST PARAMETERS**                   |                    |                    |                               |
| n_estimators                                   | 600                | 600                | 600                           |
| max_depth                                      | 8                  | 9                  | 9                             |
| learning_rate                                  | 0.025              | 0.02               | 0.02                          |
| subsample                                      | 0.85               | 0.85               | 0.85                          |
| colsample_bytree                               | 0.8                | 0.85               | 0.85                          |
| colsample_bylevel                              | 0.8                | 0.85               | 0.85                          |
| reg_alpha                                      | 0.2                | 0.15               | 0.15                          |
| reg_lambda                                     | 1.0                | 0.8                | 0.8                           |
| min_child_weight                               | 4                  | 3                  | 3                             |
| gamma                                          | -                  | -                  | -                             |
| early_stopping_rounds                          | -                  | -                  | -                             |
| **LIGHTGBM PARAMETERS**                  |                    |                    |                               |
| n_estimators                                   | 600                | 600                | 600 (400 in performance mode) |
| max_depth                                      | 8                  | 9                  | 9                             |
| learning_rate                                  | 0.025              | 0.02               | 0.02                          |
| num_leaves                                     | 60                 | 60                 | 60                            |
| subsample                                      | 0.85               | 0.85               | 0.85                          |
| colsample_bytree                               | 0.8                | 0.85               | 0.85                          |
| colsample_bylevel                              | -                  | -                  | -                             |
| reg_alpha                                      | 0.2                | 0.15               | 0.15                          |
| reg_lambda                                     | 1.0                | 0.8                | 0.8                           |
| min_child_weight                               | 4                  | 3                  | 3                             |
| early_stopping_rounds                          | -                  | -                  | -                             |
| **CATBOOST PARAMETERS**                  |                    |                    |                               |
| iterations                                     | 600                | 600                | 600 (400 in performance mode) |
| depth                                          | 8                  | 9                  | 9                             |
| learning_rate                                  | 0.025              | 0.02               | 0.02                          |
| l2_leaf_reg                                    | 1.0                | 0.8                | 0.8                           |
| early_stopping_rounds                          | -                  | -                  | -                             |
| **RANDOM FOREST PARAMETERS**             |                    |                    |                               |
| n_estimators                                   | 600                | 600                | 600                           |
| max_depth                                      | 20                 | 18                 | 15                            |
| min_samples_split                              | 8                  | 5                  | 6                             |
| min_samples_leaf                               | 3                  | 2                  | 2                             |
| **NEURAL NETWORK PARAMETERS**            |                    |                    |                               |
| hidden_layers                                  | (256, 128, 64)     | (256, 128, 64)     | (256, 128, 64)                |
| learning_rate                                  | adaptive           | adaptive           | adaptive                      |
| max_iter                                       | 500                | 400                | 400                           |
| batch_size                                     | 32                 | 32                 | 32                            |
| alpha                                          | 0.001              | 0.0005             | 0.0005                        |
| early_stopping                                 | True               | True               | True                          |
| **CROSS-VALIDATION & FEATURE SELECTION** |                    |                    |                               |
| TimeSeriesSplit folds                          | 5                  | 5                  | 5                             |
| Feature Selection (Main)                       | 75%                | 75%                | 65%                           |
| Feature Selection (RF)                         | 75%                | 75%                | 65%                           |
| Feature Selection (Method)                     | 75%                | 75%                | 65%                           |
| Stacking CV folds                              | 5                  | 5                  | 5                             |
| Calibration CV folds                           | 5                  | 3                  | 3                             |
| Method Calibration CV folds                    | 5                  | 3                  | 3                             |
| **META-LEARNER PARAMETERS**              |                    |                    |                               |
| XGBoost Meta n_estimators                      | 200                | 200                | 200                           |
| XGBoost Meta max_depth                         | 4                  | 4                  | 4                             |
| XGBoost Meta learning_rate                     | 0.05               | 0.05               | 0.05                          |
| LightGBM Meta n_estimators                     | 200                | 200                | 200                           |
| LightGBM Meta max_depth                        | 4                  | 4                  | 4                             |
| LightGBM Meta learning_rate                    | 0.05               | 0.05               | 0.05                          |
| Neural Network Meta layers                     | (64, 32)           | (64, 32)           | (64, 32)                      |
| Neural Network Meta max_iter                   | 300                | 300                | 300                           |
| **METHOD PREDICTION PARAMETERS**         |                    |                    |                               |
| XGBoost Method n_estimators                    | 600                | 600                | 600                           |
| XGBoost Method max_depth                       | 8                  | 9                  | 9                             |
| XGBoost Method learning_rate                   | 0.025              | 0.02               | 0.02                          |
| LightGBM Method n_estimators                   | 600                | 600                | 600 (400 in performance mode) |
| LightGBM Method max_depth                      | 8                  | 9                  | 9                             |
| LightGBM Method learning_rate                  | 0.025              | 0.02               | 0.02                          |
| Random Forest Method n_estimators              | 600                | 600                | 600                           |
| Random Forest Method max_depth                 | 20                 | 18                 | 18                            |
| Neural Network Method layers                   | (128, 64, 32)      | (128, 64, 32)      | (128, 64, 32)                 |
| Neural Network Method max_iter                 | 400                | 400                | 400                           |

## **PERFORMANCE COMPARISON**

### **ACCURACY CHARACTERISTICS**

| **Model**             | **Bias Correction** | **Method** | **Key Features**              |
| --------------------------- | ------------------------- | ---------------- | ----------------------------------- |
| **exe Base**          | None                      | Standard         | High accuracy, no bias correction   |
| **Two Runs**          | Two-run prediction        | Standard         | Bias correction via corner swapping |
| **Data Augmentation** | Smart augmentation        | Standard         | 1.75x dataset size, bias threshold  |

## **REGULARIZATION COMPARISON**

| **Model**             | **XGBoost reg_alpha** | **XGBoost reg_lambda** | **LightGBM reg_alpha** | **LightGBM reg_lambda** | **CatBoost l2_leaf_reg** |
| --------------------------- | --------------------------- | ---------------------------- | ---------------------------- | ----------------------------- | ------------------------------ |
| **exe Base**          | 0.2                         | 1.0                          | 0.2                          | 1.0                           | 1.0                            |
| **Two Runs**          | 0.15                        | 0.8                          | 0.15                         | 0.8                           | 0.8                            |
| **Data Augmentation** | 0.15                        | 0.8                          | 0.15                         | 0.8                           | 0.8                            |

## **FEATURE SELECTION EVOLUTION**

| **Model**             | **Main Models** | **Random Forest** | **Method Models** | **Cross-Validation** |
| --------------------------- | --------------------- | ----------------------- | ----------------------- | -------------------------- |
| **exe Base**          | 75%                   | 75%                     | 75%                     | 5 folds                    |
| **Two Runs**          | 75%                   | 75%                     | 75%                     | 5 folds                    |
| **Data Augmentation** | 65%                   | 65%                     | 65%                     | 5 folds                    |

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
- **Data Augmentation**: Smart augmentation with bias threshold (1.75x dataset) + Advanced statistical features + Enhanced method prediction

### **ADVANCED FEATURES (Data Augmentation Model)**

- **GPU Acceleration**: XGBoost + LightGBM on GPU for faster training
- **Smart Data Augmentation**: 1.75x dataset size with bias threshold analysis
- **Performance Mode**: Adaptive parameters (600 estimators/iterations for accuracy, 400 for speed)
- **Enhanced Statistical Features**: 200+ advanced features for better predictions
- **Dynamic Ensemble**: Adaptive ensemble weighting based on performance
- **Advanced Validation**: Multi-strategy validation approach
- **Hyperparameter Optimization**: Automated parameter tuning
- **Feature Caching**: 2x speedup on repeated runs
- **Momentum Consistency**: Performance consistency analysis
- **Pressure Resistance**: Ability to perform under pressure
- **Clutch Factor**: Performance in close fights
- **Adaptability Score**: Mid-fight adjustment ability
- **Finish Timing**: When fighters typically finish
- **Pace Control**: Fight tempo control ability
- **Defensive Soundness**: Overall defensive capability
- **Offensive Efficiency**: Strike accuracy and effectiveness
- **Fight IQ Differential**: Tactical intelligence comparison
- **Physical Advantage**: Size and strength factors
- **Technical Superiority**: Skill level comparison
- **Mental Toughness**: Adversity handling ability
- **Game Plan Execution**: Strategy adherence
- **Injury Resistance**: Durability and recovery
- **Weight Cut Impact**: Performance impact of weight cuts
- **Optimized Parameters**: Tuned for maximum accuracy with GPU acceleration
