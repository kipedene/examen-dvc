import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

# model
model = RandomForestRegressor()

# GridSearchCv
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=3,
                           verbose=2, n_jobs=-1)

grid_search.fit(X_train, y_train)

# fit and save best params + model
best_params = grid_search.best_params_
joblib.dump(best_params, 'models/best_params.pkl')

best_model = RandomForestRegressor(**best_params)
best_model.fit(X_train, y_train)
joblib.dump(best_model, 'models/rf_model.pkl')

print("Successful training !!!")