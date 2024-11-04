import joblib, json
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# load model and make prediction
best_model = joblib.load("models/rf_model.pkl")
y_pred = best_model.predict(X_test)

# save new dataset with prediction
data = pd.DataFrame({"pred":y_pred})
data.to_csv('data/pred/predictions.csv', index=False)

# save metrics
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = {
    'rmse': rmse,
    'r2': r2
}
with open('metrics/scores.json', 'w') as f:
    json.dump(metrics, f)

print("successful predictions !!!")