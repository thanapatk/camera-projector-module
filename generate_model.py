import pandas as pd
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


data = pd.read_csv("transformation_calibration.csv")

X = data[["scene_depth", "object_depth"]]
y_tx = data["tx"]
y_ty = data["ty"]
y_bias = data["bias"]

poly_transformer = PolynomialFeatures(degree=7)

X_poly = poly_transformer.fit_transform(X)

model_tx = LinearRegression()
model_ty = LinearRegression()
model_bias = LinearRegression()

model_tx.fit(X_poly, y_tx)
model_ty.fit(X_poly, y_ty)
model_bias.fit(X_poly, y_bias)

with open("models.pkl", "wb") as f:
    pickle.dump((model_tx, model_ty, model_bias, poly_transformer), f)
