import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures


m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.6 * X ** 2 + X + 2 + np.random.randn(m, 1)

reg = linear_model.LinearRegression()
reg.fit(X, y)

X_plot = np.linspace(-4, 6, 100)
y_plot = reg.predict(X_plot.reshape(-1, 1))
plt.scatter(X, y, label="Дані")
plt.plot(X_plot, y_plot, label="Прогноз", color="red")
plt.legend()
plt.show()


poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
reg = linear_model.LinearRegression()
reg.fit(X_poly, y)

print("Features X[0]:", X[0])
print("Features after transformation:", X_poly[0])

print("Regression coefficient =", reg.coef_)
print("Regression interception =", reg.intercept_)

X_plot = np.linspace(-4, 6, 100)
y_plot = reg.predict(poly_features.transform(X_plot.reshape(-1, 1)))
plt.scatter(X, y, label="Дані")
plt.plot(X_plot, y_plot, label="Прогноз", color="red")
plt.legend()
plt.show()
