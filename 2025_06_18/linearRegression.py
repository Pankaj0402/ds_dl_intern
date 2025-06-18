import numpy as np
class LinearRegression:
    def __init__(self):
        self.b0 = 0
        self.b1 = 0
    def fit(self, X, y):
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        ssxy , ssx = 0, 0
        for _ in range(len(X)):
            ssxy += (X[_] - X_mean) * (y[_] - y_mean)
            ssx += (X[_] - X_mean) ** 2
        self.b1 = ssxy / ssx
        self.b0 = y_mean - (self.b1 * X_mean)
        return self.b0, self.b1
    def predict(self, X):
        y_hat = self.b0 + self.b1 * X
        return y_hat
    
if __name__ == "__main__":
    x = np.array([[160], [171], [182], [180], [154]])
    y = np.array([72, 76, 77, 83, 76])
    
   # print(f'the shape of height is {height.shape}')
   # print(f'the shape of weight is {weight.shape}')
    model = LinearRegression()
    b0, b1 = model.fit(x, y)
    print(f'b0: {b0}, b1: {b1}')
    
    x_test = np.array([[176]])
    y_hat = model.predict(x_test)
    print(f'Predicted weights for heights {x_test.flatten()}: {y_hat}')

