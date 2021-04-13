import numpy as np

class Model():
    #attributes:
    learning_rate: float
    epoch: int
    gamma: float
    lim: float
    w: np.ndarray
    u: float    #first component of the weight vector

    #Methods:
    def __init__(self, learning_rate=0.0002, epoch=5, gamma=0.95, lim=0.5):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.gamma = gamma
        self.lim = lim

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
            X = np.reshape(X, (X.shape[0], -1))
            n = X.shape[0]
            d = X.shape[1]

            #creating a weight vector
            self.w = np.ndarray(shape=(d, 1), dtype=float)
            self.w.fill(0)

            #creating a moment vector
            moment = self.w

            #creating a first componenets of weight and moment vector
            moment_first_component = 0
            self.u = 0

            #Model trainnig
            for _ in range(self.epoch):
                for i in range(n):
                    #arguments needed to calculate the gradient
                    t = -X[i]@self.w - self.u
                    l = 1 / (1 + np.exp(t))

                    #gradient calculation
                    g = (l - y[i]) * (l**2) * np.exp(t)
                    grad = g * X[i].transpose()
                    grad = np.reshape(grad, self.w.shape)

                    #weight update
                    moment = self.gamma * moment + self.learning_rate * grad
                    moment_first_component = self.gamma * moment_first_component + self.learning_rate *g
                    self.u = self.u - moment_first_component
                    self.w = self.w - moment

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.reshape(X, (X.shape[0], -1))
        y = np.ndarray(shape=(X.shape[0],), dtype=int)
        for i in range(X.shape[0]):
            p = 1 / (1 + np.exp(-X[i]@self.w - self.u))
            if p > self.lim:
                y[i] = 1
            else:
                y[i] = 0
        return y

    @staticmethod
    def evaluate(y_true: np.ndarray, y_predict: np.ndarray) ->float:
        errors = 0
        for i in range(y_predict.shape[0]):
            if y_predict[i] != y_true[i]:
                errors += 1
        return 1-(errors/y_predict.shape[0])
