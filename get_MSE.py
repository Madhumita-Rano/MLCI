from sklearn.metrics import mean_squared_error,explained_variance_score,mean_squared_log_error
import numpy as np

def get_MSE():
    fmse = open("file","a")

    train_act = np.loadtxt("train.out",usecols =3)
    train_pred = np.loadtxt("train.out",usecols = 4)
    train_MSE = mean_squared_error(train_act,train_pred)

    valid_act = np.loadtxt("validation.out",usecols = 3)
    valid_pred = np.loadtxt("validation.out",usecols = 4)
    valid_MSE = mean_squared_error(valid_act,valid_pred)

    fmse.write(f"Train MSE error >> {train_MSE}\tValidation MSE error >> {valid_MSE}\n")
    fmse.close()
