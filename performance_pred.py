### EVALUATION OF PREDICTION TIME ###
import numpy as np
import matplotlib.pyplot as plt
from scratch import DNN, LoadingData
from time import time

data = LoadingData()
model_dnn = DNN()
model_dnn.build(data.X,data.Y)

X, Y = model_dnn.get_input_array(data.X), data.Y
model_dnn.load(path="model/scratch.h5")

Time_predicting = []

proportion_extracted = [1-0.05*i for i in range(20)]

for size in proportion_extracted:
    t0 = time()
    random_idx = np.random.choice(np.arange(len(X)), int(size*len(X)), replace=False)
    X_extract = X[random_idx]
    Y_pred = model_dnn.predict(X_extract)
    t1 = time()
    Time_predicting.append(t1-t0)

plt.plot([int(len(X)*p) for p in proportion_extracted],Time_predicting)
plt.xlabel("Input size")
plt.ylabel("Prediction time (second)")
plt.title("The prediction time as a function of the size of the input size")
plt.savefig('Figure/Prediction_time.png', dpi=144)
