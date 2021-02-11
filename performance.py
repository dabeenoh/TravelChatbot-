# %%
import numpy as np
import matplotlib.pyplot as plt
from scratch import DNN, LoadingData
from time import time
# %%
data = LoadingData()
model_dnn = DNN()
model_dnn.build(data.X_train,data.Y_train)

X_train = model_dnn.get_input_array(data.X_train)
Y_train = data.Y_train
X_test_, Y_test_ = model_dnn.get_input_array(data.X_test),data.Y_test # text data (test)

# %%
Time_modelbuilding = []
Time_testing = []
Loss, Accuracy = [], []
proportion_extracted = [1,0.95,0.90,0.85,0.80,0.75,0.70,0.60,0.50,0.40,0.30]
for size in proportion_extracted:
    t0 = time()
    model_dnn = DNN()
    model_dnn.build(data.X_train,data.Y_train)
    random_idx = np.random.choice(np.arange(len(X_train)), int(size*len(X_train)), replace=False)
    X_extract = X_train[random_idx]
    Y_extract = Y_train[random_idx]
    model_dnn.train(X_extract,Y_extract,epochs=9000,verbose=True)
    t1 = time()
    Time_modelbuilding.append(t1-t0)

    Y_pred = model_dnn.predict(X_test_)
    loss, accuracy = model_dnn.evaluate(Y_pred,Y_test_)
    Loss.append(loss)
    Accuracy.append(accuracy)

    Time_testing.append(time()-t1)

# %%
plt.plot(proportion_extracted,Time_testing)
plt.xlabel("Proportion of initial training set (%s)"%len(X_train))
plt.ylabel("Testing time (second)")
plt.title("The testing time does not depend on the input size")

# %%
plt.plot(proportion_extracted,Time_modelbuilding)
plt.xlabel("Proportion of initial training set (%s)"%len(X_train))
plt.ylabel("Training time (second)")
plt.title("The training time as a function of the size of the training set")

# %%
plt.plot(proportion_extracted,Accuracy)
plt.xlabel("Proportion of initial training set (%s)"%len(X_train))
plt.ylabel("Accuracy of the model")
plt.title("The accuracy on test set as a function of the size of the training set")

# %%
plt.plot(proportion_extracted,Loss)
plt.xlabel("Proportion of initial training set (%s)"%len(X_train))
plt.ylabel("Loss of the model")
plt.title("The loss on test set as a function of the size of the training set")

# %%
