import struct
import numpy as np
import sys
import pickle
import math
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def loadlocal_mnist(images_path, labels_path):
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

def my_model(features, weights):
    inner_sum = 0
    
    for ind, elem in enumerate(features):
        inner_sum += elem * weights[ind]
        
    return inner_sum

def my_model(features, weights):
    inner_sum = 0
    
    for ind, elem in enumerate(features):
        inner_sum += elem * weights[ind]
        
    return inner_sum

def predict(X, weights):
    predicted_values = []
    
    for record in X:
        result = my_model(record, weights)
        predicted_values.append(result)
        
    return np.array(predicted_values)

def init_weights(features_num, a, b):
    return a + (b - a) * np.random.random(features_num)

def my_loss(predicted_value, true_value):
    result = (predicted_value - true_value) ** 2        
    
    return result

def my_loss_fun(predicted_values, true_values):
    loss_sum = 0.0
    
    for ind, elem in enumerate(predicted_values):
        loss_sum += my_loss(elem, true_values[ind])
        
    return loss_sum / len(predicted_values)

def loss_in_point(X, y, weights):
    predicted_values = []
    
    for record in X:
        result = my_model(record, weights)
        predicted_values.append(result)
        
    return my_loss_fun(predicted_values, y)

def mse_grad_regul(X, y, model_weights, lambda_coef = 0.1):
    records_num, _ = X.shape
    weights_grad = 2.0 * lambda_coef * model_weights[:]
    weights_grad[0] = 0.0
    current_loss = loss_in_point(X, y, model_weights)
    y_pred = predict(X, model_weights)
    grad = -2.0 / records_num * np.matmul(y - y_pred, X) + weights_grad
    
    return grad, current_loss

def grad_descent_regul(X, y, initial_weights, learning_rate=0.1, iter_num=1000, lambda_coef = 0.1):
    opt_weights = initial_weights[:]
    
    for iteration in xrange(iter_num):
        grad, current_loss_val = mse_grad_regul(X, y, opt_weights, lambda_coef)
        opt_weights -= learning_rate * grad
        
    return opt_weights

def reading (Y,number):
    massiv=np.copy(Y)
    for ind,elem in enumerate(massiv):
        if elem ==number:
            True
        else:
            massiv[ind]=0
    return massiv

def reading_0 (Y,number):
    massiv=np.copy(Y)
    for ind,elem in enumerate(massiv):
        if elem ==number:
            True
        else:
            massiv[ind]=1
    return massiv

X, Y = loadlocal_mnist(
        images_path= sys.argv[1], 
        labels_path= sys.argv[2])

record_num, features_num = X.shape
bias_f = np.ones((record_num, 1))
X = np.hstack((bias_f, X))

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.25, random_state=1)

# For 0
y=reading_0(y_train,0)
regul_coef = 0.01
records_num, features_num = X_train.shape
initial_weights = init_weights(features_num, -0.0025, 0.0025)
opt_weights_0 = grad_descent_regul(X_train, y, initial_weights, learning_rate=0.0000001, iter_num=1, lambda_coef = regul_coef)
# For 1 
y=reading(y_train,1)
regul_coef = 0.01
records_num, features_num = X_train.shape
initial_weights = init_weights(features_num, -0.0025, 0.0025)
opt_weights_1 = grad_descent_regul(X_train, y, initial_weights, learning_rate=0.0000001, iter_num=200, lambda_coef = regul_coef)
# For 2
y=reading(y_train,2)
regul_coef = 0.01
records_num, features_num = X_train.shape
initial_weights = init_weights(features_num, -0.000005, 0.000005)
opt_weights_2 = grad_descent_regul(X_train, y, initial_weights, learning_rate=0.0000001, iter_num=200, lambda_coef = regul_coef)
# For 3
y=reading(y_train,3)
regul_coef = 0.01
records_num, features_num = X_train.shape
initial_weights = init_weights(features_num, -0.000005, 0.000005)
opt_weights_3 = grad_descent_regul(X_train, y, initial_weights, learning_rate=0.0000001, iter_num=200, lambda_coef = regul_coef)
#For 4 
y=reading(y_train,4)
regul_coef = 0.01
records_num, features_num = X_train.shape
initial_weights = init_weights(features_num, -0.000005, 0.000005)
opt_weights_4 = grad_descent_regul(X_train, y, initial_weights, learning_rate=0.0000001, iter_num=200, lambda_coef = regul_coef)
#For 5
y=reading(y_train,5)
regul_coef = 0.01
records_num, features_num = X_train.shape
initial_weights = init_weights(features_num, -0.000005, 0.000005)
opt_weights_5 = grad_descent_regul(X_train, y, initial_weights, learning_rate=0.0000001, iter_num=200, lambda_coef = regul_coef)
#For 6
y=reading(y_train,6)
regul_coef = 0.01
records_num, features_num = X_train.shape
initial_weights = init_weights(features_num, -0.000005, 0.000005)
opt_weights_6 = grad_descent_regul(X_train, y, initial_weights, learning_rate=0.0000001, iter_num=200, lambda_coef = regul_coef)
#For 7
y=reading(y_train,7)
regul_coef = 0.01
records_num, features_num = X_train.shape
initial_weights = init_weights(features_num, -0.000005, 0.000005)
opt_weights_7 = grad_descent_regul(X_train, y, initial_weights, learning_rate=0.0000001, iter_num=200, lambda_coef = regul_coef)
#For 8 
y=reading(y_train,8)
regul_coef = 0.01
records_num, features_num = X_train.shape
initial_weights = init_weights(features_num, -0.000005, 0.000005)
opt_weights_8 = grad_descent_regul(X_train, y, initial_weights, learning_rate=0.0000001, iter_num=200, lambda_coef = regul_coef)
#For 9 
y=reading(y_train,9)
regul_coef = 0.01
records_num, features_num = X_train.shape
initial_weights = init_weights(features_num, -0.000005, 0.000005)
opt_weights_9 = grad_descent_regul(X_train, y, initial_weights, learning_rate=0.0000001, iter_num=200, lambda_coef = regul_coef)
#Result
optimal_weights_final = [opt_weights_0,opt_weights_1,opt_weights_2,opt_weights_3,opt_weights_4,opt_weights_5,opt_weights_6,opt_weights_7,opt_weights_8,opt_weights_9]
readl_digit=[0,1,2,3,4,5,6,7,8,9]

predicted_labels_final = []
for ind, record in enumerate(X):
    current_max=[]
    for val,vesa in enumerate(optimal_weights_final):
        predicted_score = my_model(record, vesa)
        current_max.append(predicted_score)
    predicted_labels_final.append(readl_digit[current_max.index(max(current_max))])

    
np.savetxt('final_model.txt', (optimal_weights_final )) 
cl = classification_report(Y, predicted_labels_final)
print cl 