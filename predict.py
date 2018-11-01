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

X, Y = loadlocal_mnist(
        images_path= sys.argv[1], 
        labels_path= sys.argv[2])

w_all=np.loadtxt(sys.argv[3])
vesa0,vesa1,vesa2,vesa3,vesa4,vesa5,vesa6,vesa7,vesa8,vesa9 = w_all

optimal_weights_final = [vesa0,vesa1,vesa2,vesa3,vesa4,vesa5,vesa6,vesa7,vesa8,vesa9]
readl_digit=[0,1,2,3,4,5,6,7,8,9]

predicted_labels_final = []
for ind, record in enumerate(X):
    current_max=[]
    for val,vesa in enumerate(optimal_weights_final):
        predicted_score = my_model(record, vesa)
        current_max.append(predicted_score)
    predicted_labels_final.append(readl_digit[current_max.index(max(current_max))])

cl = classification_report(Y, predicted_labels_final)
print cl