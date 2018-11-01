# mrg_mlcourse_module1
Train
python train.py -x_train_dir="data/train-images-idx3-ubyte" -y_train_dir="data/train-labels-idx1-ubyte" -model_output_dir="model_binary/model.pkl"

Predict
python predict.py -x_test_dir="data/t10k-images-idx3-ubyte" -y_test_dir="data/t10k-labels-idx1-ubyte" -model_input_dir="model_binary/model.pkl"
