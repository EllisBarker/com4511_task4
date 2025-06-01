Python version used: Python 3.12.2
Libraries needed: argparse, matplotlib, numpy, scikit-learn

The 'task4.py' program handles 2 different experiment types:
1. Plotting the loss and DET curves when using the neural network on the 3 datasets (training, validation, testing).
2. Obtaining an average EER value and early stopping rate when regenerating the model multiple times for a set of predefined parameters.

The following arguments are required for the program to run:
- ```--experiment```: Provide 'experiment1' for experiment type 1, provide 'experiment2' for experiment type 2.
- ```--activation_function```: Provide 'sigmoid' for usage of the Sigmoid activation function, and 'relu' for usage of the Rectified Linear Unit activation function.
- ```--learning_rate```: Provide one of these values (0.1, 0.05, 0.01, 0.005, 0.001) to be used for backpropagation learning.

It is expected that this code is to be executed from within the task4/code directory in order to ensure the relative filepaths for the audio and label data is accessible.

An example of how to run this code is as follows:
```python task4.py --experiment experiment2 --activation_function sigmoid --learning_rate 0.01```