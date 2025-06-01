from sklearn.metrics import roc_curve
import argparse
import matplotlib.pyplot as plt
import numpy as np

class VoiceActivityDetectionSystem:
    def __init__(self, 
                 activation_function: str, 
                 early_stop_threshold: int,
                 learning_rate: float):
        """
        Create a voice activity detection system given a specific activation function.
        
        Args:
            activation_function (str): The type of activation function to use in the network.
            early_stop_threshold (int): The number of epochs of increasing loss needed to halt training.
            learning_rate (float): The value used in updating of weights and biases in back propagation.
        """
        self.activation_function = activation_function
        # Pick the loss function based on the activation function
        loss_functions = {"sigmoid" : self.loss_cross_entropy,
                          "relu" : self.loss_mean_squared_error}
        self.loss_function = loss_functions.get(self.activation_function)
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_count = 0
        self.early_stop_index = 0
        self.learning_rate = learning_rate
        self.neighbouring_features = 2
        self.input_size = None
        self.layer_size = 64
        self.output_size = 1
        self.num_epochs = 50

    def load_file(self, filepath: str) -> np.ndarray:
        """
        Load and store a file given a specific filepath.

        Args:
            filepath (str): The path of the file to be loaded.
        
        Returns:
            numpy.ndarray: The contents of the file loaded.
        """
        with open(filepath, "rb") as f:
            return np.load(f)

    def prepare_data(self, training_f: str, validation_f: str, test_f: list[str]):
        """
        Load training/validation/testing audio data and apply preprocessing to relax independence.

        Args:
            training_f (str): The name of the training data (used to load audio/label information).
            validation_f (str): The name of the validation data (used to load audio/label information).
            test_f (str): The name of the test data (used to load audio/label information).
        """
        self.train_audio = self.stack_data_features(self.load_file("audio/" + str(training_f) + ".npy"))
        self.train_labels = self.adjust_data_labels(self.load_file("labels/" + str(training_f) + ".npy"))
        self.validation_audio = self.stack_data_features(self.load_file("audio/" + str(validation_f) + ".npy"))
        self.validation_labels = self.adjust_data_labels(self.load_file("labels/" + str(validation_f) + ".npy"))

        # Load both test data for interpretibility of results
        test_audio1 = self.stack_data_features(self.load_file("audio/" + str(test_f[0]) + ".npy"))
        test_labels1 = self.adjust_data_labels(self.load_file("labels/" + str(test_f[0]) + ".npy"))
        test_audio2 = self.stack_data_features(self.load_file("audio/" + str(test_f[1]) + ".npy"))
        test_labels2 = self.adjust_data_labels(self.load_file("labels/" + str(test_f[1]) + ".npy"))
        self.test_audio = np.concatenate((test_audio1, test_audio2), axis=0)
        self.test_labels = np.concatenate((test_labels1, test_labels2), axis=0)
        
        # Input size for neural network defined by the number of features for a frame and its neighbours.
        self.input_size = len(self.train_audio[0])

    def stack_data_features(self, data: np.ndarray) -> np.ndarray:
        """
        Stack neighbouring frames in audio data in order to relax independence assumptions
        and allow context from previous and future frames.

        Args:
            data (numpy.ndarray): The data to apply frame stacking to.

        Returns:
            numpy.ndarray: The data after having stacking applied to it.
        """
        num_samples, num_features = data.shape
        stacked_data = []
        for i in range(self.neighbouring_features, num_samples - self.neighbouring_features):
            frame = []
            for j in range(-self.neighbouring_features, self.neighbouring_features+1):
                for k in range(num_features):
                    frame.append(data[i + j, k])
            stacked_data.append(frame)
        return np.array(stacked_data)
    
    def adjust_data_labels(self, data: np.ndarray) -> np.ndarray: 
        """
        Change data labels to match those found after having applied feature stacking.

        Args:
            data (numpy.ndarray): The data to adjust the labels for.

        Returns:
            numpy.ndarray: The data after having had its labels aligned post-feature stacking.
        """
        return data[self.neighbouring_features : -self.neighbouring_features]

    def feed_forward(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Feed training data through a 2-layer neural network.

        Args:
            data (numpy.ndarray): The data with which to run through the network.

        Returns:
            numpy.ndarray: Results from the first activation function.
            numpy.ndarray: The pre-activation value from the hidden layer.
            numpy.ndarray: Predictions obtained for the data provided.
        """
        # Hidden layer
        linear1 = np.dot(data, self.weight1) + self.bias1
        if self.activation_function == "relu":
            activation1 = self.relu(linear1)
        elif self.activation_function == "sigmoid":
            activation1 = self.sigmoid(linear1)
        # Output layer
        linear2 = np.dot(activation1, self.weight2) + self.bias2
        if self.activation_function == "relu":
            prediction = self.relu(linear2)
        elif self.activation_function == "sigmoid":
            prediction = self.sigmoid(linear2)
        return activation1, linear1, prediction

    def back_propagate(self, 
                       data: np.ndarray,
                       labels: np.ndarray,
                       activation1: np.ndarray,
                       linear1: np.ndarray, 
                       prediction: np.ndarray, 
                       num_samples: int):
        """
        Update weights and biases used in neural network for next epoch (based on gradients).

        Args:
            data (numpy.ndarray): The data with which to update the weights/biases of the network.
            labels (numpy.ndarray): The true labels for the data provided.
            activation1 (numpy.ndarray): The results of the first activation function in feed_forward.
            linear1 (numpy.ndarray): The pre-activation value from the hidden layer.
            prediction (numpy.ndarray): The predictions made from the feed forward part of the network.
            num_samples (int): The number of samples within the given data.
        """
        output_error = prediction - labels.reshape(-1, 1)
        weight2_gradient = np.dot(activation1.T, output_error) / num_samples
        bias2_gradient = np.sum(output_error, axis=0, keepdims=True) / num_samples

        backward_signal = np.dot(output_error, self.weight2.T)
        if self.activation_function == "relu":
            hidden_error = backward_signal * self.relu_derivative(linear1)
        elif self.activation_function == "sigmoid":
            hidden_error = backward_signal * self.sigmoid_derivative(linear1)
        weight1_gradient = np.dot(data.T, hidden_error) / num_samples
        bias1_gradient = np.sum(hidden_error, axis=0, keepdims=True) / num_samples

        self.weight1 -= self.learning_rate * weight1_gradient
        self.bias1 -= self.learning_rate * bias1_gradient
        self.weight2 -= self.learning_rate * weight2_gradient
        self.bias2 -= self.learning_rate * bias2_gradient

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function (reducing values to a range from 0 to 1).

        Args:
            x (numpy.ndarray): The input to the sigmoid activation function.

        Returns:
            numpy.ndarray: The output of the sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, A: np.ndarray) -> np.ndarray:
        """
        Compute gradients for sigmoid outputs for use in back propagation.

        Args:
            A (numpy.ndarray): The output of the sigmoid activation function.

        Returns:
            numpy.ndarray: The gradients of the sigmoid output.
        """
        return self.sigmoid(A) * (1 - self.sigmoid(A))
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """
        ReLU (Rectified Linear Unit) activation function (changing values to range from 0 to infinity).

        Args:
            x (numpy.ndarray): The input to the ReLU activation function.

        Returns:
            numpy.ndarray: The output of the ReLU activation function.
        """
        return np.maximum(0, x)
    
    def relu_derivative(self, A: np.ndarray) -> np.ndarray:
        """
        Compute gradients for ReLU outputs for use in back propagation.

        Args:
            A (numpy.ndarray): The output of the ReLU activation function.

        Returns:
            numpy.ndarray: The gradients of the ReLU output.
        """
        return (A > 0).astype(float)
    
    def loss_cross_entropy(self, prediction: np.ndarray, num_samples: int, labels: np.ndarray) -> float:
        """
        Compute loss via cross entropy method to show the performance of the neural network change over time
        (for use with the Sigmoid activation function).

        Args:
            prediction (numpy.ndarray): The predictions made from the neural network.
            num_samples (int): The number of samples within the given data.
            labels (numpy.ndarray): The true labels of the given data.

        Returns:
            float: The loss value for the network based on the prediction.
        """
        # Prevent log(0) from occurring through use of 1e-12
        prediction = np.clip(prediction, 1e-12, 1 - 1e-12)
        loss = -(1 / num_samples) * np.sum(labels * np.log(prediction + 1e-12) + (1 - labels) * np.log(1 - prediction - 1e-12))
        return loss

    def loss_mean_squared_error(self, prediction: np.ndarray, num_samples: int, labels: np.ndarray) -> float:
        """
        Compute loss via mean squared error to show the performance of the neural network change over time
        (for use with the ReLU activation function).

        Args:
            prediction (numpy.ndarray): The predictions made from the neural network.
            num_samples (int): The number of samples within the given data.
            labels (numpy.ndarray): The true labels of the given data.

        Returns:
            float: The loss value for the network based on the prediction.
        """
        loss = (1 / num_samples) * np.sum(np.square(prediction - labels))
        return loss

    def set_model_attributes(self):
        """
        Set the weights and biases of the neural network to random values.
        """
        # Initialise weights and biases
        self.weight1 = np.random.randn(self.input_size, self.layer_size) * 0.01
        self.bias1 = np.zeros((1, self.layer_size))
        self.weight2 = np.random.randn(self.layer_size, 1) * 0.01
        self.bias2 = np.zeros((1, self.output_size))

    def train_model(self):
        """
        Finetune weights and biases through continuous prediction and backpropagation in order to improve
        neural network accuracy.
        """
        num_samples_train = self.train_labels.shape[0]
        num_samples_val = self.validation_labels.shape[0]
        num_samples_test = self.test_labels.shape[0]
        self.losses_train = []
        self.losses_val = []
        self.losses_test = []
        early_stop_no = 0

        for epoch in range(self.num_epochs):
            # Store loss for testing data
            _, _, prediction_test = self.feed_forward(self.test_audio)
            loss_test = self.loss_function(prediction_test, num_samples_test, self.test_labels)
            self.losses_test.append(loss_test)

            # Evaluate learning using validation data (allow early stopping if loss increases)
            _, _, prediction_val = self.feed_forward(self.validation_audio)
            loss_val = self.loss_function(prediction_val, num_samples_val, self.validation_labels)
            self.losses_val.append(loss_val)
            print (f"Current epoch: {epoch+1}, Validation loss: {loss_val}")
            if epoch != 0 and self.early_stop_threshold != None:
                if self.losses_val[epoch] > self.losses_val[epoch-1]:
                    early_stop_no += 1
                else:
                    early_stop_no = 0
                if early_stop_no >= self.early_stop_threshold:
                    self.early_stop_count += 1
                    self.early_stop_index += epoch+1
                    break

            # Update weights and biases using training data
            activation1, linear1, prediction_train = self.feed_forward(self.train_audio)
            loss_train = self.loss_function(prediction_train, num_samples_train, self.train_labels)
            self.losses_train.append(loss_train)
            self.back_propagate(self.train_audio, self.train_labels, activation1, linear1, prediction_train, num_samples_train)

    def evaluate_model(self, audio, labels) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Obtain predictions for test data based on the model previously trained.

        Args:
            audio (numpy.ndarray): The audio of the data with which to evaluate the model.
            labels (numpy.ndarray): The labels of the data with which to evaluate the model.

        Returns:
            numpy.ndarray: The false positive rate.
            numpy.ndarray: The false negative rate.
            int: The index of the EER point on the DET curve.
        """
        _, _, prediction = self.feed_forward(audio)
        fpr, tpr, _ = roc_curve(labels, prediction)
        fnr = 1 - tpr
        eer_index = np.nanargmin(np.absolute((fnr - fpr)))
        return fpr, fnr, eer_index

    def plot_det_curve(self, fpr: np.ndarray, fnr: np.ndarray, eer_index: int):
        """
        Visualise the DET curve after obtaining predictions for test data.

        Args:
            fpr (numpy.ndarray): The false positive rate.
            fnr (numpy.ndarray): The false negative rate.
            eer_index (int): The index of the EER point on the DET curve.
        """
        plt.figure(1)
        plt.plot(fpr, fnr, label="DET Curve")
        plt.plot(fpr[eer_index], fnr[eer_index], "go", label="EER Point")
        plt.title("Detection Error Tradeoff (DET) Curve")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("False Negative Rate (FNR)")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_det_curves(self, 
                       fpr1: np.ndarray, 
                       fpr2: np.ndarray,
                       fpr3: np.ndarray,
                       fnr1: np.ndarray,
                       fnr2: np.ndarray,
                       fnr3: np.ndarray, 
                       eer_index1: int,
                       eer_index2: int,
                       eer_index3: int):
        """
        Visualise the DET curves after obtaining predictions for training, validation and test data.

        Args:
            fpr1 (numpy.ndarray): The first false positive rate (training).
            fpr2 (numpy.ndarray): The second false positive rate (validation).
            fpr3 (numpy.ndarray): The second false positive rate (testing).
            fnr1 (numpy.ndarray): The first false negative rate (training).
            fnr2 (numpy.ndarray): The second false negative rate (validation).
            fnr3 (numpy.ndarray): The second false negative rate (testing).
            eer_index1 (int): The index of the EER point on the DET curve (training).
            eer_index2 (int): The index of the EER point on the DET curve (validation).
            eer_index3 (int): The index of the EER point on the DET curve (testing).
        """
        plt.figure(1)
        plt.plot(fpr1, fnr1, label="DET Curve (Training)")
        plt.plot(fpr1[eer_index1], fnr1[eer_index1], "go", label="EER Point (Training)")
        plt.plot(fpr2, fnr2, label="DET Curve (Validation)")
        plt.plot(fpr2[eer_index2], fnr2[eer_index2], "ro", label="EER Point (Validation)")
        plt.plot(fpr3, fnr3, label="DET Curve (Testing)")
        plt.plot(fpr3[eer_index3], fnr3[eer_index3], "bo", label="EER Point (Testing)")
        plt.title("Detection Error Tradeoff (DET) Curve")
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("False Negative Rate (FNR)")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_loss_curves(self, losses_train: np.ndarray, losses_val: np.ndarray, losses_test: np.ndarray):
        """
        Plot the loss values for the 3 datasets (for experiment 2).

        Args:
            losses_train (numpy.ndarray): The loss values for the training data.
            losses_val (numpy.ndarray): The loss values for the validation data.
            losses_test (numpy.ndarray): The loss values for the testing data.
        """
        plt.figure(1)
        plt.plot(range(1, len(losses_train) + 1), losses_train, label="Loss Curve (Training)")
        plt.plot(range(1, len(losses_val) + 1), losses_val, label="Loss Curve (Validation)")
        plt.plot(range(1, len(losses_test) + 1), losses_test, label="Loss Curve (Testing)")
        plt.title("Loss Over Epochs")
        plt.xlabel("Epoch Number")
        plt.ylabel("Loss Value")
        plt.legend()
        plt.grid()
        plt.show()

def experiment1():
    """
    Compare average EER values and early stopping rates for different activation functions and learning rates.
    """
    eers = []
    for a in range(10):
        test.set_model_attributes()
        test.train_model()
        fpr, fnr, eer_index = test.evaluate_model(test.test_audio, test.test_labels)
        eers.append((fpr[eer_index] + fnr[eer_index]) / 2)
    print (f"Average EER value: {np.mean(eers)}")
    print (f"Rate of early stop occurrences: {test.early_stop_count / 10 * 100}")
    if test.early_stop_count > 0:
        print (f"Average no. of epochs before early stop: {test.early_stop_index / test.early_stop_count}")
    else:
        print ("Average no. of epochs before early stop: N/A")

def experiment2():
    """
    Obtain and plot loss and DET curves for model evaluation using different datasets.
    """
    test.set_model_attributes()
    test.train_model()
    test.plot_loss_curves(test.losses_train, test.losses_val, test.losses_test)
    train_fpr, train_fnr, train_eer_index = test.evaluate_model(test.train_audio, test.train_labels)
    val_fpr, val_fnr, val_eer_index = test.evaluate_model(test.validation_audio, test.validation_labels)
    test_fpr, test_fnr, test_eer_index = test.evaluate_model(test.test_audio, test.test_labels)
    test.plot_det_curves(train_fpr, val_fpr, test_fpr, 
                        train_fnr, val_fnr, test_fnr, 
                        train_eer_index, val_eer_index, test_eer_index)

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, choices=["experiment1", "experiment2"], required=True)
parser.add_argument("--activation_function", type=str, choices=["sigmoid", "relu"], required=True)
parser.add_argument("--learning_rate", type=float, choices=[0.1, 0.05, 0.01, 0.005, 0.001], required=True)
args = parser.parse_args()
test = VoiceActivityDetectionSystem(args.activation_function, 3, args.learning_rate)
test.prepare_data("NIS-2E07004-NI00x-R7", "EDI-1E07002-ED00x-R7", ["CMU-0E07000-CM00x-R7", "CMU-0E07000-CM01x-R7"])
if args.experiment == "experiment1":
    experiment1()
elif args.experiment == "experiment2":
    experiment2()
else:
    print ("ERROR: Invalid experiment name.")
