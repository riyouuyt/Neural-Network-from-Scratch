# 🧠 **Neural Networks: An Overview**

Neural networks are computational models inspired by the human brain's neural architecture. They are capable of learning patterns and relationships within data to make predictions or classifications. At the core, they consist of interconnected nodes (neurons), organized into layers—input, hidden, and output layers—that process information through weighted connections.


## Overview
The goal of this project is to build a neural network model that can accurately recognize handwritten digits. The MNIST dataset, which consists of 60,000 training images and 10,000 test images, is used for training and evaluating the model. The project demonstrates the process of designing and training a neural network from scratch.

## Project Structure
The project consists of the following files:

* neural_network.py: Contains the implementation of the neural network, including functions for initializing parameters, forward and backward propagation, updating parameters, and training the model.
* data_prep.py: Preprocesses the MNIST dataset by transforming it into numpy arrays, shuffling the data, and splitting it into training and testing sets.
* Neural Network.ipynb: Executes the training process and evaluates the model's performance on the test set. It also displays sample predictions and calculates the accuracy.
* README.md: The documentation file you are currently reading.

Getting Started
To run the project, follow these steps:

* Clone the repository to your local machine or download the project files.
* Install the required dependencies mentioned in the requirements.txt file.
* Open a terminal or command prompt and navigate to the project directory.
* Run the ipynb script using Python: python main.py.
* Acknowledgments

## 🔗 Architecture and Layers

The neural network in this code consists of an input layer, at least one hidden layer using ReLU activation, and an output layer utilizing softmax activation. The layers are interconnected by weights (W) and biases (b).

## 📊 Data Preparation

The MNIST dataset, a collection of handwritten digits, is loaded and divided into training and testing sets. These images are represented as arrays, each pixel containing grayscale values.

## 💡 Forward Propagation

The `forward` function performs the forward pass through the network. It computes weighted sums (Z) and activations (A) for each layer, using matrix multiplications (dot products) with weights and adding biases. The ReLU activation function introduces non-linearity by allowing only positive values to pass through.

Mathematically:

$$ Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]} $$

$$ A^{[l]} = g(Z^{[l]}) $$

where 

\( g() \) 

is ReLU for hidden layers and softmax for the output layer.


## 🧮 One-Hot Encoding

The `one_hot` function encodes the labels into a one-hot format, crucial for multi-class classification tasks, where each digit class is represented as a binary vector.

## 🔄 Backward Propagation

During backpropagation, the network adjusts its weights and biases to minimize the error. The gradients of the loss function with respect to the parameters are computed to update the weights using gradient descent.

## ⚙️ Updating Parameters

The `update_parameters` function uses the gradients computed during backpropagation to update the weights and biases by subtracting a fraction of the gradient scaled by the learning rate.

## 🔁 Training the Network

The `train` function orchestrates the training process, iterating through forward and backward passes for a specified number of iterations. It evaluates accuracy periodically and updates weights accordingly.

## 🎯 Testing and Prediction

The trained network is evaluated on the test set to calculate accuracy. Predictions are made on test images, and accuracy is measured by comparing predicted labels to actual labels.

## 📈 Results and Evaluation

The progression of accuracy across training iterations highlights the model's learning process, steadily improving from around 93.27% to 93.31% accuracy on the training data. This signifies the model's capacity to better fit the patterns within the training set as it iteratively adjusts its parameters. Eventually, the evaluation on the unseen test data, yielding an accuracy of approximately 92.68%, demonstrates the model's ability to generalize well to new, previously unseen instances. This high test set accuracy, alongside the incremental improvement during training iterations, indicates the model's effectiveness in learning and making accurate predictions on novel data, suggesting robust performance and a good ability to generalize beyond the training dataset.

## 🖼️ Visualizing Predictions

The code includes a visualization of five random test images alongside their predicted labels.

![image](https://github.com/riyouuyt/Neural-Network-from-Scratch/assets/122600889/e548e177-9e28-4297-a358-e2d7e1e50286)

## 📝 Conclusion

This neural network, trained on the MNIST dataset, demonstrates the fundamentals of building and training a simple feedforward neural network for digit recognition. Understanding its components—forward and backward passes, activation functions, and parameter updates—provides insights into the mechanics of neural networks.


This project is inspired by Samson Zhang's YouTube tutorial on building a neural network from scratch. His tutorial provided valuable insights and guidance in developing this project. You can find the tutorial [here](https://www.youtube.com/watch?v=w8yWXqWQYmU&t=1675s&pp=ygUObmV1cmFsIG5ldHdvcms%3D).

Special thanks to Samson Zhang for sharing his knowledge and expertise, which served as a foundation for this project.
