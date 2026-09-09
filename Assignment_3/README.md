# 🧠 Assignment 3 — Forward Propagation, Backpropagation & Hyperparameter Analysis

## 1. 📌 Assignment Title

**Implement Forward Propagation and Backpropagation Using TensorFlow/Keras and Analyze the Effect of Different Learning Rates and Number of Epochs on Model Performance**

---

## 2. 📝 Problem Statement

Implement forward propagation and backpropagation using TensorFlow/Keras and analyze the effect of different learning rates and the number of epochs on model performance.

---

## 3. 🎯 Objective

The objectives of this assignment are:

- Understand the working of a Multilayer Perceptron (MLP)
- Understand forward propagation
- Understand backpropagation
- Understand how weights are updated during training
- Understand the Adam optimizer
- Understand the role of the learning rate
- Understand the effect of the number of epochs
- Train an MLP using TensorFlow/Keras
- Analyze training and validation performance
- Compare different learning rates
- Compare different numbers of epochs
- Evaluate model performance using accuracy and loss

---

## 4. 📊 Dataset Information

### Iris Dataset

The **Iris dataset** is used for the experiments.

The dataset contains measurements of three Iris flower species:

- Setosa
- Versicolor
- Virginica

### Input Features

| Feature | Description |
|---|---|
| Sepal Length | Length of the sepal |
| Sepal Width | Width of the sepal |
| Petal Length | Length of the petal |
| Petal Width | Width of the petal |

### Dataset Properties

| Property | Value |
|---|---:|
| Total Samples | 150 |
| Input Features | 4 |
| Classes | 3 |
| Training Samples | 120 |
| Testing Samples | 30 |

The dataset is loaded using Scikit-learn.

---

## 5. 🔄 Dataset Preprocessing

### Step 1 — Load the Dataset

The Iris dataset is loaded using:

```python
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target
```

### Step 2 — Train-Test Split

The dataset is divided into:

- 80% training data
- 20% testing data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Step 3 — Feature Standardization

The input features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Standardization helps ensure that the input features are on a comparable scale before being provided to the neural network.

---

## 6. 🔬 Methodology

The experiment follows the following workflow:

```text
Iris Dataset
      ↓
Data Loading
      ↓
Train-Test Split
      ↓
Feature Standardization
      ↓
Build MLP
      ↓
Forward Propagation
      ↓
Calculate Loss
      ↓
Backpropagation
      ↓
Update Weights
      ↓
Model Evaluation
      ↓
Learning Rate Experiment
      ↓
Epoch Experiment
      ↓
Performance Analysis
```

---

### Forward Propagation

Forward propagation is the process of passing input data through the neural network to generate predictions.

For a neuron, the weighted sum is calculated as:

```text
z = Wx + b
```

The activation function is then applied.

For the hidden layers, ReLU is used:

```text
f(x) = max(0, x)
```

The final Softmax layer produces class probabilities.

The class with the highest probability becomes the predicted class.

---

### Backpropagation

Backpropagation is the process of updating the network weights based on the prediction error.

The process is:

```text
Forward Propagation
        ↓
Calculate Prediction
        ↓
Calculate Loss
        ↓
Calculate Gradients
        ↓
Propagate Error Backward
        ↓
Update Weights
```

The loss function used is **Sparse Categorical Crossentropy**.

---

### Adam Optimizer

The **Adam optimizer** is used for updating the neural network weights.

Adam combines ideas from momentum and RMSProp and adapts the learning rate for individual parameters.

---

## 7. 🏗️ Model Architecture

A Multilayer Perceptron is implemented using TensorFlow/Keras.

### Architecture

```text
Input Layer
4 Features
    ↓
Dense Layer
16 Neurons
ReLU Activation
    ↓
Dense Layer
8 Neurons
ReLU Activation
    ↓
Output Layer
3 Neurons
Softmax Activation
```

### Model Configuration

| Component | Configuration |
|---|---|
| Input Features | 4 |
| Hidden Layer 1 | 16 neurons |
| Activation | ReLU |
| Hidden Layer 2 | 8 neurons |
| Activation | ReLU |
| Output Layer | 3 neurons |
| Output Activation | Softmax |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Evaluation Metric | Accuracy |

---

## 8. 💻 Implementation

### Import Libraries

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

### Load Dataset

```python
iris = load_iris()

X = iris.data
y = iris.target

print("Features Shape:", X.shape)
print("Classes:", np.unique(y))
```

### Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Standardize Features

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Build the MLP

```python
model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])
```

### Compile the Model

```python
optimizer = Adam(learning_rate=0.01)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Train the Model

```python
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    verbose=1
)
```

### Evaluate the Model

```python
loss, accuracy = model.evaluate(X_test, y_test)

print("Loss:", loss)
print("Accuracy:", accuracy)
```

---

## 9. 📈 Training and Evaluation

The model is trained using:

- Adam optimizer
- Learning rate = `0.01`
- 50 epochs
- Sparse Categorical Crossentropy loss
- Accuracy as the evaluation metric

### Accuracy Curve

Training and validation accuracy are plotted against the number of epochs.

```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epochs")
plt.legend()
plt.show()
```

### Loss Curve

Training and validation loss are plotted against the number of epochs.

```python
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()
plt.show()
```

The submitted experiment shows that accuracy generally increases rapidly during the early epochs, while training and validation loss decrease substantially as the model learns. :contentReference[oaicite:3]{index=3}

---

## 10. 📊 Results and Visualizations

### Learning Rate Experiment

Three different learning rates are evaluated:

```python
learning_rates = [0.1, 0.01, 0.001]
```

Each model is trained for 30 epochs and evaluated on the test set. :contentReference[oaicite:4]{index=4}

### Results

| Learning Rate | Test Accuracy |
|---:|---:|
| `0.1` | **30.00%** |
| `0.01` | **96.67%** |
| `0.001` | **100.00%** |

### Observation

The learning rate has a significant effect on model performance.

- `0.1` is too high and results in poor performance.
- `0.01` provides strong performance.
- `0.001` provides the highest observed accuracy in this experiment.

---

### Epoch Experiment

The following epoch values are compared:

```python
epochs_list = [10, 30, 50]
```

The learning rate is fixed at `0.01` while the number of epochs is changed. :contentReference[oaicite:5]{index=5}

### Results

| Number of Epochs | Test Accuracy |
|---:|---:|
| `10` | **96.67%** |
| `30` | **96.67%** |
| `50` | **93.33%** |

### Observation

Increasing the number of epochs does not always improve test accuracy.

In this experiment:

- 10 epochs achieved **96.67%**
- 30 epochs also achieved **96.67%**
- 50 epochs achieved **93.33%**

This demonstrates that training for more epochs does not necessarily guarantee better generalization.

---

### Summary of Experiments

| Experiment | Parameter | Accuracy |
|---|---:|---:|
| Learning Rate | `0.1` | **30.00%** |
| Learning Rate | `0.01` | **96.67%** |
| Learning Rate | `0.001` | **100.00%** |
| Epochs | `10` | **96.67%** |
| Epochs | `30` | **96.67%** |
| Epochs | `50` | **93.33%** |

---

## 11. 🔎 Analysis and Discussion

This experiment demonstrates the relationship between neural network training parameters and model performance.

### Effect of Learning Rate

The learning rate controls the size of the weight updates during optimization.

A very high learning rate can cause the optimizer to make excessively large updates and fail to converge effectively.

This is demonstrated by:

```text
Learning Rate = 0.1
Accuracy      = 30.00%
```

The learning rate of `0.01` provides significantly better performance:

```text
Learning Rate = 0.01
Accuracy      = 96.67%
```

The lowest tested learning rate of `0.001` produces the highest observed accuracy:

```text
Learning Rate = 0.001
Accuracy      = 100.00%
```

Therefore, selecting an appropriate learning rate is important for stable and effective training.

---

### Effect of Number of Epochs

An epoch represents one complete pass through the training dataset.

The experiment compares 10, 30, and 50 epochs.

The results show:

```text
10 Epochs → 96.67%
30 Epochs → 96.67%
50 Epochs → 93.33%
```

The results indicate that increasing the number of epochs beyond a suitable point may not improve generalization and can potentially reduce test performance.

---

### Forward Propagation and Backpropagation

The complete training process consists of:

1. Forward propagation
2. Loss calculation
3. Gradient calculation
4. Backpropagation
5. Weight updates
6. Repeated optimization over multiple epochs

This process allows the neural network to learn the relationship between the Iris input features and their corresponding classes.

---

## 12. ✅ Conclusion

The assignment successfully demonstrates **forward propagation and backpropagation using TensorFlow/Keras** with an MLP applied to the Iris dataset.

The model uses:

- 4 input features
- Two hidden layers with 16 and 8 neurons
- ReLU activation
- 3-neuron Softmax output layer
- Adam optimizer
- Sparse Categorical Crossentropy loss

The experiments demonstrate that both **learning rate** and **number of epochs** have a significant effect on model performance.

The learning-rate experiment achieved its best observed accuracy of **100.00% with a learning rate of 0.001**, while a learning rate of `0.1` resulted in only **30.00% accuracy**.

The epoch experiment achieved **96.67% accuracy at 10 and 30 epochs**, while the 50-epoch experiment achieved **93.33%**.

Overall, the experiment demonstrates that selecting appropriate hyperparameters is important for achieving good model performance and generalization.

---
