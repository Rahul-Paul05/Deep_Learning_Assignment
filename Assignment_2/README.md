# 🧠 Assignment 2 — Multilayer Perceptron (MLP) for Iris Classification

## 1. 📌 Assignment Title

**Design and Implement a Multilayer Perceptron (MLP) for Classification of the Iris Dataset and Evaluate its Performance Using Accuracy and a Confusion Matrix**

---

## 2. 📝 Problem Statement

Design and implement a **Multilayer Perceptron (MLP)** for classification of the Iris dataset and evaluate its performance using accuracy and a confusion matrix.

---

## 3. 🎯 Objective

The objectives of this assignment are:

- Load and explore the Iris dataset
- Understand the features and target classes
- Visualize the dataset
- Split the dataset into training and testing sets
- Standardize the input features
- Design an MLP classification model
- Train the neural network
- Analyze training and validation performance
- Evaluate the model using test accuracy
- Generate a confusion matrix
- Generate a classification report
- Predict the class of a new Iris flower sample

---

## 4. 📊 Dataset Information

### Iris Dataset

The **Iris dataset** is a commonly used classification dataset containing measurements of three different species of Iris flowers:

- Setosa
- Versicolor
- Virginica

### Features

The dataset contains four numerical features:

| Feature | Description |
|---|---|
| Sepal Length | Length of the sepal in centimeters |
| Sepal Width | Width of the sepal in centimeters |
| Petal Length | Length of the petal in centimeters |
| Petal Width | Width of the petal in centimeters |

### Dataset Size

| Property | Value |
|---|---:|
| Total Samples | 150 |
| Features | 4 |
| Classes | 3 |
| Training Samples | 120 |
| Testing Samples | 30 |

The dataset is loaded using Scikit-learn:

```python
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target
```

---

## 5. 🔄 Dataset Preprocessing

The following preprocessing steps are performed.

### Step 1 — Load the Dataset

The Iris dataset is loaded using `sklearn.datasets.load_iris()`.

### Step 2 — Explore the Dataset

The dataset is examined using:

- Feature names
- Target class names
- Dataset shape
- First few records
- Statistical summary
- Missing-value check

### Step 3 — Data Visualization

The following visualizations are created:

- Species distribution
- Feature correlation heatmap
- Pair plot
- Box plots
- Feature distribution histograms

### Step 4 — Train-Test Split

The dataset is divided into:

- **80% training data**
- **20% testing data**

The split is performed using `random_state=42` and stratification.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

This results in:

```text
Training Data Shape: (120, 4)
Testing Data Shape:  (30, 4)
```

### Step 5 — Feature Standardization

The input features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Standardization helps bring the input features onto a comparable scale before training the neural network.

---

## 6. 🔬 Methodology

The overall methodology is:

```text
Iris Dataset
      ↓
Data Exploration
      ↓
Data Visualization
      ↓
Train-Test Split
      ↓
Feature Standardization
      ↓
Build MLP Model
      ↓
Compile Model
      ↓
Train Model
      ↓
Evaluate Model
      ↓
Confusion Matrix
      ↓
Classification Report
      ↓
New Sample Prediction
      ↓
Conclusion
```

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
| Metric | Accuracy |
| Epochs | 100 |
| Batch Size | 8 |
| Validation Split | 20% |

---

## 8. 💻 Implementation

### Import Required Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

### Load the Dataset

```python
iris = load_iris()

X = iris.data
y = iris.target
```

### Create the MLP Model

```python
model = Sequential()

model.add(Dense(16, activation="relu", input_shape=(4,)))
model.add(Dense(8, activation="relu"))
model.add(Dense(3, activation="softmax"))
```

### Compile the Model

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Train the Model

```python
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)
```

### Generate Predictions

```python
y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=1)
```

---

## 9. 📈 Training and Evaluation

The MLP model is trained for **100 epochs** with:

- Batch size: `8`
- Validation split: `20%`
- Optimizer: `Adam`
- Loss: `Sparse Categorical Crossentropy`
- Metric: `Accuracy`

Training and validation accuracy are plotted to analyze the model's learning behavior.

Training and validation loss are also plotted to observe the change in prediction error during training.

---

## 10. 📊 Results and Visualizations

### Test Accuracy

The trained MLP achieved:

```text
Test Accuracy: 93.33%
```

### Confusion Matrix

The resulting confusion matrix is:

```text
[[10  0  0]
 [ 0  9  1]
 [ 0  1  9]]
```

This indicates:

- **Setosa:** 10 out of 10 samples correctly classified
- **Versicolor:** 9 out of 10 samples correctly classified
- **Virginica:** 9 out of 10 samples correctly classified

Only two test samples were misclassified.

### Classification Report

The classification report shows approximately:

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Setosa | 1.00 | 1.00 | 1.00 |
| Versicolor | 0.90 | 0.90 | 0.90 |
| Virginica | 0.90 | 0.90 | 0.90 |
| **Overall Accuracy** | | | **0.93** |

### Visualizations

The experiment includes:

- Iris species distribution
- Feature correlation heatmap
- Pair plot
- Feature box plots
- Feature histograms
- Training accuracy curve
- Validation accuracy curve
- Training loss curve
- Validation loss curve
- Confusion matrix
- Classification report

---

## 11. 🔎 Analysis and Discussion

The experiment demonstrates how an MLP can be used to classify Iris flower species based on four numerical input features.

### Observations

- The Iris dataset contains three clearly defined target classes.
- Exploratory data analysis helps understand the distribution and relationship between features.
- Standardization is applied before training the neural network.
- The MLP contains two hidden layers with ReLU activation.
- The output layer contains three neurons with Softmax activation because the problem has three classes.
- The Adam optimizer is used for training.
- Training and validation accuracy increase during training.
- Training and validation loss decrease as the model learns.
- The final test accuracy is **93.33%**.
- Setosa is classified perfectly in the test set.
- The model has minor confusion between Versicolor and Virginica.

### New Sample Prediction

A new Iris flower sample is provided:

```text
[5.1, 3.5, 1.4, 0.2]
```

After standardization and prediction, the model classifies the sample as:

```text
Predicted Flower: Setosa
```

---

## 12. ✅ Conclusion

The assignment successfully demonstrates the implementation of a **Multilayer Perceptron (MLP)** for Iris flower classification using TensorFlow/Keras.

The Iris dataset was explored and visualized, the input features were standardized using `StandardScaler`, and an MLP with two hidden layers was trained using the Adam optimizer.

The model achieved a **test accuracy of 93.33%**. The confusion matrix and classification report demonstrate strong classification performance, with Setosa being classified perfectly and only minor confusion between Versicolor and Virginica.

The experiment provides practical understanding of:

- MLP architecture
- Data preprocessing
- Feature standardization
- Neural network training
- Accuracy evaluation
- Confusion matrix analysis
- Classification reports
- Prediction of new samples

This assignment builds a foundation for understanding more advanced neural network training concepts such as **forward propagation, backpropagation, learning rates, and epoch analysis** in the subsequent assignments.
