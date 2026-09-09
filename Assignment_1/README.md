# 🧠 Assignment 1 — TensorFlow/Keras Setup & Data Preprocessing

## 1. 📌 Assignment Title

**Install and Configure TensorFlow/Keras in Google Colab and Perform Data Preprocessing, Normalization, Train-Test Split, and Visualization on a Sample Dataset**

---

## 2. 📝 Problem Statement

Install and configure TensorFlow/Keras in Google Colab. Perform data preprocessing, normalization, train-test split, and visualization on a sample dataset.

---

## 3. 🎯 Objective

The objective of this assignment is to:

- Configure and use TensorFlow/Keras in Google Colab
- Load a sample dataset using Keras
- Split the dataset into training and testing sets
- Visualize sample images
- Normalize pixel values
- Build a basic Sequential Neural Network
- Train the model using the training data
- Evaluate the model
- Predict the class of test images
- Visualize the prediction result

---

## 4. 📊 Dataset Information

### MNIST Dataset

The **MNIST handwritten digit dataset** is used for this assignment.

MNIST contains grayscale images of handwritten digits from **0 to 9**.

Each image has a size of:

```text
28 × 28 pixels
```

### Dataset Split

| Dataset | Number of Images | Image Shape |
|---|---:|---|
| Training | 60,000 | 28 × 28 |
| Testing | 10,000 | 28 × 28 |

The dataset is loaded directly using:

```python
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

---

## 5. 🔄 Dataset Preprocessing

The following preprocessing steps are performed:

### Step 1 — Load the Dataset

The MNIST dataset is loaded using Keras.

### Step 2 — Train-Test Split

The dataset is already divided into:

- Training data
- Testing data

The training data contains **60,000 images**, while the testing data contains **10,000 images**.

### Step 3 — Visualization

Five sample training images are displayed using Matplotlib to understand the input data.

### Step 4 — Normalization

The pixel values originally range from **0 to 255**.

The values are normalized by dividing them by `255.0`:

```python
x_train = x_train / 255.0
x_test = x_test / 255.0
```

After normalization:

```text
Minimum Pixel Value = 0.0
Maximum Pixel Value = 1.0
```

Normalization helps provide appropriately scaled input values for model training.

---

## 6. 🔬 Methodology

The following workflow is used:

```text
MNIST Dataset
      ↓
Load Dataset
      ↓
Train-Test Split
      ↓
Visualize Sample Images
      ↓
Normalize Pixel Values
      ↓
Build Neural Network
      ↓
Compile Model
      ↓
Train Model
      ↓
Evaluate Model
      ↓
Generate Predictions
      ↓
Visualize Prediction
```

---

## 7. 🏗️ Model Architecture

A **Sequential Neural Network** is implemented using TensorFlow/Keras.

### Architecture

```text
Input Image
   ↓
Flatten
   ↓
Dense Layer — 128 neurons
   ↓
ReLU Activation
   ↓
Dense Layer — 64 neurons
   ↓
ReLU Activation
   ↓
Dense Output Layer — 10 neurons
   ↓
Softmax Activation
```

### Model Configuration

| Component | Configuration |
|---|---|
| Input | 28 × 28 grayscale image |
| Flatten Layer | Converts image into a 1D vector |
| Hidden Layer 1 | 128 neurons, ReLU |
| Hidden Layer 2 | 64 neurons, ReLU |
| Output Layer | 10 neurons, Softmax |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Evaluation Metric | Accuracy |
| Epochs | 5 |
| Validation Split | 20% |

---

## 8. 💻 Implementation

### Import Libraries

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
```

### Load the MNIST Dataset

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training Data Shape:", x_train.shape)
print("Testing Data Shape:", x_test.shape)
```

### Visualize Sample Images

```python
plt.figure(figsize=(10,5))

for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(y_train[i])
    plt.axis('off')

plt.show()
```

### Normalize the Data

```python
x_train = x_train / 255.0
x_test = x_test / 255.0

print("Maximum Pixel Value:", x_train.max())
print("Minimum Pixel Value:", x_train.min())
```

### Build the Model

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### Compile the Model

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Train the Model

```python
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.2
)
```

### Generate Predictions

```python
predictions = model.predict(x_test)

print("Predicted Label:", np.argmax(predictions[0]))
print("Actual Label:", y_test[0])
```

### Visualize the Prediction

```python
plt.imshow(x_test[0], cmap='gray')
plt.title("Predicted: {}".format(np.argmax(predictions[0])))
plt.axis('off')
plt.show()
```

---

## 9. 📈 Training and Evaluation

The model is trained for **5 epochs** using the Adam optimizer.

A **20% validation split** is used during training.

### Training Performance

| Epoch | Training Accuracy | Validation Accuracy |
|---:|---:|---:|
| 1 | ~91.98% | ~96.5% |
| 2 | ~96.41% | ~96.7% |
| 3 | ~97.92% | ~96.93% |
| 4 | ~98.18% | ~97.03% |
| 5 | ~98.63% | ~97.32% |

The training output shows that the model's accuracy increases over the five epochs while the loss decreases.

---

## 10. 📊 Results and Visualizations

### Dataset Visualization

Five sample MNIST images are visualized before training.

The displayed examples represent handwritten digits such as:

```text
5   0   4   1   9
```

### Normalization Result

After normalization:

```text
Maximum Pixel Value: 1.0
Minimum Pixel Value: 0.0
```

### Model Training

The model reaches approximately:

```text
Training Accuracy:   98.63%
Validation Accuracy: 97.32%
```

by the fifth epoch.

### Prediction Result

For the displayed test image:

```text
Predicted Label: 7
Actual Label:    7
```

The prediction is therefore correct for the shown test example.

---

## 11. 🔎 Analysis and Discussion

The experiment demonstrates the basic Deep Learning workflow using TensorFlow/Keras.

### Observations

- The MNIST dataset can be loaded directly using Keras.
- Normalization changes pixel values from the original `0–255` range to `0–1`.
- The Flatten layer converts the `28 × 28` image into a one-dimensional representation.
- ReLU activation is used in the hidden layers.
- Softmax is used in the output layer for multi-class digit classification.
- The Adam optimizer is used to update the model weights.
- Training accuracy improves consistently over the five epochs.
- The model achieves approximately **98.63% training accuracy** and **97.32% validation accuracy** after five epochs.
- The displayed test example is correctly classified as digit **7**.

The experiment provides a practical introduction to dataset preprocessing, neural network construction, training, evaluation, and prediction.

---

## 12. ✅ Conclusion

The assignment successfully demonstrates the basic implementation workflow of a Deep Learning model using **TensorFlow/Keras in Google Colab**.

The MNIST dataset was loaded and visualized, the images were normalized, and a Sequential Neural Network was constructed using Flatten and Dense layers.

The model was trained for five epochs using the **Adam optimizer** and **Sparse Categorical Crossentropy** loss function.

The final training output achieved approximately **98.63% training accuracy** and **97.32% validation accuracy**, and the demonstrated test image was correctly predicted as digit **7**.

This assignment provides a foundation for understanding more advanced neural network architectures and Deep Learning techniques in the subsequent assignments.
