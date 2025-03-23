# Kubernetes-Cluster-Issues-Prediction




**Kubernetes Issue Prediction using Machine Learning**

This project aims to predict issues in Kubernetes clusters using machine learning models. By analyzing historical cluster metrics and logs, the system identifies potential failures and anomalies, allowing proactive issue resolution.

## **Project Aim:**

- **Proactive Issue Resolution**: Helps predict and prevent failures before they occur.
- **Reduced Downtime**: Maintains Kubernetes cluster stability.
- **Automated Anomaly Detection**: Reduces the need for manual monitoring.
- **Improved Resource Utilization**: Optimizes Kubernetes resource allocation and enhances performance.

## **Steps for usage:**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sumanth1018/Kubernetes-Cluster-Issues-Prediction#
   cd <repository_folder>
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare the Dataset**:
   - Ensure the dataset is in CSV format.
   - Place it in the designated directory.
4. **Run the Model**:
   ```bash
   python train_model.py
   ```
5. **Evaluate Predictions**:
   - Check logs and outputs.
   - Use visualizations to analyze performance.


## **Dataset Description**

The dataset contains various Kubernetes cluster metrics, including:

**Date first seen**: The timestamp when the network flow was first observed.

**Duration**: The total duration (in seconds) of the network flow.

**Proto**: The protocol used in the network flow (e.g., TCP, UDP).

**Src IP Addr**: The source IP address from which the connection originated.

**Src Pt**: The source port number of the connection.

**Dst IP Addr**: The destination IP address to which the connection was made.

**Dst Pt**: The destination port number of the connection.

**Packets**: The total number of packets exchanged in this flow.

**Bytes**: The total amount of data transferred (in bytes) in this flow.

**Flows**: The number of flow records related to this particular connection.

**Flags**: TCP connection flags that describe the state of the connection.

**Tos**: Type of Service (ToS) field, used for packet priority in networking.

**class**: Indicates whether the flow is classified as "normal" or an attack.

**attackType**: Specifies the type of attack (if any); "---" means no attack.

**attackID**: Unique identifier for the attack (if any).

**attackDescription**: A brief description of the attack type.

## **Data Preprocessing**

- **Handling Missing Values**: Missing values are replaced with zero.
- **Feature Engineering**:
  - Extracted time-based features (Hour, Day, Month) from timestamps.
  - Dropped original timestamp column.

## **Machine Learning Models Used**

Here's a step-by-step breakdown of your code, explaining all the stages, algorithms, and their multiple runs:

---

## **1. Data Loading & Preprocessing**
- **Load dataset:** Read the CIDDS-001 dataset (`CSV` file) using `pandas`.
- **Display basic information:** Print dataset shape and column names.
- **Check unique values in 'class' column:** Understand classification categories before encoding.

### **Column Cleaning**
- **Drop unnecessary columns:** Remove irrelevant columns (`attackID, attackDescription, attackType`).
- **Convert 'Bytes' column:** Handle `K, M, G` suffixes by converting them into numerical values.
- **Encode categorical columns:** Convert categorical columns (`Proto, Flags, class`) using `one-hot encoding`.

### **Final Cleaning**
- **Remove redundant columns** such as timestamps and IP addresses (`Date first seen, Src IP Addr, Dst IP Addr`).

---

## **2. Train-Test Split**
- **Feature (`X`) and label (`y`) separation:**
  - Features: All columns except `class_*` (encoded class labels).
  - Labels: Encoded `class_*` columns.
- **Split dataset (80%-20%)** into:
  - `X_train, X_test`
  - `y_train, y_test`
- **Print dataset shape** for verification.

---

## **3. Machine Learning Model: Random Forest**
- **Initialize RandomForestClassifier** with 100 trees (`n_estimators=100`).
- **Train the model** using `X_train, y_train`.
- **Predict on `X_test`** to generate class probabilities.
- **Convert one-hot labels back to class labels** using `idxmax()`.
- **Evaluate the model:**
  - Calculate **accuracy** (`0.9998`).
  - Print **classification report** (precision, recall, F1-score).

### **Feature Importance Analysis**
- Extract **feature importances** from RandomForest.
- Plot **bar chart** showing the most influential features.

### **Overfitting Check**
- Compare **training accuracy (1.0000) vs testing accuracy (0.9998)**.
- Perform **5-fold cross-validation**, achieving a **mean accuracy of 0.9996**.

---

## **4. Data Reshaping for LSTM**
- **Convert `X_train, X_test` into NumPy arrays**.
- **Reshape data for LSTM (`samples, timesteps, features`)**:
  - `X_train_seq.shape → (138270, 1, 36)`
  - `X_test_seq.shape → (34568, 1, 36)`

---

## **5. Deep Learning Model: LSTM**
- **Define an LSTM model**:
  - **LSTM layer** (64 units, `return_sequences=False`).
  - **Dense layer** (32 neurons, ReLU activation).
  - **Dropout (30%)** to prevent overfitting.
  - **Output layer** (3 neurons, softmax activation for classification).
- **Compile the model** using:
  - Optimizer: **Adam**
  - Loss function: **Categorical Crossentropy**
- **Train for 10 epochs** with batch size `32`:
  - Accuracy: **0.8958**
  - Loss: **0.2928**
- **Evaluate on test data**:
  - Test Accuracy: **0.8946**
  - Test Loss: **0.2958**
- **Plot Accuracy & Loss curves**.

---

## **6. Model Tuning: Increase Epochs**
- **Train for 20 epochs** to improve performance.
- **New evaluation results:**
  - Test Accuracy: **0.8957**
  - Test Loss: **0.2506**
- **Plot updated Accuracy & Loss curves**.

---

## **7. Early Stopping**
- Implement **EarlyStopping (patience=3)** to prevent overfitting.
- **Retrain LSTM for 20 epochs** with early stopping:
  - Test Accuracy: **0.9035**
  - Test Loss: **0.2059**
- **Plot updated Accuracy & Loss curves**.

---

## **8. Advanced LSTM: Regularization & Stacking**
- **Define a deeper LSTM model** with:
  - **Stacked LSTM layers** (128 + 64 units).
  - **L2 Regularization (0.01)** to control complexity.
  - **Increased Dropout (40%)** for better generalization.
- **Compile with a lower learning rate (`1e-4`)**.
- **Use ModelCheckpoint to save the best model**.
- **Train for 50 epochs** with early stopping.
- **Final results:**
  - Test Accuracy: **0.9356**
  - Test Loss: **0.2054**

---

## **9. Final Predictions & Model Evaluation**
- **Make predictions on `X_test_seq`**.
- **Convert probabilities to class labels** using `argmax`.
- **Compare predictions with true labels**.
- **Calculate final accuracy: 0.9356**.

---

### **Summary of Model Performance**
| Model  | Training Accuracy | Testing Accuracy | Overfitting? |
|---------|------------------|------------------|--------------|
| Random Forest | 1.0000 | 0.9998 | Yes (overfit) |
| LSTM (10 epochs) | ~0.89 | ~0.89 | No |
| LSTM (20 epochs) | ~0.90 | ~0.90 | No |
| LSTM (Regularized) | ~0.94 | ~0.94 | No |


## **Model Training and Evaluation**

- **Accuracy Improvement**:
  - Before hyperparameter tuning: **89%**
  - After tuning: **93.5%**
- **Evaluation Metrics**:
  - **Accuracy Score**: Measures prediction correctness.
  - **Precision & Recall**: Important for anomaly detection.
  - **Confusion Matrix**: Analyzes true positives and false negatives.

## **Future Enhancements**

- Implement advanced deep learning techniques for real-time monitoring.
- Integrate with Kubernetes monitoring tools like Prometheus.
- Develop an auto-healing mechanism for detected issues.

## **Conclusion**

This project demonstrates how machine learning can enhance Kubernetes cluster stability by predicting failures in advance. While traditional ML models faced overfitting, **LSTM was adopted for its ability to process sequential data effectively**, and hyperparameter tuning further improved accuracy from **89% to 93.5%**, making it a robust solution for proactive issue detection.

