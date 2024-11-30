import numpy as np
from daal4py import decision_forest_classification_training, decision_forest_classification_prediction

# Simulated data: features (bacteria_count, virus_rna_level) and labels (0 = no outbreak, 1 = outbreak)
X = np.random.rand(1000, 2) * 100
y = np.random.randint(0, 2, 1000)

# Split into train and test sets
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train a Decision Forest model
train_result = decision_forest_classification_training(
    nTrees=100, featuresPerNode=2, maxTreeDepth=10
).compute(X_train, y_train)

# Predict on test data
prediction_result = decision_forest_classification_prediction().compute(X_test, train_result.model)
predictions = prediction_result.prediction

# Evaluate accuracy
accuracy = (predictions.flatten() == y_test).mean()
print(f"Model Accuracy: {accuracy:.2f}")
