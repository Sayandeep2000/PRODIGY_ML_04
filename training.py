import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Determine the desired length (e.g., the most common length)
desired_length = 42  # This should be set based on your data or requirements

# Function to pad or trim the sequences
def pad_or_trim(sequence, length):
    if len(sequence) > length:
        return sequence[:length]
    elif len(sequence) < length:
        return sequence + [0] * (length - len(sequence))
    else:
        return sequence

# Process the data to have uniform length
processed_data = [pad_or_trim(sample, desired_length) for sample in data_dict['data']]

# Convert the data to a NumPy array
data = np.array(processed_data)
labels = np.array(data_dict['labels'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict on the test set
y_predict = model.predict(x_test)

# Calculate the accuracy score
score = accuracy_score(y_predict, y_test)
print(f"Accuracy: {score}")

# Perform 5-fold cross-validation
cross_val_scores = cross_val_score(model, data, labels, cv=5)
print(f"Cross-Validation Scores: {cross_val_scores}")
print(f"Mean Cross-Validation Score: {np.mean(cross_val_scores)}")

# Calculate and print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_predict)
print(f"Confusion Matrix:\n{conf_matrix}")

# Print the classification report
class_report = classification_report(y_test, y_predict)
print(f"Classification Report:\n{class_report}")

# Predict on a few test samples
for i in range(10):
    print(f"True Label: {y_test[i]}, Predicted Label: {y_predict[i]}")


f = open('model.p','wb')
pickle.dump({'model': model},f)
f.close()

