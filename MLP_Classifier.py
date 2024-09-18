import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# Load the dataset
file_path = 'Crânios Egípcios Formatado.csv'
df = pd.read_csv(file_path)

# Use the skull measurements as features (X) and the labels (y)
X_era = df[['X1', 'X2', 'X3', 'X4']].values
y_era = df['label'].map({
  'Pré-dinástico primitivo': 0,
  'Pré-dinástico antigo': 1,
  '12 e 13 dinastias': 2,
  'Período ptolemaico': 3,
  'Período romano': 4
}).values

# One-hot encode the labels
y_era = to_categorical(y_era, num_classes=5)  # Ensure 5 classes

# Split the data into training and testing sets
X_train_era, X_test_era, y_train_era, y_test_era = train_test_split(X_era, y_era, test_size=0.2, random_state=42)

# Normalize the data
scaler_era = StandardScaler()
X_train_era = scaler_era.fit_transform(X_train_era)
X_test_era = scaler_era.transform(X_test_era)

batch_size = 16
epochs = 100
allow_data_augmentation = True

# Data augmentation: Multiple techniques for data augmentation
def augment_data(data, labels):
  if allow_data_augmentation:
    augmented_data = data.copy()
    augmented_labels = labels.copy()
    
    # Method 1: Add different levels of random Gaussian noise
    for noise_factor in [0.05, 0.1, 0.2]:
        noisy_data = data + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        augmented_data = np.vstack((augmented_data, noisy_data))
        augmented_labels = np.vstack((augmented_labels, labels))
    
    # Method 2: Slight perturbation based on mean and standard deviation of features
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    perturbed_data = data + np.random.normal(loc=data_mean, scale=data_std, size=data.shape)
    augmented_data = np.vstack((augmented_data, perturbed_data))
    augmented_labels = np.vstack((augmented_labels, labels))
    
    return augmented_data, augmented_labels
  else:
    return data, labels

# Augment the training data
X_train_era_aug, y_train_era_aug = augment_data(X_train_era, y_train_era)

# Define the TensorFlow model with Dropout layers to prevent overfitting
model = Sequential([
  Dense(64, activation='relu', input_shape=(4,), kernel_regularizer=l2(0.001)),  # L2 regularization
  Dropout(0.4),  # Increase dropout to 40%
  Dense(32, activation='relu', kernel_regularizer=l2(0.001)),  # Add another layer with L2
  Dropout(0.4),
  Dense(5, activation='softmax')  # 5 output classes (one for each era)
])

# Compile the model with Adam optimizer and categorical cross-entropy loss
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting and reduce learning rate on plateau
# Increased patience for early stopping to allow for longer plateaus before stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001)

# Train the model with smaller batch size and store the training history
history = model.fit(X_train_era_aug, y_train_era_aug, epochs=epochs, validation_data=(X_test_era, y_test_era),
                    batch_size=batch_size, callbacks=[early_stopping, reduce_lr])

# Plot training & validation loss and accuracy using seaborn
sns.set(style='whitegrid')
plt.figure(figsize=(12, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
sns.lineplot(x=range(len(history.history['accuracy'])), y=history.history['accuracy'], label='Train Accuracy', color='blue')
sns.lineplot(x=range(len(history.history['val_accuracy'])), y=history.history['val_accuracy'], label='Test Accuracy', color='green')
plt.title('Train and Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
sns.lineplot(x=range(len(history.history['loss'])), y=history.history['loss'], label='Train Loss', color='red')
sns.lineplot(x=range(len(history.history['val_loss'])), y=history.history['val_loss'], label='Test Loss', color='orange')
plt.title('Train and Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.tight_layout()
plt.savefig('tensorflow_loss_accuracy_plot.png')
plt.show()

# Save the model
model.save('tensorflow_cranios_classifier.h5')