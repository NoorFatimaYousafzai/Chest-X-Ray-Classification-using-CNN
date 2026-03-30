# i222610 Noor Fatima SE-A Generative Artifical Intelligence Assignment 01

import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score

print("TensorFlow version:", tf.__version__)

dataset_path = "C:/Users/O M E N/Desktop/Gen AI/Chest X-Ray Dataset"

train_dir = os.path.join(dataset_path, "train")
val_dir   = os.path.join(dataset_path, "val")
test_dir  = os.path.join(dataset_path, "test")

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.4
L2_LAMBDA = 0.0007
PATIENCE = 8

# Data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Rescaling validation and test data
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Test generator must have shuffle=false for correct evaluation.
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# Display sample images.
images, labels = next(train_generator)

plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(images[i])
    plt.title("Pneumonia" if labels[i] == 1 else "Normal")
    plt.axis("off")
plt.show()


# Custom CNN Model
def build_custom_cnn():
    model = models.Sequential([

        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        # Convolution Block 01
        layers.Conv2D(32, (3,3), activation='relu',
                      kernel_regularizer=regularizers.l2(L2_LAMBDA)),
        layers.MaxPooling2D(2,2),
         # Convolution Block 02
        layers.Conv2D(64, (3,3), activation='relu',
                      kernel_regularizer=regularizers.l2(L2_LAMBDA)),
        layers.MaxPooling2D(2,2),
        # Convolution Block 03
        layers.Conv2D(128, (3,3), activation='relu',
                      kernel_regularizer=regularizers.l2(L2_LAMBDA)),
        layers.MaxPooling2D(2,2),
        # Flatten and fully connected layers
        layers.Flatten(),

        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(L2_LAMBDA)),
        layers.Dropout(DROPOUT_RATE),
        # Output layer for binar classification
        layers.Dense(1, activation='sigmoid')
    ])

    return model

custom_model = build_custom_cnn()

# Compile Model
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

custom_model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

custom_model.summary()

# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True
)

# Train custom model
history_custom = custom_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stop]
)

# Plotting training history
def plot_history(history, title):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title(title + " Accuracy")

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title(title + " Loss")

    plt.show()

plot_history(history_custom, "Custom CNN")

# Evaluation of Custome CNN Model
test_generator.reset()
y_pred_probs = custom_model.predict(test_generator)
y_pred = (y_pred_probs > 0.5).astype(int)
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_true, y_pred))

# Precision, Recall and F1 Score
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="AUC = %0.2f" % roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

print("AUC:", roc_auc)


# ResNet50 Model (Pre-trained)
# y_true = test_generator.classes
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freezing base layers
base_model.trainable = False

model_resnet = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model_resnet.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_resnet = model_resnet.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stop]
)


# Evaluate ResNet50
test_generator.reset()
y_pred_probs_resnet = model_resnet.predict(test_generator)
y_pred_resnet = (y_pred_probs_resnet > 0.5).astype(int)

cm_resnet = confusion_matrix(y_true, y_pred_resnet)

plt.figure(figsize=(6,6))
sns.heatmap(cm_resnet, annot=True, fmt='d', cmap='Oranges')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("ResNet50 Confusion Matrix")
plt.show()

print("ResNet50 Classification Report")
print(classification_report(y_true, y_pred_resnet))

precision_resnet = precision_score(y_true, y_pred_resnet)
recall_resnet = recall_score(y_true, y_pred_resnet)
f1_resnet = f1_score(y_true, y_pred_resnet)

fpr_resnet, tpr_resnet, _ = roc_curve(y_true, y_pred_probs_resnet)
roc_auc_resnet = auc(fpr_resnet, tpr_resnet)

print("ResNet50 Precision:", precision_resnet)
print("ResNet50 Recall:", recall_resnet)
print("ResNet50 F1 Score:", f1_resnet)
print("ResNet50 AUC:", roc_auc_resnet)


# VGG16 Model (Pre-trained)
base_model_vgg = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freezing base layers
base_model_vgg.trainable = False

model_vgg = models.Sequential([
    base_model_vgg,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model_vgg.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_vgg = model_vgg.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stop]
)


# Evaluate VGG16
test_generator.reset()
y_pred_probs_vgg = model_vgg.predict(test_generator)
y_pred_vgg = (y_pred_probs_vgg > 0.5).astype(int)

cm_vgg = confusion_matrix(y_true, y_pred_vgg)

plt.figure(figsize=(6,6))
sns.heatmap(cm_vgg, annot=True, fmt='d', cmap='Greens')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("VGG16 Confusion Matrix")
plt.show()

print("VGG16 Classification Report")
print(classification_report(y_true, y_pred_vgg))

precision_vgg = precision_score(y_true, y_pred_vgg)
recall_vgg = recall_score(y_true, y_pred_vgg)
f1_vgg = f1_score(y_true, y_pred_vgg)

fpr_vgg, tpr_vgg, _ = roc_curve(y_true, y_pred_probs_vgg)
roc_auc_vgg = auc(fpr_vgg, tpr_vgg)

print("VGG16 Precision:", precision_vgg)
print("VGG16 Recall:", recall_vgg)
print("VGG16 F1 Score:", f1_vgg)
print("VGG16 AUC:", roc_auc_vgg)


print("\n================ MODEL COMPARISON ================")

print("ResNet50    -> Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, AUC: {:.4f}"
      .format(precision_resnet, recall_resnet, f1_resnet, roc_auc_resnet))

print("VGG16       -> Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, AUC: {:.4f}"
      .format(precision_vgg, recall_vgg, f1_vgg, roc_auc_vgg))

# Prediction Results
plt.figure(figsize=(10,10))

for i in range(9):
    img, label = test_generator[i]
    pred = custom_model.predict(img)
    
    plt.subplot(3,3,i+1)
    plt.imshow(img[0])
    plt.title("Pred: " + ("Pneumonia" if pred > 0.5 else "Normal"))
    plt.axis("off")

plt.show()