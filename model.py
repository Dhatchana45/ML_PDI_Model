import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
import pandas as pd

# Define hyperparameters
batch_size = 64
epochs = 20
learning_rate = 0.001

# Load the pre-trained EfficientNetB0 model (include_top=False to exclude the top classification layer)
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
base_model.trainable = True

# Add custom classification layers on top of the base model
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
# Load disease labels dataset
disease_label_df = pd.read_csv("C:\\Users\\DELL\\Documents\\PDI model deployment\\Disease label.csv")

# Extract unique disease labels
disease_labels = disease_label_df['image_label'].unique()

# Get the number of classes
num_classes = len(disease_labels)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Load training data
train_generator = train_datagen.flow_from_dataframe(
    dataframe=disease_label_df,
    x_col="image_path",
    y_col="image_label",
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

# Define the path to the Disease Dataset directory
disease_dataset_dir = "C:\\Users\\DELL\\Documents\\Disease Dataset"

# Data augmentation for validation data (optional)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load validation data
validation_generator = validation_datagen.flow_from_directory(
    disease_dataset_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator))
model.save("trained.h5")

