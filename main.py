import itertools

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_cnn3(input_shape):
    model = Sequential([
        Conv2D(64, (3, 3), activation="relu", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(256, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(512, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])
    return model


def plot_confusion_matrix(cm, classes, title="Confusion matrix", cmap=plt.cm.Blues, model_index=None, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)
    fmt = "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"figs/Model_Confusion_matrix.png", dpi=150)


def train_and_evaluate_cnn_model(dataset_directory, test_directory, image_height, image_width, batch_sz, num_epochs):
    input_shape = (image_height, image_width, 3)
    model = create_cnn3(input_shape)  # Create the model using the CNN architecture defined earlier

    model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])

    # Data Generators
    train_data_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2
    )
    train_gen = train_data_gen.flow_from_directory(
        dataset_directory,
        target_size=(image_height, image_width),
        batch_size=batch_sz,
        class_mode="binary",
        seed=42,
        subset="training"
    )
    val_gen = train_data_gen.flow_from_directory(
        dataset_directory,
        target_size=(image_height, image_width),
        batch_size=batch_sz,
        class_mode="binary",
        seed=42,
        subset="validation"
    )
    test_data_gen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_data_gen.flow_from_directory(
        directory=test_directory,
        target_size=(image_height, image_width),
        class_mode="binary",
        batch_size=batch_sz,
        seed=42,
        shuffle=False
    )

    # Training and Evaluation
    training_history = model.fit(train_gen, epochs=num_epochs, validation_data=val_gen)

    # Save the entire model including architecture, optimizer, and weights
    model.save(
        'saved_model_and_weights/model_cnn3.h5')  # Save the model at the specified path after training is complete

    # Optional: Save only the weights
    model.save_weights(
        'saved_model_and_weights/model_cnn3.weights.h5')  # Save only the weights of the model at the specified path

    # Evaluation and other outputs
    validation_loss, validation_accuracy = model.evaluate(val_gen)
    predictions = np.round(model.predict(test_gen))
    np.save(f"outputs/predictions.npy", predictions)

    # Plotting
    plt.figure()
    plt.plot(training_history.history["loss"], label="Train Loss")
    plt.plot(training_history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Loss Over Epochs")
    plt.savefig(f"figs/Loss.png", dpi=150)

    plt.figure()
    plt.plot(training_history.history["accuracy"], label="Train Accuracy")
    plt.plot(training_history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy Over Epochs")
    plt.savefig(f"figs/Accuracy.png", dpi=150)

    cm = confusion_matrix(test_gen.classes, predictions)
    cm_plot_label = test_gen.class_indices
    plot_confusion_matrix(cm, cm_plot_label, figsize=(6, 6))


if __name__ == "__main__":
    train_and_evaluate_cnn_model("./data/train", "./data/test", 64, 64, 32, 30)
    print("Code executed successfully!")
