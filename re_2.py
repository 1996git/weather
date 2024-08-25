import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Dense, Flatten 
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam 
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np

# グローバル変数の初期化
train_dir = None
test_dir = None
model = None

# 1. データの準備
def setup_data_generators(train_dir, test_dir):
    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        raise ValueError("Both training and test directories must be valid directories.")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'  # 'binary' for two classes
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'  # 'binary' for two classes
    )

    return train_generator, test_generator

# モデルの構築
def build_model(num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)  # sigmoid for binary classification

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',  # binary_crossentropy for two classes
                  metrics=['accuracy'])
    
    return model

# 画像の分類
def classify_image(img_path, model):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])  # Get the index of the highest probability
    result = f"Class index: {class_idx}, Probability: {predictions[0][class_idx]:.2f}"

    result_label.config(text=result)

# ファイル選択ダイアログの設定
def select_train_dir():
    global train_dir
    train_dir = filedialog.askdirectory(title="Select Training Data Directory")
    if train_dir:
        print("Selected train directory:", train_dir)
        if test_dir:  # Ensure test_dir is also selected
            start_button.config(state=tk.NORMAL)

def select_test_dir():
    global test_dir
    test_dir = filedialog.askdirectory(title="Select Test Data Directory")
    if test_dir:
        print("Selected test directory:", test_dir)
        if train_dir:  # Ensure train_dir is also selected
            start_button.config(state=tk.NORMAL)

def start_training():
    global model
    try:
        if not (train_dir and test_dir):
            result_label.config(text="Both training and test directories must be selected.")
            return
        train_generator, test_generator = setup_data_generators(train_dir, test_dir)
        model = build_model(1)  # Use 1 for binary classification
        history = model.fit(
            train_generator,
            epochs=10,
            validation_data=test_generator
        )
        result_label.config(text="Training complete!")
    except ValueError as e:
        result_label.config(text=str(e))

def select_file():
    global model
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if file_path and model:
        classify_image(file_path, model)

# GUIの設定
root = tk.Tk()
root.title("動物認識アプリ")

train_button = tk.Button(root, text="Select Training Data Directory", command=select_train_dir)
train_button.pack(pady=5)

test_button = tk.Button(root, text="Select Test Data Directory", command=select_test_dir)
test_button.pack(pady=5)

start_button = tk.Button(root, text="Start Training", command=start_training, state=tk.DISABLED)
start_button.pack(pady=5)

select_button = tk.Button(root, text="Select Image for Classification", command=select_file)
select_button.pack(pady=20)

result_label = tk.Label(root, text="Please select directories and start training")
result_label.pack(pady=20)

root.mainloop()
