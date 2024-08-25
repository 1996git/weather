import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tkinter import Tk, filedialog
from PIL import Image, ImageDraw, ImageFont

# TensorFlowモデルのロード
model = MobileNetV2(weights='imagenet')

def recognize_and_annotate_image(img_path):
    # 画像の読み込みと前処理
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # 画像認識
    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=1)[0]
    label = decoded_preds[0][1]  # ラベル
    confidence = decoded_preds[0][2]  # 確信度

    # 認識結果の表示
    print(f"認識結果: {label} ({confidence:.2f})")

    # 画像に認識結果を追加
    img_pil = Image.open(img_path)
    draw = ImageDraw.Draw(img_pil)
    
    try:
        # フォントの指定
        font = ImageFont.truetype("arial.ttf", size=30)
    except IOError:
        # フォントファイルが見つからない場合はデフォルトフォントを使用
        font = ImageFont.load_default()

    text = f"{label} ({confidence:.2f})"
    
    # テキストのバウンディングボックスを取得
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    text_x = 10
    text_y = 10

    # 画像の左上にテキストを描画
    draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))

    # 保存先のファイルパスを取得
    output_path = img_path.replace('.jpg', '_annotated.jpg')  # 拡張子を変更して保存
    img_pil.save(output_path)
    print(f"画像に情報を追加して保存しました: {output_path}")

def main():
    # Tkinterを使ってファイルダイアログを開く
    Tk().withdraw()
    file_paths = filedialog.askopenfilenames(title="画像ファイルを選択", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])

    for file_path in file_paths:
        recognize_and_annotate_image(file_path)

if __name__ == "__main__":
    main()
