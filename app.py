from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import io
import base64
import cv2
from datetime import datetime
from tensorflow.keras.models import Model

# ✅ Local directory to store images
SAVE_DIR = 'images'
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ Load model (update path if needed)
MODEL_PATH = 'model/model-92.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Class info
class_labels = ["Acne", "Eczema", "Melanoma", "Normal Skin", "Psoriasis", "Warts"]
class_thresholds = {i: 0.7 for i in range(len(class_labels))}

# ✅ Preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image).astype('float32') / 255.0
    return np.expand_dims(image_array, axis=0)

# ✅ Grad-CAM implementation
def grad_cam(model, img_array, layer_name):
    grad_model = Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_class = tf.argmax(predictions[0])
        loss = predictions[:, pred_class]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0].numpy()
    for i in range(conv_outputs.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    return heatmap

# ✅ Flask setup
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No image uploaded'}), 400
    try:
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))
        processed = preprocess_image(image)

        # Predict
        preds = model.predict(processed)[0]
        result = [{"class": class_labels[i], "percentage": float(f"{p:.5f}")} for i, p in enumerate(preds)]
        max_idx = np.argmax(preds)
        confidence = preds[max_idx]
        conf_msg = f"Confidence: {confidence:.2%}" if confidence >= class_thresholds[max_idx] else "Low confidence. Better check with a specialist."

        # ✅ Grad-CAM
        img_cv2 = cv2.cvtColor(np.array(image.resize((224, 224))), cv2.COLOR_RGB2BGR)
        img_array = np.expand_dims(img_cv2[..., ::-1] / 255.0, axis=0)
        heatmap = grad_cam(model, img_array, "block5_conv2")  # NOTE: Update to actual layer name if different
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(np.uint8(img_cv2), 0.6, heatmap, 0.4, 0)

        _, buffer = cv2.imencode('.jpg', superimposed_img)
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')

        # ✅ Save image
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        safe_class = class_labels[max_idx].replace(" ", "_")
        filename = f"{safe_class}_{timestamp}_{file.filename}"
        save_path = os.path.join(SAVE_DIR, filename)
        image.save(save_path)

        return jsonify({
            "prediction": result,
            "predicted_class": class_labels[max_idx],
            "confidence": conf_msg,
            "saved_to": save_path,
            "gradcam_image_base64": heatmap_base64
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ✅ Main entry point for Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
