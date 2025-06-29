from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__)

# ✅ Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ Load class names
with open('class_names.txt') as f:
    class_names = [line.strip() for line in f if line.strip()]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    file_bytes = np.frombuffer(file.read(), np.uint8)

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ✅ Make sure input dtype matches TFLite model input
    if input_details[0]['dtype'] == np.float32:
        img = img.astype(np.float32)
        # If you normalized during training, divide by 255.
        # For you, you said you did NOT normalize:
        # So: keep it as 0–255 float
    elif input_details[0]['dtype'] == np.uint8:
        img = img.astype(np.uint8)
    else:
        return jsonify({'error': f'Unsupported input dtype {input_details[0]["dtype"]}'}), 400

    # Add batch dim
    img_array = np.expand_dims(img, axis=0)

    # ✅ Set input
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # ✅ Run inference
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_idx = np.argmax(output_data[0])
    predicted_class = class_names[predicted_idx]

    return jsonify({'class': predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)