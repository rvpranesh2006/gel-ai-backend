from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2

from utils.image_processing import process_image
from utils.anomaly_detection import detect_anomaly

app = Flask(__name__)

# ✅ ALLOW ALL ORIGINS (DEV MODE – SAFE FOR PROJECT)
CORS(app)

THRESHOLD = 0.05


def decode_base64_image(data):
    header, encoded = data.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        # ✅ THIS FIXES PREFLIGHT
        return "", 200

    try:
        data = request.get_json()
        image_base64 = data.get("image")

        if image_base64 is None:
            return jsonify({"error": "No image received"}), 400

        img = decode_base64_image(image_base64)

        gray, denoised, edges, bands = process_image(img)

        error, recon, heatmap = detect_anomaly(bands)

        result = "Anomalous" if error > THRESHOLD else "Normal"

        return jsonify({
            "error": float(error),
            "threshold": THRESHOLD,
            "result": result
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
