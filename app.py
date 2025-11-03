from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import os
import base64
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

# Enable CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'best.pt'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load YOLO model
try:
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


# -------------------- Helper Functions -------------------- #

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_image_url(unique_id, file_ext):
    """Generate a public URL for the annotated image."""
    return f"{request.host_url}results/{unique_id}/{unique_id}.{file_ext}"


# -------------------- Data Dictionaries -------------------- #

CONSTELLATION_NAMES = {
    0: 'Aquila', 1: 'Bootes', 2: 'Canis Major', 3: 'Canis Minor',
    4: 'Cassiopeia', 5: 'Cygnus', 6: 'Gemini', 7: 'Leo',
    8: 'Lyra', 9: 'Moon', 10: 'Orion', 11: 'Pleiades',
    12: 'Sagittarius', 13: 'Scorpius', 14: 'Taurus', 15: 'Ursa Major'
}

CONSTELLATION_DESCRIPTIONS = {
    "Aquila": "Aquila, the Eagle, is a constellation on the celestial equator...",
    "Bootes": "Bootes is a constellation in the northern sky...",
    "Canis Major": "Contains Sirius, the brightest star in the night sky.",
    "Ursa Major": "Ursa Major, the Great Bear, contains the Big Dipper asterism, useful for navigation."
    # (shortened for brevity)
}

CONSTELLATION_FACTS = {
    "Aquila": "• Contains Altair\n• Part of Summer Triangle\n• Visible June-November",
    "Bootes": "• Home to Arcturus\n• Visible March-September",
    "Ursa Major": "• Contains Big Dipper\n• Visible year-round"
}


# -------------------- Routes -------------------- #

@app.route('/', methods=['GET'])
def api_check():
    """Basic API health check"""
    return jsonify({'status': 'ok', 'message': 'API is running smoothly!'})


@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if API is running and model is loaded"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'total_classes': len(CONSTELLATION_NAMES),
        'constellations': list(CONSTELLATION_NAMES.values())
    })


@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Main prediction endpoint"""
    if request.method == 'OPTIONS':
        return '', 204

    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Upload PNG/JPG'}), 400

    try:
        unique_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.{file_ext}")
        file.save(input_path)
        print(f"📸 File saved: {input_path}")

        # Perform inference
        results = model.predict(
            input_path,
            save=True,
            project=RESULTS_FOLDER,
            name=unique_id,
            exist_ok=True,
            conf=0.25,
            iou=0.45
        )

        result = results[0]
        detections = []
        constellation_counts = {}

        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            constellation_name = CONSTELLATION_NAMES.get(cls_id, f"Unknown_{cls_id}")

            detections.append({
                'constellation': constellation_name,
                'confidence': round(confidence * 100, 2),
                'bbox': box.xyxy[0].tolist()
            })

            constellation_counts[constellation_name] = constellation_counts.get(constellation_name, 0) + 1

        annotated_image_url = generate_image_url(unique_id, file_ext)

        found_constellations = [{
            'name': name,
            'count': count,
            'description': CONSTELLATION_DESCRIPTIONS.get(name, "No description available."),
            'facts': CONSTELLATION_FACTS.get(name, "No additional facts available.")
        } for name, count in constellation_counts.items()]

        found_constellations.sort(key=lambda x: x['count'], reverse=True)

        # Clean up uploaded image
        try:
            os.remove(input_path)
        except:
            pass

        return jsonify({
            'success': True,
            'total_detections': len(detections),
            'constellations_found': len(constellation_counts),
            'constellations': found_constellations,
            'detections': detections,
            'annotated_image_url': annotated_image_url,
            'model_info': {
                'classes': len(CONSTELLATION_NAMES),
                'confidence_threshold': 0.25
            }
        })

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/results/<path:filename>')
def serve_results(filename):
    """Serve saved YOLO result images"""
    return send_file(os.path.join(RESULTS_FOLDER, filename))


@app.route('/api/constellations', methods=['GET'])
def get_constellation_info():
    constellations = [{
        'id': i,
        'name': name,
        'description': CONSTELLATION_DESCRIPTIONS.get(name, "No description available."),
        'facts': CONSTELLATION_FACTS.get(name, "No additional facts available.")
    } for i, name in CONSTELLATION_NAMES.items()]
    return jsonify({'total': len(constellations), 'constellations': constellations})


@app.route('/api/constellation/<name>', methods=['GET'])
def get_single_constellation(name):
    for cname in CONSTELLATION_NAMES.values():
        if cname.lower() == name.lower():
            return jsonify({
                'name': cname,
                'description': CONSTELLATION_DESCRIPTIONS.get(cname, "No description available."),
                'facts': CONSTELLATION_FACTS.get(cname, "No additional facts available.")
            })
    return jsonify({'error': 'Constellation not found'}), 404


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
