from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import os
import base64
from PIL import Image
import io
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

# FIXED: Enable CORS with proper configuration for all origins
CORS(app, resources={
    r"/api/*": {
        "origins": "*",  # Allow all origins (for development)
        "allow_headers": ["Content-Type", "Accept"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False
    }
})

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'best.pt'

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load the trained YOLO model
try:
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Constellation names mapping
CONSTELLATION_NAMES = {
    0: 'Aquila', 1: 'Bootes', 2: 'Canis Major', 3: 'Canis Minor',
    4: 'Cassiopeia', 5: 'Cygnus', 6: 'Gemini', 7: 'Leo',
    8: 'Lyra', 9: 'Moon', 10: 'Orion', 11: 'Pleiades',
    12: 'Sagittarius', 13: 'Scorpius', 14: 'Taurus', 15: 'Ursa Major'
}

# Detailed constellation descriptions
CONSTELLATION_DESCRIPTIONS = {
    "Aquila": "Aquila, the Eagle, is a constellation on the celestial equator. Its brightest star, Altair, is one of the vertices of the Summer Triangle asterism.",
    "Bootes": "Bootes is a constellation in the northern sky, containing the bright star Arcturus, the fourth-brightest star in the night sky.",
    "Canis Major": "Canis Major, the Greater Dog, contains Sirius, the brightest star in the night sky. It represents one of Orion's hunting dogs.",
    "Canis Minor": "Canis Minor, the Lesser Dog, contains Procyon, forming one vertex of the Winter Triangle asterism.",
    "Cassiopeia": "Cassiopeia is easily recognizable due to its distinctive 'W' or 'M' shape formed by five bright stars.",
    "Cygnus": "Cygnus, the Swan, contains Deneb and forms the Northern Cross asterism.",
    "Gemini": "Gemini, the Twins, contains the bright stars Castor and Pollux, named after twin brothers in Greek mythology.",
    "Leo": "Leo, the Lion, contains Regulus and is easily identified by its distinctive sickle-shaped asterism.",
    "Lyra": "Lyra represents Orpheus's lyre and contains Vega, one of the brightest stars visible from Earth.",
    "Moon": "The Moon is Earth's only natural satellite and the brightest object in the night sky after the Sun.",
    "Orion": "Orion is one of the most recognizable constellations, containing Betelgeuse, Rigel, and the famous Orion's Belt.",
    "Pleiades": "The Pleiades, also known as the Seven Sisters, is an open star cluster in Taurus, one of the nearest to Earth.",
    "Sagittarius": "Sagittarius contains the center of our Milky Way galaxy and is represented as a centaur drawing a bow.",
    "Scorpius": "Scorpius contains the red supergiant Antares and has a distinctive J-shaped pattern representing a scorpion.",
    "Taurus": "Taurus contains Aldebaran and two famous star clusters: the Pleiades and the Hyades.",
    "Ursa Major": "Ursa Major, the Great Bear, contains the Big Dipper asterism, useful for navigation."
}

# Quick facts
CONSTELLATION_FACTS = {
    "Aquila": "• Contains Altair\n• Part of Summer Triangle\n• Visible June-November",
    "Bootes": "• Home to Arcturus\n• Visible March-September\n• Name means 'plowman'",
    "Canis Major": "• Contains Sirius\n• Follows Orion\n• Best visible in winter",
    "Canis Minor": "• Contains Procyon\n• Forms Winter Triangle\n• Small constellation",
    "Cassiopeia": "• Circumpolar constellation\n• Distinctive 'W' shape\n• Named after a queen",
    "Cygnus": "• Northern Cross\n• Contains Deneb\n• Located in Milky Way",
    "Gemini": "• Third zodiac sign\n• Twins Castor & Pollux\n• Best in Jan-Feb",
    "Leo": "• Fifth zodiac sign\n• Contains Regulus\n• Best visible in spring",
    "Lyra": "• Contains Vega\n• Home to Ring Nebula\n• Orpheus's lyre",
    "Moon": "• 384,400 km from Earth\n• Diameter: 3,474 km\n• Only visited celestial body",
    "Orion": "• Most recognizable\n• Contains Orion Nebula\n• Best in winter",
    "Pleiades": "• Seven Sisters\n• 444 light-years away\n• Over 1,000 stars",
    "Sagittarius": "• Points to galactic center\n• Ninth zodiac sign\n• Best in summer",
    "Scorpius": "• Contains Antares\n• Zodiac constellation\n• Best in summer",
    "Taurus": "• Second zodiac sign\n• Contains Pleiades\n• Best in winter",
    "Ursa Major": "• Third largest\n• Contains Big Dipper\n• Visible year-round"
}


@app.route('/', methods=['GET'])
def api_check():
    """Basic API health check"""
    return jsonify({
        'status': 'ok',
        'message': 'API is running smoothly!'
    })


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
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 204
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Check if image file is present
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload PNG or JPG'}), 400
    
    try:
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.{file_ext}")
        
        # Save uploaded file
        file.save(input_path)
        print(f"File saved: {input_path}")
        
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
        
        # Process results
        result = results[0]
        detections = []
        constellation_counts = {}
        
        # Extract detection information
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
        
        # Get the annotated image path
        annotated_image_path = os.path.join(RESULTS_FOLDER, unique_id, f"{unique_id}.{file_ext}")
        
        # Convert annotated image to base64
        with open(annotated_image_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Prepare constellation details
        found_constellations = []
        for const_name, count in constellation_counts.items():
            found_constellations.append({
                'name': const_name,
                'count': count,
                'description': CONSTELLATION_DESCRIPTIONS.get(const_name, "No description available."),
                'facts': CONSTELLATION_FACTS.get(const_name, "No additional facts available.")
            })
        
        # Sort by count
        found_constellations.sort(key=lambda x: x['count'], reverse=True)
        
        # Cleanup
        try:
            os.remove(input_path)
        except:
            pass
        
        print(f"Successfully processed image. Found {len(constellation_counts)} unique constellations")
        
        return jsonify({
            'success': True,
            'total_detections': len(detections),
            'constellations_found': len(constellation_counts),
            'constellations': found_constellations,
            'detections': detections,
            'annotated_image': f"data:image/{file_ext};base64,{img_data}",
            'model_info': {
                'classes': len(CONSTELLATION_NAMES),
                'confidence_threshold': 0.25
            }
        })
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/constellations', methods=['GET'])
def get_constellation_info():
    """Get information about all supported constellations"""
    constellations = []
    for cls_id, name in CONSTELLATION_NAMES.items():
        constellations.append({
            'id': cls_id,
            'name': name,
            'description': CONSTELLATION_DESCRIPTIONS.get(name, "No description available."),
            'facts': CONSTELLATION_FACTS.get(name, "No additional facts available.")
        })
    return jsonify({
        'total': len(constellations),
        'constellations': constellations
    })


@app.route('/api/constellation/<name>', methods=['GET'])
def get_single_constellation(name):
    """Get detailed information about a specific constellation"""
    constellation_name = None
    for const_name in CONSTELLATION_NAMES.values():
        if const_name.lower() == name.lower():
            constellation_name = const_name
            break
    
    if not constellation_name:
        return jsonify({'error': 'Constellation not found'}), 404
    
    return jsonify({
        'name': constellation_name,
        'description': CONSTELLATION_DESCRIPTIONS.get(constellation_name, "No description available."),
        'facts': CONSTELLATION_FACTS.get(constellation_name, "No additional facts available.")
    })


if __name__ == '__main__':
    # For production on Render
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
