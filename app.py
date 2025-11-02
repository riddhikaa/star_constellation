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
CORS(app)  # Enable CORS for frontend communication

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'best.pt'  # Place your best.pt file here

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

# Constellation names mapping (from your dataset)
CONSTELLATION_NAMES = {
    0: 'Aquila',
    1: 'Bootes',
    2: 'Canis Major',
    3: 'Canis Minor',
    4: 'Cassiopeia',
    5: 'Cygnus',
    6: 'Gemini',
    7: 'Leo',
    8: 'Lyra',
    9: 'Moon',
    10: 'Orion',
    11: 'Pleiades',
    12: 'Sagittarius',
    13: 'Scorpius',
    14: 'Taurus',
    15: 'Ursa Major'
}

# Detailed constellation descriptions
CONSTELLATION_DESCRIPTIONS = {
    "Aquila": "Aquila, the Eagle, is a constellation on the celestial equator. Its brightest star, Altair, is one of the vertices of the Summer Triangle asterism. The constellation represents the eagle that carried Zeus's thunderbolts in Greek mythology.",
    
    "Bootes": "Bootes is a constellation in the northern sky, containing the bright star Arcturus. Its name comes from the Greek word meaning 'herdsman' or 'plowman'. Arcturus is the fourth-brightest star in the night sky and the brightest in the northern celestial hemisphere.",
    
    "Canis Major": "Canis Major, the Greater Dog, is a constellation in the southern celestial hemisphere. It contains Sirius, the brightest star in the night sky. In Greek mythology, it represents one of the dogs following Orion, the hunter.",
    
    "Canis Minor": "Canis Minor, the Lesser Dog, is a small constellation in the northern celestial hemisphere. Its brightest star, Procyon, forms one vertex of the Winter Triangle asterism. It represents the smaller of Orion's two hunting dogs.",
    
    "Cassiopeia": "Cassiopeia is a constellation in the northern sky, named after the vain queen in Greek mythology. It is easily recognizable due to its distinctive 'W' or 'M' shape formed by five bright stars. The constellation is circumpolar in northern latitudes, meaning it never sets below the horizon.",
    
    "Cygnus": "Cygnus, the Swan, is a northern constellation lying on the plane of the Milky Way. Its brightest star, Deneb, forms one vertex of the Summer Triangle. The constellation's most recognizable feature is the asterism known as the Northern Cross.",
    
    "Gemini": "Gemini, the Twins, is a constellation of the zodiac lying between Taurus and Cancer. Its brightest stars are Castor and Pollux, named after the twin brothers in Greek and Roman mythology. The constellation is most visible in the winter sky.",
    
    "Leo": "Leo, the Lion, is a constellation of the zodiac lying between Cancer and Virgo. Its brightest star, Regulus, is one of the brightest stars in the night sky. The constellation is easily identifiable by a distinctive sickle-shaped asterism representing the lion's head and chest.",
    
    "Lyra": "Lyra is a small constellation in the northern sky, representing the lyre of Orpheus in Greek mythology. Its brightest star, Vega, is one of the brightest stars visible from Earth and forms one vertex of the Summer Triangle. Lyra also contains the famous Ring Nebula.",
    
    "Moon": "The Moon is Earth's only natural satellite and the fifth largest satellite in the Solar System. It is the brightest object in the night sky after the Sun. The Moon's gravitational influence produces Earth's tides and slightly lengthens Earth's day.",
    
    "Orion": "Orion is one of the most prominent and recognizable constellations in the night sky. Named after a hunter in Greek mythology, it contains some of the brightest stars including Betelgeuse and Rigel. The three stars forming Orion's Belt are among the most recognizable patterns in the sky.",
    
    "Pleiades": "The Pleiades, also known as the Seven Sisters, is an open star cluster in the constellation Taurus. It is one of the nearest star clusters to Earth and the most obvious to the naked eye. In Greek mythology, the Pleiades were the seven daughters of Atlas and Pleione.",
    
    "Sagittarius": "Sagittarius is a zodiac constellation in the southern celestial hemisphere, traditionally represented as a centaur drawing a bow. It is one of the brightest constellations and contains the center of our Milky Way galaxy. The constellation is best viewed in summer.",
    
    "Scorpius": "Scorpius, the Scorpion, is a zodiac constellation lying between Libra and Sagittarius. Its brightest star, Antares, is a red supergiant and one of the largest stars visible to the naked eye. The constellation's distinctive J-shaped pattern of bright stars represents a scorpion.",
    
    "Taurus": "Taurus, the Bull, is a large and prominent constellation in the northern hemisphere's winter sky. It contains the bright star Aldebaran, which represents the bull's eye, and two famous star clusters: the Pleiades and the Hyades. In Greek mythology, the constellation represents Zeus in the form of a bull.",
    
    "Ursa Major": "Ursa Major, the Great Bear, is a constellation in the northern sky. Its most recognizable feature is the Big Dipper asterism, which is one of the most familiar patterns in the sky. The Big Dipper's pointer stars lead to Polaris, the North Star, making it useful for navigation."
}

# Mythological and astronomical facts
CONSTELLATION_FACTS = {
    "Aquila": "• Contains the bright star Altair\n• Part of the Summer Triangle\n• Visible from June to November\n• Ancient constellation recognized by Ptolemy",
    
    "Bootes": "• Home to the brightest star in the northern hemisphere: Arcturus\n• Contains several notable galaxies\n• Visible from March to September\n• Name means 'plowman' or 'ox-driver'",
    
    "Canis Major": "• Contains Sirius, the brightest star in Earth's night sky\n• Follows Orion across the sky\n• Best visible in winter\n• Home to several star clusters",
    
    "Canis Minor": "• One of Orion's two hunting dogs\n• Contains the bright star Procyon\n• Small constellation with few notable stars\n• Forms part of the Winter Triangle",
    
    "Cassiopeia": "• Circumpolar constellation (never sets)\n• Distinctive 'W' or 'M' shape\n• Named after an Ethiopian queen\n• Contains several notable nebulae",
    
    "Cygnus": "• Also known as the Northern Cross\n• Located in the Milky Way\n• Contains Deneb, one of the most luminous stars\n• Rich in deep-sky objects",
    
    "Gemini": "• Third zodiac constellation\n• Represents the twins Castor and Pollux\n• Best viewed in January and February\n• Contains several meteor shower radiants",
    
    "Leo": "• Fifth zodiac constellation\n• Contains the bright star Regulus\n• Home to many galaxies\n• Best visible in spring",
    
    "Lyra": "• Small but prominent constellation\n• Contains Vega, fifth brightest star\n• Home to the Ring Nebula (M57)\n• Represents Orpheus's lyre",
    
    "Moon": "• Average distance: 384,400 km from Earth\n• Diameter: 3,474 km\n• Synchronous rotation (same side always faces Earth)\n• Only celestial body humans have visited",
    
    "Orion": "• One of the most recognizable constellations\n• Contains Betelgeuse and Rigel\n• Home to the Orion Nebula (M42)\n• Best viewed in winter months",
    
    "Pleiades": "• Open star cluster, not a constellation\n• Also called M45 or Seven Sisters\n• About 444 light-years from Earth\n• Contains over 1,000 confirmed stars",
    
    "Sagittarius": "• Points toward the galactic center\n• Ninth zodiac constellation\n• Rich in nebulae and star clusters\n• Best viewed in summer",
    
    "Scorpius": "• One of the zodiac constellations\n• Contains the red supergiant Antares\n• Located in a rich star field\n• Best viewed in summer",
    
    "Taurus": "• Second zodiac constellation\n• Contains two famous star clusters\n• Home to the Crab Nebula (M1)\n• Best viewed in winter",
    
    "Ursa Major": "• Third largest constellation\n• Contains the Big Dipper asterism\n• Pointer stars lead to Polaris\n• Visible year-round in northern latitudes"
}


@app.route('/', methods=['GET'])
def api_check():
    """Basic API health check"""
    return jsonify({
        'status': 'ok',
        'message': 'API is running smoothly!'
    }), 200


@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if the API is running and model is loaded"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'total_classes': len(CONSTELLATION_NAMES),
        'constellations': list(CONSTELLATION_NAMES.values())
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
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
        
        # Perform inference
        results = model.predict(
            input_path,
            save=True,
            project=RESULTS_FOLDER,
            name=unique_id,
            exist_ok=True,
            conf=0.25,  # Confidence threshold
            iou=0.45    # IoU threshold for NMS
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
                'bbox': box.xyxy[0].tolist()  # Bounding box coordinates [x1, y1, x2, y2]
            })
            
            # Count constellations
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
        
        # Sort by count (most detected first)
        found_constellations.sort(key=lambda x: x['count'], reverse=True)
        
        # Cleanup uploaded file (keep results for debugging if needed)
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
        return jsonify({'error': str(e)}), 500

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
    # Find constellation by name (case-insensitive)
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
    # For local development
    app.run(debug=True, host='0.0.0.0', port=5000)
