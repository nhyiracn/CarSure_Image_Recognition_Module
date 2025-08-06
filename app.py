import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Load YOLOv8 model once at startup
try:
    model = YOLO("yolov8n.pt")
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    model = None

# Vehicle class IDs in COCO dataset
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck
VEHICLE_NAMES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class VehicleDetectionResult:
    def __init__(self, is_vehicle, detected_classes, confidence_scores, bounding_boxes=None, processed_image_path=None):
        self.is_vehicle = is_vehicle
        self.detected_classes = detected_classes
        self.confidence_scores = confidence_scores
        self.bounding_boxes = bounding_boxes or []
        self.processed_image_path = processed_image_path
        self.timestamp = datetime.now().isoformat()

def validate_image(image_path):
    """Validate image file and check if it's corrupted"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        return False

def preprocess_image(image_path, target_size=(640, 640)):
    """Preprocess image for better detection"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Resize while maintaining aspect ratio
        h, w = img.shape[:2]
        scale = min(target_size[0]/w, target_size[1]/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        resized = cv2.resize(img, (new_w, new_h))
        
        # Create padded image
        padded = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        y_offset = (target_size[1] - new_h) // 2
        x_offset = (target_size[0] - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return None

def detect_vehicles_enhanced(image_path, confidence_threshold=0.3, save_annotated=True):
    """Enhanced vehicle detection with detailed results"""
    if not model:
        return VehicleDetectionResult(False, [], [], None, None)
    
    try:
        # Validate image first
        if not validate_image(image_path):
            logger.error("Image validation failed")
            return VehicleDetectionResult(False, [], [], None, None)
        
        # Run inference
        results = model(image_path, conf=confidence_threshold)
        
        detected_vehicles = []
        confidence_scores = []
        bounding_boxes = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            # Process each detection
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id in VEHICLE_CLASS_IDS:
                    vehicle_name = VEHICLE_NAMES.get(cls_id, 'unknown_vehicle')
                    detected_vehicles.append(vehicle_name)
                    confidence_scores.append(conf)
                    
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bounding_boxes.append({
                        'class': vehicle_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        # Save annotated image if requested and vehicles detected
        processed_image_path = None
        if save_annotated and detected_vehicles:
            try:
                # Generate unique filename
                filename = f"annotated_{uuid.uuid4().hex}.jpg"
                processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
                
                # Save annotated image
                annotated_img = results[0].plot()
                cv2.imwrite(processed_image_path, annotated_img)
                
            except Exception as e:
                logger.error(f"Failed to save annotated image: {e}")
        
        is_vehicle = len(detected_vehicles) > 0
        
        return VehicleDetectionResult(
            is_vehicle=is_vehicle,
            detected_classes=detected_vehicles,
            confidence_scores=confidence_scores,
            bounding_boxes=bounding_boxes,
            processed_image_path=processed_image_path
        )
        
    except Exception as e:
        logger.error(f"Vehicle detection failed: {e}")
        return VehicleDetectionResult(False, [], [], None, None)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/api/detect-vehicle', methods=['POST'])
def api_detect_vehicle():
    """API endpoint for vehicle detection"""
    
    # Check if file was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Get optional parameters
        confidence_threshold = float(request.form.get('confidence', 0.3))
        vin_number = request.form.get('vin', '')
        stakeholder_id = request.form.get('stakeholder_id', '')
        
        # Detect vehicles
        result = detect_vehicles_enhanced(file_path, confidence_threshold)
        
        # Prepare response
        response_data = {
            'is_vehicle': result.is_vehicle,
            'detected_classes': result.detected_classes,
            'confidence_scores': result.confidence_scores,
            'bounding_boxes': result.bounding_boxes,
            'timestamp': result.timestamp,
            'vin_number': vin_number,
            'stakeholder_id': stakeholder_id
        }
        
        # Add processed image URL if available
        if result.processed_image_path:
            response_data['annotated_image_url'] = f"/processed/{os.path.basename(result.processed_image_path)}"
        
        # Log result for admin review system
        log_detection_result(file_path, result, vin_number, stakeholder_id)
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def log_detection_result(image_path, result, vin_number, stakeholder_id):
    """Log detection result for admin review (placeholder for database integration)"""
    log_entry = {
        'timestamp': result.timestamp,
        'image_path': image_path,
        'is_vehicle': result.is_vehicle,
        'detected_classes': result.detected_classes,
        'confidence_scores': result.confidence_scores,
        'vin_number': vin_number,
        'stakeholder_id': stakeholder_id,
        'status': 'approved' if result.is_vehicle else 'rejected'
    }
    
    # TODO: Save to database for admin review
    logger.info(f"Detection logged: {log_entry}")

@app.route('/processed/<filename>')
def processed_file(filename):
    """Serve processed/annotated images"""
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)