from ultralytics import YOLO

# Load YOLOv8n model once at import time
model = YOLO("yolov8n.pt")

VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck

def is_vehicle(image_path, confidence_threshold=0.3):
    results = model(image_path)
    detected_vehicles = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if conf >= confidence_threshold and cls_id in VEHICLE_CLASS_IDS:
                detected_vehicles.append(result.names[cls_id])

    return (len(detected_vehicles) > 0), detected_vehicles
