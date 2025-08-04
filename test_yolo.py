from vehicle_detection import is_vehicle
from ultralytics import YOLO


image_path = "test_motor.jpg"
contains_vehicle, classes = is_vehicle(image_path)

if contains_vehicle:
    print(f"✅ Vehicle(s) detected: {classes}")
else:
    print("🚫 No vehicle detected.")

model = YOLO("yolov8n.pt")
results = model(image_path)
for r in results:
    r.show()



