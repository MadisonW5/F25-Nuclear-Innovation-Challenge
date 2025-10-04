from ultralytics import YOLO

# Load your trained model
model = YOLO(r"C:\Users\snail\OneDrive\Documents\GitHub\F25-Nuclear-Innovation-Challenge\runs\detect\train\weights\best.pt")

# Run detection on one test image
results = model.predict(
    source=r"C:\Users\snail\OneDrive\Documents\GitHub\F25-Nuclear-Innovation-Challenge\dataset photos\img_0000.jpg",
    show=True,
    conf=0.2
)