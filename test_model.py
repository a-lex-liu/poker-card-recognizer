from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model("demo.jpg")
results[0].save()
results[0].show()
