from ultralytics import YOLO

# load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# run inference
results = model("demo.jpg")  # returns a list of Results

# get the first result
r = results[0]

# loop over each detection
for box in r.boxes:
    cls_id = int(box.cls[0])      # class index
    conf = float(box.conf[0])     # confidence score
    label = r.names[cls_id]       # human-readable label
    print(f"{label}: {conf:.2f}")

r.save()