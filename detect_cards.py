from ultralytics import YOLO
import sys

def main():
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python detect_cards.py <image_path>")
        return

    image_path = sys.argv[1]

    # Load trained model
    model = YOLO("runs/detect/train/weights/best.pt")

    # Run inference
    results = model(image_path)

    # Get first result
    r = results[0]

    print(f"\nDetected cards in: {image_path}\n")

    # Print detections
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = r.names[cls_id]

        print(f"{label}: {conf:.2f}")

    # Save output image
    r.save()

    print("\nPrediction image saved to runs/detect/predict/")

if __name__ == "__main__":
    main()