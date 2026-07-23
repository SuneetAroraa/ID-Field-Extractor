from ultralytics import YOLO
import cv2
from clean import contour_based_crop

def detect_and_crop(model,image_path):
    results = model(image_path)
    image = cv2.imread(image_path)

    best_box = None
    highest_conf = -1

    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf[0].item()   
            if conf > highest_conf:
                highest_conf = conf
                best_box = box.xyxy[0].tolist() 

    if best_box is not None:
            x1, y1, x2, y2 = best_box
            cropped = image[int(y1):int(y2), int(x1):int(x2)]
            return cropped
    
    return contour_based_crop(image_path)


def main():
    model = YOLO("yolov8n.pt")  # downloads automatically on first run
    image_path = "/Users/suneetarora/Desktop/ ID Field Extractor/01_alb_id/images/HA/HA01_04.tif"
    cropped = detect_and_crop(model,image_path)
    if cropped is not None:
        cv2.imwrite("Cropped_output.jpg", cropped)
    else:
        print("Detection Failed")

if __name__ == "__main__":
    main()