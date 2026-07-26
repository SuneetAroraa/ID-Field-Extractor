from detection import detect_and_crop
#from ocr import ocr
import cv2
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # downloads automatically on first run
    image_path = "/Users/suneetarora/Desktop/ ID Field Extractor/01_alb_id/images/HA/HA01_03.jpg"
    #image_path = input('Enter Image Path: ')
    cropped = detect_and_crop(model,image_path)
    if cropped is not None:
        #ocr(cropped)
        cv2.imwrite("Cropped_output.jpg", cropped)
    else:
        print("Detection Failed")

if __name__ == "__main__":
    main()