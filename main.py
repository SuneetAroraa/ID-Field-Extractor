from detection import detect_and_crop
from ocr import ocr,get_reader
import cv2
from ultralytics import YOLO

def main():
    reader = get_reader()
    model = YOLO("yolov8n.pt")
    image_path = input('Enter Image Path: ')
    cropped = detect_and_crop(model,image_path)
    if cropped is None:
         print("Detection Failed")
         return

    fields = ocr(cropped, reader)
    print(fields)

if __name__ == "__main__":
    main()