import easyocr
import cv2
from parse import extract

def get_reader():
    return easyocr.Reader(["en"])
 
def ocr(image, reader):
    results = reader.readtext(image)
    fields = extract(results)
    return fields
 
if __name__ == "__main__":
    reader = get_reader()
    image_path = input("Enter image path: ")
    fields = ocr(image_path, reader)
    print(fields)