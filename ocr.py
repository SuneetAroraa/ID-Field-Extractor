import easyocr
import cv2
from parse import extract

reader = easyocr.Reader(["en"])
#cleaned_crop = "/Users/suneetarora/Desktop/ ID Field Extractor/Cropped_output.jpg"
cleaned_crop = '/Users/suneetarora/Desktop/ ID Field Extractor/01_alb_id/images/HA/HA01_03.jpg'

def ocr(cleaned_crop):
    results = reader.readtext(cleaned_crop)
    fields = extract(results)
    print(fields)
    return fields

ocr(cleaned_crop)