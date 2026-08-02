import easyocr
import cv2
from parse import extract
from trace_logging import log

def get_reader():
    return easyocr.Reader(["en"])
 
def ocr(image, reader,trace_path = "trace.jsonl"):
    results = reader.readtext(
    image,
    detail=1,
    paragraph=False,
    contrast_ths=0.1,
    adjust_contrast=0.7,
    mag_ratio=1.5,       
    text_threshold=0.6,
    low_text=0.3
)

    confidences = [conf for (_, _, conf) in results]
    filtered_count = sum(1 for c in confidences if c < 0.4)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    log(trace_path, "ocr", {
        "num_text_blocks": len(results),
        "filtered_below_threshold": filtered_count,
        "confidence_avg": avg_conf
    })

    fields = extract(results,image.shape[0])
    return fields

 
if __name__ == "__main__":
    reader = get_reader()
    image_path = input("Enter image path: ")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Could not load image")
    else:
        fields = ocr(image, reader)
        print(fields)
