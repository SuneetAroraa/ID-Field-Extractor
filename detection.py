from clean import clean_image, detectwarp_image
import cv2
from trace_logging import log

def detect_and_crop(model, image_path, trace_path="trace.jsonl"):
    results = model(image_path)
    image = cv2.imread(image_path)

    best_box = None
    highest_conf = -1

    for result in results:
        for box in result.boxes:
            conf = box.conf[0].item()
            if conf > highest_conf:
                highest_conf = conf
                best_box = box.xyxy[0].tolist()

    if best_box is not None:
        x1, y1, x2, y2 = best_box
        cropped = image[int(y1):int(y2), int(x1):int(x2)]
        grayscale = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        ocr_result = cv2.resize(grayscale, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("Cropped_output.jpg", clean_image(ocr_result))
        log(trace_path, "detection", {
            "method": "yolo",
            "box_found": True,
            "confidence": highest_conf,
            "box": best_box
        })
        return ocr_result

    warped = detectwarp_image(image_path)
    if warped is not None:
        cv2.imwrite("Cropped_output.jpg", clean_image(warped))
        log(trace_path, "detection", {
            "method": "warp_fallback",
            "confidence": None,
            "box": None
        })
        return warped

    log(trace_path, "detection", {
        "method": "none",
        "error": "no box detected and warp fallback failed",
        "confidence": None,
        "box": None
    })
    return None