from DocScanner_Deskew import contour_approximation, perspective_transform, detect_skew, skew_correction, apply_adaptive_threshold
import cv2

def detectwarp_image(image_path):
    image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

    doc_contour = contour_approximation(image)

    if doc_contour is not None:
        warped = perspective_transform(image, doc_contour)
        upscaled = cv2.resize(warped, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        return upscaled

    else:
        return None
    
def clean_image(image):
    angle = detect_skew(image)
    deskewed = skew_correction(image, angle)
    thresholded = apply_adaptive_threshold(deskewed)
    return thresholded

