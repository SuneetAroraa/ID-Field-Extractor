import cv2
import numpy as np

def apply_adaptive_threshold(img):
    if len(img.shape) == 3: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img,(5,5),0)
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 15)
    return th

def detect_skew(img):
    edge = cv2.Canny(img,50,150)
    lines = cv2.HoughLines(edge,1,np.pi/180,200)

    angles = []

    if lines is not None:
        for rho,theta in lines[:,0]:
            angle = (theta * 180 / np.pi) - 90
            if abs(angle) < 45:
                angles.append(angle)

    if len(angles) == 0:
        return 0.0
    
    skew_angle = np.median(angles)
    return skew_angle if abs(skew_angle) > 0.5 else 0.0
    
def skew_correction(img,angle):
    h,w = img.shape[:2]
    center = (w//2,h//2)

    m = cv2.getRotationMatrix2D(center,angle,1.0)
    corrected = cv2.warpAffine(img,m,(w,h),borderMode=cv2.BORDER_REPLICATE)
    return corrected

def output_size(doc_contour):
    tl,tr,br,bl = doc_contour
    width_top = np.linalg.norm(tr-tl)
    width_bottom = np.linalg.norm(br-bl)
    width = int(max(width_top,width_bottom))

    height_left = np.linalg.norm(bl-tl)
    height_right = np.linalg.norm(br - tr)
    height = int(max(height_left, height_right))
 
    return width, height

def order_points(doc_contour):
    pts = doc_contour.reshape(4, 2).astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]   
    ordered[2] = pts[np.argmax(s)]   
 
    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)] 
    ordered[3] = pts[np.argmax(diff)] 
 
    return ordered

def perspective_transform(image,doc_contour):
    pts = order_points(doc_contour)
    width,height = output_size(pts)
    new_points = np.array([[0,0],[width,0],[width,height],[0,height]],dtype="float32")
    matrix = cv2.getPerspectiveTransform(pts,new_points)
    pers_transform = cv2.warpPerspective(image,matrix,(width,height))
    return pers_transform

def contour_approximation(image):
    blur = cv2.GaussianBlur(image,(5,5),0)
    edged = cv2.Canny(blur,50,150)

    contours, heirarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    doc_contour = None
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx) == 4:
            doc_contour = approx
            break

    return doc_contour
