from DocScanner_Deskew import main
import cv2

#image_path = "/Users/suneetarora/Desktop/ ID Field Extractor/01_alb_id/images/HA/HA01_15.tif"
def contour_based_crop(image_path):
    return main(image_path)

# img = contour_based_crop(image_path)
# cv2.imwrite('Testimage.jpg',img)
# cv2.waitKey(0)