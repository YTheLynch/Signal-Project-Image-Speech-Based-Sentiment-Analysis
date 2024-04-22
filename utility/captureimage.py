import cv2
import numpy as np

counter = 0

def main():
    cam_port = 0
    cam = cv2.VideoCapture(cam_port)
    

    def capture_image():
        global counter
        filename = f"captured_image{counter}.jpg"
        counter += 1
        cv2.imwrite(f"Captures/{filename}", image)
        print('Image Captured')

    while True:
        result, image = cam.read()
        image = cv2.flip(image, 1)
        imagetext = np.zeros(image.shape, np.uint8)
        imagetext[:, :] = image
        text = "Press 'c' to capture"
        cv2.putText(imagetext, text, (210, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        
        cv2.imshow("Image Capture", imagetext)
        
        
        key = cv2.waitKey(1)
        if key == ord('c'):
            capture_image()
        
        if cv2.getWindowProperty("Image Capture", cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv2.destroyAllWindows()
    return counter


if _name_ == '_main_':
    main()