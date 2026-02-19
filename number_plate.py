import cv2
import os

# Haar Cascade is a pre-trained ML model for object detection
harcasade ="model/haarcascade_russian_plate_number.xml"
plate_cascade= cv2.CascadeClassifier(harcasade)

# Create folder if not exists
os.makedirs("plates", exist_ok=True)

cap = cv2.VideoCapture(0)
# 0 is  default laptop webcam, Creates a video stream object
cap.set(3, 640) #3- width
cap.set(4, 400) #4- height

min_area= 500 
# ignores very small detections (noise)
count=6


while True:
    success, img = cap.read()
    # Stops loop if camera fails
    if not success:
        break

    # Haar Cascade works better & faster on grayscale ,Removes color info
    img_gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates= plate_cascade.detectMultiScale(img_gray, 1.1, 4 )
    # Returns a list of rectangles (x, y, w, h) where plates are detected
    # 1.1 is scale factor (how much image size is reduced each time)
    # 4 → minimum neighbors (higher = fewer false positives)

    for(x,y,w,h) in plates:
        area = w*h
        # Prevents false detections on tiny objects
        if (area > min_area):
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2) #greenbox
            cv2.putText(img, "number plate",(x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
            img_roi = img[y: y+h, x: x+w]
            cv2.imshow("ROI", img_roi) #region of interest

    cv2.imshow("Result", img)

    key= cv2.waitKey(1) & 0xFF

    if key == ord('s') and img_roi is not None:
        cv2.imwrite(f"plates/scanned_img_{count}.jpg", img_roi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0, 255), 2)
        cv2.imshow("Results", img)

        print(f"[INFO] Plate saved: scanned_img_{count}.jpg")
        cv2.waitKey(300)
        count +=1

    if key== ord('q'):
        break
cap.release()          # Releases the webcam so other apps can use it
cv2.destroyAllWindows() # Closes all OpenCV windows