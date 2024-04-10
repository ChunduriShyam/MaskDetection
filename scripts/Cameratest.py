import cv2

cap = cv2.VideoCapture(0)  # 0 is usually the default camera

if not cap.isOpened():
    print("Cannot open camera")
else:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Camera Test', frame)
        cv2.waitKey(0)  # Wait for a key press to close
    else:
        print("Can't receive frame (stream end?). Exiting ...")

cap.release()
cv2.destroyAllWindows()
