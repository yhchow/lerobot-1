import cv2

cap = cv2.VideoCapture("http://192.168.68.107:8080/video_feed")
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read frame")
        continue
    cv2.imwrite("test_frame.jpg", frame)
    print("✅ Frame saved to test_frame.jpg")
    break


cap.release()
cv2.destroyAllWindows()
