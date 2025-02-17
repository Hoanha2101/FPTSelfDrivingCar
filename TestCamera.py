import cv2

# Mở camera (0 là camera mặc định, có thể thay đổi nếu có nhiều camera)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()

while True:
    # Đọc khung hình từ camera
    ret, frame = cap.read()
    if not ret:
        print("Không thể nhận dữ liệu từ camera")
        break

    # Hiển thị khung hình
    cv2.imshow('Camera', frame)

    # Nhấn 'q' để thoát, 's' để chụp ảnh
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("captured_image.jpg", frame)
        print("Ảnh đã được lưu: captured_image.jpg")

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
