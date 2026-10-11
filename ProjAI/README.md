# Nhận diện cử chỉ tay (MediaPipe)

Ứng dụng nhận diện cử chỉ tay thời gian thực qua webcam, vẽ khung xương bàn tay và hiển thị tên cử chỉ.

## Yêu cầu

- Python 3.10+
- Webcam

## Cài đặt

```bash
cd ProjAI
pip install -r requirements.txt
```

## Model

Tải file `gesture_recognizer.task` và đặt vào thư mục `models/`:

https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task

## Chạy

```bash
py main.py
```

Trên Windows có thể dùng `python main.py` nếu `py` chưa có trong PATH.

## Phím tắt

| Phím | Chức năng |
|------|-----------|
| Q / Esc | Thoát |
| S | Lưu ảnh chụp vào `captures/` |

