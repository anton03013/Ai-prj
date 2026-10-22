"""Real-time MediaPipe gesture recognition. Q/ESC quits; S saves a screenshot."""
from pathlib import Path
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "gesture_recognizer.task"
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12), (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

def draw_hand(frame, landmarks, handedness="", gesture=""):
    h, w = frame.shape[:2]
    pts = [(max(0,min(w-1,int(p.x*w))), max(0,min(h-1,int(p.y*h)))) for p in landmarks]
    for a,b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (70,220,70), 2, cv2.LINE_AA)
    for i,(x,y) in enumerate(pts):
        cv2.circle(frame,(x,y),5,(40,90,255),-1,cv2.LINE_AA)
        cv2.circle(frame,(x,y),7,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame,str(i),(x+5,y-5),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1,cv2.LINE_AA)
    if pts and (handedness or gesture):
        label = " | ".join(v for v in (handedness,gesture) if v)
        x,y=pts[0]
        cv2.putText(frame,label,(max(5,x),max(25,y-15)),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2,cv2.LINE_AA)

def main():
    if not MODEL.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL}\nTải gesture_recognizer.task vào thư mục models/.")
    options = vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release(); recognizer.close()
        raise RuntimeError("Không mở được webcam. Kiểm tra quyền Camera, đóng ứng dụng đang dùng webcam hoặc thử camera index 1.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    last_ts = 0
    prev = time.perf_counter(); fps = 0.0
    print("Đang nhận diện. Nhấn Q/ESC để thoát, S để lưu ảnh.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Không đọc được khung hình từ webcam."); break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = max(time.monotonic_ns() // 1_000_000, last_ts + 1)
            last_ts = ts
            result = recognizer.recognize_for_video(mp_image, ts)
            for i, landmarks in enumerate(result.hand_landmarks):
                hand = result.handedness[i][0].category_name if i < len(result.handedness) and result.handedness[i] else ""
                gesture = "No gesture"
                if i < len(result.gestures) and result.gestures[i]:
                    g = result.gestures[i][0]
                    gesture = f"{g.category_name} {g.score:.2f}"
                draw_hand(frame, landmarks, hand, gesture)
            now=time.perf_counter(); dt=now-prev
            if dt>0:
                current=1.0/dt; fps=current if fps==0 else .9*fps+.1*current
            prev=now
            cv2.putText(frame,f"FPS: {fps:.1f}",(15,30),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(frame,"Q/ESC: quit | S: save screenshot",(15,frame.shape[0]-15),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
            cv2.imshow("MediaPipe Gesture Recognizer - Webcam",frame)
            key=cv2.waitKey(1)&0xFF
            if key in (ord('q'),27): break
            if key==ord('s'):
                out=ROOT/"captures"; out.mkdir(exist_ok=True)
                path=out/f"gesture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                if cv2.imwrite(str(path),frame): print(f"Đã lưu: {path}")
    finally:
        camera.release(); cv2.destroyAllWindows(); recognizer.close()

if __name__ == "__main__":
    main()
