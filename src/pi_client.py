import cv2
import socket
import pickle
import time
import queue
import threading
import numpy as np
from ultralytics import YOLO
from config import *
# ---------------- CONFIGURATION ----------------
SERVER_IP = "192.168.X.X"  # <- REPLACE with your server/laptop's LAN IP
SERVER_PORT = 5000
FPS = 3
JPEG_QUALITY = 80
MIN_PERSON_CONF = 0.3

# ---------------- INITIALIZATION ----------------
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

send_q = queue.Queue(maxsize=3)
stop_flag = threading.Event()

# ---------------- THREADING LOGIC ----------------
def sender_thread(sock, q):
    """
    Reads data from the queue and sends it over the socket.
    """
    try:
        while not stop_flag.is_set():
            data = q.get()
            if data is None:
                break
            try:
                # Send data length followed by actual data
                sock.sendall(len(data).to_bytes(4, "big") + data)
            except Exception as e:
                print(f"[ERROR] Send failed: {e}")
                break
            finally:
                q.task_done()
    finally:
        try:
            sock.close()
        except:
            pass

def create_connection_with_retry():
    """
    Attempts to connect to the server with exponential backoff.
    """
    backoff = 1
    while not stop_flag.is_set():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            print(f"[INFO] Trying to connect to {SERVER_IP}:{SERVER_PORT} ...")
            s.connect((SERVER_IP, SERVER_PORT))
            s.settimeout(None)  # Remove timeout after connected
            print("[INFO] Connected to server")
            return s
        except Exception as e:
            print(f"[WARN] Connect failed: {e}. Retrying in {backoff}s")
            try:
                s.close()
            except:
                pass
            time.sleep(backoff)
            backoff = min(30, backoff * 2)
    
    raise RuntimeError("Stopped before connection established")

# ---------------- MAIN LOOP ----------------
def main():
    # 1. Establish Connection
    try:
        sock = create_connection_with_retry()
    except RuntimeError:
        print("Exiting because stop flag set")
        return

    # 2. Start Sender Thread
    th = threading.Thread(target=sender_thread, args=(sock, send_q), daemon=True)
    th.start()

    prev = 0
    print("[INFO] Client started. Press Ctrl+C to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # FPS Limiting
            now = time.time()
            if now - prev < 1.0 / FPS:
                time.sleep(0.01)
                continue
            prev = now

            # 3. Object Detection (Person Check)
            # Resize for faster inference
            small = cv2.resize(frame, (640, 360))
            results = model(small, verbose=False)
            
            person_found = False
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    # Class 0 is 'person' in COCO dataset
                    if cls == 0 and conf >= MIN_PERSON_CONF:
                        person_found = True
                        break
                if person_found:
                    break

            # Skip sending if no person detected
            if not person_found:
                continue

            # 4. Encoding & Queuing
            # Encode frame to JPEG
            _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            data = pickle.dumps(buffer)

            try:
                send_q.put_nowait(data)
            except queue.Full:
                # Drop frame if queue is full (network congestion)
                pass

    except KeyboardInterrupt:
        print("\n[INFO] Stopping client...")
    
    finally:
        stop_flag.set()
        try:
            send_q.put(None)  # Signal thread to stop
        except:
            pass
        th.join(timeout=2)
        cap.release()
        print("[INFO] Client stopped.")

if __name__ == "__main__":
    main()