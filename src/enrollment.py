import os
import cv2
import torch
import numpy as np
import pickle as pkl
import mysql.connector
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from config import *
# ---------------- CONFIGURATION ----------------
KEY_FILE = "aes_secret.key"
DB_HOST = "localhost"         # Or use '192.168.X.X'
DB_USER = "root"
DB_PASS = "YOUR_DB_PASSWORD"  
DB_NAME = "face_db"

# ---------------- AES KEY MANAGEMENT ----------------
if not os.path.exists(KEY_FILE):
    key = get_random_bytes(32)  # AES-256 key
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    print(f"[INFO] Generated new AES-256 key: {KEY_FILE}")
else:
    with open(KEY_FILE, "rb") as f:
        key = f.read()
    print(f"[INFO] Loaded AES-256 key from file: {KEY_FILE}")

# ---------------- DATABASE CONNECTION ----------------
try:
    db = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME
    )
    cursor = db.cursor()
    print("[INFO] Database connected successfully.")
except mysql.connector.Error as err:
    print(f"[ERROR] Database connection failed: {err}")
    exit()

# ---------------- LOAD MODELS ----------------
print("[INFO] Loading models...")
yolo_model = YOLO("yolov8n-face.pt")
facenet = InceptionResnetV1(pretrained="vggface2").eval()

# ---------------- MAIN EXECUTION ----------------
def main():
    # 1. Initialize Video Capture
    cap = cv2.VideoCapture(1)  # Change index to 0 or 1 depending on your camera
    
    if not cap.isOpened():
        print("[ERROR] Could not open video device.")
        return

    # 2. Get User Details
    name = input("Enter authorized person's name: ").strip()
    if not name:
        print("[ERROR] Name cannot be empty.")
        cap.release()
        return

    print(f"[INFO] Starting enrollment for '{name}'. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            break

        # 3. Face Detection (YOLO)
        results = yolo_model(frame, verbose=False)

        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)

                # Extract face ROI
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                # 4. Preprocess for FaceNet
                try:
                    face_rgb = cv2.resize(face, (160, 160))
                    face_rgb = cv2.cvtColor(face_rgb, cv2.COLOR_BGR2RGB)
                    face_tensor = torch.tensor(face_rgb / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

                    # 5. Generate Embedding
                    embedding = facenet(face_tensor).detach().numpy().astype(np.float32).flatten()
                    
                    # Normalize embedding
                    norm_val = np.linalg.norm(embedding)
                    if norm_val > 0:
                        embedding = embedding / norm_val

                    # Serialize embedding
                    emb_pickle = pkl.dumps(embedding, protocol=pkl.HIGHEST_PROTOCOL)

                    # 6. Encryption (AES-GCM)
                    cipher = AES.new(key, AES.MODE_GCM)
                    ciphertext, tag = cipher.encrypt_and_digest(emb_pickle)
                    nonce = cipher.nonce

                    # 7. Store in Database
                    sql = """
                        INSERT INTO authorized_faces (name, embedding_cipher, aes_nonce, aes_tag)
                        VALUES (%s, %s, %s, %s)
                    """
                    cursor.execute(sql, (name, ciphertext, nonce, tag))
                    db.commit()

                    # 8. Visual Feedback
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Saved: {name}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    print(f"[INFO] Saved encrypted embedding for {name}")

                except Exception as e:
                    print(f"[ERROR] Processing face failed: {e}")

        cv2.imshow("Authorized Face Enrollment", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    cursor.close()
    db.close()
    print("[INFO] Enrollment finished.")

if __name__ == "__main__":
    main()