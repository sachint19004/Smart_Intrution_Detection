import os

# ---------------- DATABASE CONFIGURATION ----------------
DB_HOST = "localhost"
DB_USER = "root"

DB_PASS = "YOUR_DB_PASSWORD_HERE" 
DB_NAME = "face_db"

# ---------------- NETWORK CONFIGURATION ----------------
# IP of the machine running the Receiver/Server code
SERVER_IP = "192.168.X.X"  # <--- REPLACE with your Laptop's LAN IP
SERVER_PORT = 5000
LISTEN_HOST = "0.0.0.0"    # Allows server to listen on all interfaces

# ---------------- FILE PATHS ----------------
KEY_FILE = "aes_secret.key"
INTRUDER_DIR = "Intruder_Log"
INTRUDER_XLS = "Intruder_Log.xlsx"
AUTH_XLS = "Authorized_Log.xlsx"

# ---------------- MODEL SETTINGS ----------------
YOLO_MODEL_FACE = "yolov8s-face.pt"   # For Server/Video Check
YOLO_MODEL_GENERIC = "yolov8n.pt"     # For Raspberry Pi Client
FACE_MATCH_THRESHOLD = 0.75           # Cosine similarity threshold
LOGGING_COOLDOWN = 60                 # Seconds before logging the same person again

# ---------------- CLIENT (PI) SETTINGS ----------------
FPS = 3
JPEG_QUALITY = 80
MIN_PERSON_CONF = 0.3