import cv2
import numpy as np
import sqlite3
import os
import datetime
import urllib.request
import imutils
import time
from imutils.video import VideoStream
from PIL import Image, ImageDraw, ImageFont
import io

 
def check_opencv_face_support():
    try:
        if hasattr(cv2, 'face') and hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
            return True
        return False
    except:
        return False

 
class LBPHFaceRecognizer:
    def __init__(self):
        self.model = None
        self.labels = []
        self.label_map = {}
        
    def train(self, faces, labels):
        if hasattr(cv2, 'face'):
            self.model = cv2.face.LBPHFaceRecognizer_create()
            self.model.train(faces, np.array(labels))
        self.labels = labels
        self.label_map = {i: label for i, label in enumerate(labels)}
        
    def predict(self, face):
        if self.model is not None:
            label, confidence = self.model.predict(face)
            return label, confidence
        return -1, 100

 
def download_model(url, filename):
    if not os.path.exists(filename):
        print(f"[INFO] {filename} indiriliyor...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"[INFO] {filename} başarıyla indirildi!")
            return True
        except Exception as e:
            print(f"[HATA] {filename} indirilemedi: {str(e)}")
            return False
    return True

 
def init_database():
    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            registration_date TEXT NOT NULL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            FOREIGN KEY(person_id) REFERENCES persons(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            access_time TEXT NOT NULL,
            status TEXT NOT NULL,
            confidence REAL,
            FOREIGN KEY(person_id) REFERENCES persons(id)
        )
    ''')
    
    conn.commit()
    conn.close()

 
class FaceRecognizer:
    def __init__(self):
        self.opencv_face_supported = check_opencv_face_support()
        print(f"[INFO] OpenCV face modülü desteği: {'Var' if self.opencv_face_supported else 'Yok'}")
        
     
        self.detection_model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        self.detection_config_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        self.detection_model_file = "res10_300x300_ssd_iter_140000.caffemodel"
        self.detection_config_file = "deploy.prototxt"
        
        self.model_downloaded = True
        if not download_model(self.detection_model_url, self.detection_model_file):
            self.model_downloaded = False
        if not download_model(self.detection_config_url, self.detection_config_file):
            self.model_downloaded = False
        
        if self.model_downloaded:
            try:
                self.detection_net = cv2.dnn.readNetFromCaffe(
                    self.detection_config_file, 
                    self.detection_model_file
                )
                print("[INFO] DNN yüz algılama modeli yüklendi")
            except Exception as e:
                print(f"[HATA] DNN model yüklenemedi: {str(e)}")
                self.model_downloaded = False
       
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        
        self.recognizer = None
        self.lbph_recognizer = LBPHFaceRecognizer()
        self.lbph_trained = False
        
       
        self.FACE_MATCH_THRESHOLD = 0.7
        self.CONFIDENCE_THRESHOLD = 0.7
        
        
        self.known_faces = []
        self.load_known_faces()
    
    def load_known_faces(self):
        conn = sqlite3.connect('face_recognition.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT p.id, p.name, fe.embedding 
            FROM persons p
            JOIN face_embeddings fe ON p.id = fe.person_id
        ''')
        
        self.known_faces = []
        faces_for_lbph = []
        labels_for_lbph = []
        
        for row in cursor.fetchall():
            person_id, name, embedding_blob = row
            
          
            try:
                nparr = np.frombuffer(embedding_blob, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces_for_lbph.append(img)
                    labels_for_lbph.append(person_id)
            except:
                pass
       
        if faces_for_lbph and labels_for_lbph:
            try:
                self.lbph_recognizer.train(faces_for_lbph, labels_for_lbph)
                self.lbph_trained = True
                print(f"[INFO] LBPH modeli eğitildi, {len(faces_for_lbph)} yüz")
            except Exception as e:
                print(f"[HATA] LBPH eğitim hatası: {str(e)}")
                self.lbph_trained = False
        
        conn.close()
    
    def detect_faces(self, frame):
        if self.model_downloaded:
            return self.detect_faces_dnn(frame)
        else:
            return self.detect_faces_haar(frame)
    
    def detect_faces_dnn(self, frame):
        try:
            (h, w) = frame.shape[:2]
            resized_frame = cv2.resize(frame, (300, 300))
            blob = cv2.dnn.blobFromImage(resized_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            
            self.detection_net.setInput(blob)
            detections = self.detection_net.forward()
            
            faces = []
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < self.CONFIDENCE_THRESHOLD:
                    continue
                    
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w - 1, endX), min(h - 1, endY)
                
                face_roi = frame[startY:endY, startX:endX]
                
                if face_roi.size == 0:
                    continue
                    
                faces.append({
                    'roi': face_roi,
                    'coords': (startX, startY, endX, endY),
                    'confidence': confidence
                })
                
            return faces
        except Exception as e:
            print(f"[HATA] DNN algılama: {str(e)}")
            return self.detect_faces_haar(frame)
    
    def detect_faces_haar(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            detected_faces = []
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                detected_faces.append({
                    'roi': face_roi,
                    'coords': (x, y, x+w, y+h),
                    'confidence': 1.0
                })
                
            return detected_faces
        except Exception as e:
            print(f"[HATA] Haar algılama: {str(e)}")
            return []
    
    def recognize_face(self, face_roi):
        if self.lbph_trained:
            return self.recognize_face_lbph(face_roi)
        return None
    
    def recognize_face_lbph(self, face_roi):
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            label, confidence = self.lbph_recognizer.predict(gray)
            
            if confidence < 100:
                conn = sqlite3.connect('face_recognition.db')
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM persons WHERE id = ?", (label,))
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    name = result[0]
                    normalized_confidence = 1 - (confidence / 100)
                    return {
                        'id': label,
                        'name': name,
                        'confidence': normalized_confidence
                    }
        except Exception as e:
            print(f"[HATA] LBPH tanıma: {str(e)}")
        
        return None
    
    def add_new_face(self, name, face_roi):
        conn = sqlite3.connect('face_recognition.db')
        cursor = conn.cursor()
        
        registration_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO persons (name, registration_date) VALUES (?, ?)",
            (name, registration_date)
        )
        person_id = cursor.lastrowid
        
     
        try:
            _, img_encoded = cv2.imencode('.png', face_roi)
            embedding_blob = img_encoded.tobytes()
            
            cursor.execute(
                "INSERT INTO face_embeddings (person_id, embedding) VALUES (?, ?)",
                (person_id, embedding_blob)
            )
        except Exception as e:
            print(f"[HATA] Görüntü kodlama: {str(e)}")
        
        conn.commit()
        conn.close()
        
       
        self.load_known_faces()
        
        return person_id
    
    def log_access(self, person_id, status, confidence=None):
        conn = sqlite3.connect('face_recognition.db')
        cursor = conn.cursor()
        
        access_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO access_logs (person_id, access_time, status, confidence) VALUES (?, ?, ?, ?)",
            (person_id, access_time, status, confidence)
        )
        
        conn.commit()
        conn.close()

 
def draw_face_info(frame, face_info, coords, confidence):
    startX, startY, endX, endY = coords
    
    color = (0, 255, 0) if face_info else (0, 0, 255)
    
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    
    label = "Unknown"
    if face_info:
        label = f"{face_info['name']} ({confidence*100:.1f}%)"
    
 
    cv2.rectangle(frame, (startX, startY - 30), 
                 (endX, startY), color, cv2.FILLED)
    
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    draw.text((startX + 5, startY - 25), label, font=font, fill=(0, 0, 0))
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def display_access_logs(frame, logs):
    height, width = frame.shape[:2]
    
    log_panel = np.zeros((height, 350, 3), dtype=np.uint8)
    
    cv2.putText(log_panel, "Erişim Kayıtları", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    y_offset = 70
    for i, log in enumerate(logs[:5]):
        time_str = log[1].split(' ')[1][:5]
        name = log[0] if log[0] else "Bilinmeyen"
        status = "ONAY" if log[2] == "granted" else "RED"
        color = (0, 255, 0) if status == "ONAY" else (0, 0, 255)
        
        log_text = f"{time_str} - {name} - {status}"
        cv2.putText(log_panel, log_text, (10, y_offset + i*40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    return np.hstack((frame, log_panel))

 
def main():
    init_database()
    
    face_recognizer = FaceRecognizer()
    
    print("[INFO] Kameraya bağlanılıyor...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    
    add_new_person = False
    new_person_name = ""
    
    while True:
        frame = vs.read()
        if frame is None:
            print("[HATA] Kameradan görüntü alınamıyor")
            break
        
        frame = imutils.resize(frame, width=1000)
        display_frame = frame.copy()
        height, width = frame.shape[:2]
        
       
        conn = sqlite3.connect('face_recognition.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT p.name, a.access_time, a.status 
            FROM access_logs a
            LEFT JOIN persons p ON a.person_id = p.id
            ORDER BY a.access_time DESC
            LIMIT 5
        ''')
        access_logs = cursor.fetchall()
        conn.close()
        
         
        faces = face_recognizer.detect_faces(frame)
        
        for face in faces:
            face_roi = face['roi']
            coords = face['coords']
            confidence = face['confidence']
            
            face_info = face_recognizer.recognize_face(face_roi)
            
            if face_info:
                face_recognizer.log_access(
                    face_info['id'], "granted", face_info['confidence'])
                display_confidence = face_info['confidence']
            else:
                face_recognizer.log_access(None, "denied")
                display_confidence = confidence
            
            display_frame = draw_face_info(
                display_frame, face_info, coords, display_confidence
            )
        
        
        display_frame = display_access_logs(display_frame, access_logs)
        
       
        mode_text = "Yeni Kişi Ekleme Modu" if add_new_person else "Tanıma Modu"
        cv2.putText(display_frame, mode_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        if add_new_person:
            cv2.putText(display_frame, f"İsim: {new_person_name}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
         
        cv2.putText(display_frame, "n: Yeni kii ekle", (20, height-60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display_frame, "s: Kaydet", (20, height-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(display_frame, "q: Cikis", (200, height-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow("Profesyonel Yuz Tanima Sistemi", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            add_new_person = True
            new_person_name = input("Yeni kişinin adını girin: ")
        elif key == ord('s') and add_new_person and faces:
            face_roi = faces[0]['roi']
            face_recognizer.add_new_face(new_person_name, face_roi)
            print(f"[BİLGİ] {new_person_name} başarıyla eklendi!")
            add_new_person = False
    
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()