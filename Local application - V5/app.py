from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import numpy as np
import os
import librosa
from resemblyzer import VoiceEncoder
import io
# import face_recognition # Removed dlib-dependent face_recognition
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import cv2 # OpenCV is now the primary face processing library
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)
CORS(app)

# Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'authuser',
    'password': 'securepass123',
    'database': 'biometric_auth'
}
class Config:
    SAMPLE_RATE = 16000
    VOICE_THRESHOLD = 0.75
    EMBEDDING_DIR = "monitorexamloginsys/embeddings"
    STATIC_DIR = "static"
    
    # Face recognition thresholds for LBPH: lower is better (distance)
    FACE_THRESHOLD = 80.0 # Adjusted to be a distance threshold for LBPH. Tune this!
    FINGERPRINT_THRESHOLD = 200  # Adjusted to match app.py's threshold of 200 matches

    # Path to OpenCV's built-in Haar Cascade XML for frontal face detection
    HAARCASCADE_FRONTALFACE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        print(f"Database connection error: {e}")
        return None
# Ensure directories exist
os.makedirs(Config.EMBEDDING_DIR, exist_ok=True)
os.makedirs(Config.STATIC_DIR, exist_ok=True)

# Abstract base class for biometric authentication
class BiometricAuthenticator(ABC):
    def __init__(self, auth_type: str, threshold: float):
        self.auth_type = auth_type
        self.threshold = threshold
        self.data_dir = os.path.join(Config.EMBEDDING_DIR, auth_type)
        os.makedirs(self.data_dir, exist_ok=True)
    
    @abstractmethod
    def extract_features(self, data: Any) -> Any:
        """Extract features from raw biometric data"""
        pass
    
    @abstractmethod
    def preprocess_data(self, data: Any) -> Any:
        """Preprocess the biometric data"""
        pass

    def delete_features(self, user_id: str) -> bool:
        """Delete biometric data (face or voice) for the user from the database, and remove user if all are null"""
        conn = None
        cursor = None
        try:
            conn = db_connection()
            if not conn:
                raise ValueError("Could not connect to database")

            cursor = conn.cursor()
            column_name = {
                'face': 'face_data',
                'voice': 'voice_data',
                'fingerprint': 'fingerprint_data' # Include fingerprint here for combined check
            }.get(self.auth_type)

            if not column_name:
                raise ValueError(f"Unsupported auth type for deletion: {self.auth_type}")

            update_query = f"""
                UPDATE users_biometrics
                SET {column_name} = NULL
                WHERE username = %s
            """
            cursor.execute(update_query, (user_id,))
            conn.commit()

            check_query = """
                SELECT voice_data, face_data, fingerprint_data
                FROM users_biometrics
                WHERE username = %s
            """
            cursor.execute(check_query, (user_id,))
            result = cursor.fetchone()

            if result and all(field is None for field in result):
                print(f"All biometrics null for '{user_id}'. Deleting user entry.")
                delete_query = "DELETE FROM users_biometrics WHERE username = %s"
                cursor.execute(delete_query, (user_id,))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting {self.auth_type} data for user '{user_id}': {e}")
            return False
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None and conn.is_connected():
                conn.close()
        
    def calculate_similarity(self, features1: Any, features2: Any) -> float:
        """Calculate similarity between two feature vectors"""
        # This method needs to be overridden for specific biometric types if their comparison is not simple cosine similarity
        if isinstance(features1, np.ndarray) and isinstance(features2, np.ndarray):
            a = np.array(features1)
            b = np.array(features2)
            # Add a small epsilon to avoid division by zero for norm calculation
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0 # Or raise an error, depending on desired behavior
            return float(np.dot(a, b) / (norm_a * norm_b))
        else:
            raise NotImplementedError(f"calculate_similarity not implemented for raw feature types of {self.auth_type}")
    
    def save_features(self, user_id: str, features: np.ndarray) -> bool:
        """Save user features as a blob in a database"""
        cursor = None
        conn = None
        try:
            conn=db_connection()
            if conn is None:
                raise ValueError("Database connection failed")
            
            cursor = conn.cursor()

            # Check if username exists, if so, update; otherwise, insert
            check_query = "SELECT username FROM users_biometrics WHERE username = %s"
            cursor.execute(check_query, (user_id,))
            user_exists = cursor.fetchone()

            # Convert NumPy array to bytes for storage
            embedding_blob = features.tobytes()

            if user_exists:
                query = f"UPDATE users_biometrics SET {self.auth_type}_data = %s WHERE username = %s"
                cursor.execute(query, (embedding_blob, user_id))
            else:
                query = f"INSERT INTO users_biometrics (username, {self.auth_type}_data) VALUES (%s, %s)"
                cursor.execute(query, (user_id, embedding_blob))

            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving features for {self.auth_type}: {e}")
            return False
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None and conn.is_connected():
                conn.close()
    
    def load_features(self, user_id: str) -> Optional[np.ndarray]:
        """Load user features from file"""
        cursor = None
        conn = None
        try:
            conn = db_connection()
            if conn is None:
                raise ValueError("Database connection failed")
            
            cursor = conn.cursor()
            query = f"SELECT {self.auth_type}_data FROM users_biometrics WHERE username = %s"
            cursor.execute(query, (user_id,))
            row = cursor.fetchone()

            if row is None or row[0] is None:
                print(f"No {self.auth_type} features found for user '{user_id}'")
                return None
            
            (row_data,) = row

            if not isinstance(row_data, (bytes, bytearray)):
                raise ValueError("Expected BLOB data in bytes format")
            
            # The dtype must match how it was saved (np.float32 for embeddings)
            if self.auth_type == "voice":
                embedding = np.frombuffer(row_data, dtype=np.float32)
            else: # For face (LBPH is different) or fingerprint (ORB descriptors)
                # This needs to be handled by the specific authenticator's load_features method
                # as it might not be a simple numpy array for comparison.
                # For LBPH, the model itself is "the features"
                embedding = row_data # Pass raw bytes to the subclass to handle
            return embedding
        
        except Exception as e:
            print(f"Error loading {self.auth_type} features: {e}")
            return None
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None and conn.is_connected():
                conn.close()
    
    def register_user(self, user_id: str, data: Any) -> Dict[str, Any]:
        """Register a new user"""
        print(f"Registering user '{user_id}' for {self.auth_type}")
        
        try:
            # Process data and extract features
            processed_data = self.preprocess_data(data)
            
            # Special handling for fingerprint due to tuple return from preprocess_data
            if isinstance(processed_data, tuple):
                image_data, extension = processed_data
                features = self.extract_features(image_data)
                features['extension'] = extension # Keep extension for saving image
            else:
                features = self.extract_features(processed_data)
            
            if features is None or (isinstance(features, np.ndarray) and features.size == 0) or \
               (isinstance(features, dict) and (features.get('descriptors') is None or len(features['descriptors']) == 0)):
                return {
                    "success": False,
                    "message": f"No valid {self.auth_type} features could be extracted. Please ensure the biometric data is clear and contains a detectable feature.",
                    "auth_type": self.auth_type
                }

            # Save features
            if self.save_features(user_id, features):
                return {
                    "success": True,
                    "message": f"Successfully registered '{user_id}' for {self.auth_type}.",
                    "auth_type": self.auth_type
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to save {self.auth_type} data for '{user_id}'. It might already be registered or a database error occurred.",
                    "auth_type": self.auth_type
                }
        except Exception as e:
            print(f"Error in register_user: {str(e)}")
            return {
                "success": False,
                "message": f"Error processing {self.auth_type} data: {str(e)}",
                "auth_type": self.auth_type
            }
    
    def authenticate_user(self, user_id: str, data: Any) -> Dict[str, Any]:
        """Authenticate an existing user"""
        print(f"Authenticating user '{user_id}' for {self.auth_type}")
        
        try:
            # Process data and extract features for the current attempt
            processed_data = self.preprocess_data(data)
            
            if isinstance(processed_data, tuple):
                image_data, _ = processed_data
                current_features = self.extract_features(image_data)
            else:
                current_features = self.extract_features(processed_data)
            
            if current_features is None or (isinstance(current_features, np.ndarray) and current_features.size == 0) or \
               (isinstance(current_features, dict) and (current_features.get('descriptors') is None or len(current_features['descriptors']) == 0)):
                return {
                    "success": False,
                    "message": f"No valid {self.auth_type} features could be extracted from the provided data.",
                    "auth_type": self.auth_type,
                    "similarity": 0.0
                }

            # Load stored features (which might trigger specific logic in subclasses for face/fingerprint)
            stored_features = self.load_features(user_id)
            if stored_features is None:
                return {
                    "success": False,
                    "message": f"No registered {self.auth_type} data found for user '{user_id}'.",
                    "auth_type": self.auth_type,
                    "similarity": 0.0
                }
            
            # Calculate similarity based on the specific authenticator's logic
            # This will call the overridden calculate_similarity if present in subclasses
            similarity = self.calculate_similarity(current_features, stored_features)
            
            # Determine success based on threshold (comparison logic differs by type)
            is_successful = False
            if self.auth_type == "face":
                # For LBPH, similarity is a distance, so lower is better
                is_successful = similarity < self.threshold
                print(f"Face (LBPH) Distance: {similarity:.2f}, Threshold: {self.threshold:.2f}, Match: {is_successful}")
            elif self.auth_type == "voice":
                # For Resemblyzer, similarity is cosine similarity, higher is better
                is_successful = similarity >= self.threshold
                print(f"Voice Similarity: {similarity:.2f}, Threshold: {self.threshold:.2f}, Match: {is_successful}")
            elif self.auth_type == "fingerprint":
                # For ORB, similarity is number of matches, higher is better
                is_successful = similarity >= self.threshold
                print(f"Fingerprint Matches: {similarity:.0f}, Threshold: {self.threshold:.0f}, Match: {is_successful}")

            if is_successful:
                return {
                    "success": True,
                    "message": f"Authentication successful for '{user_id}' using {self.auth_type}.",
                    "auth_type": self.auth_type,
                    "similarity": round(similarity, 3)
                }
            else:
                return {
                    "success": False,
                    "message": f"Authentication failed for '{user_id}' using {self.auth_type}.",
                    "auth_type": self.auth_type,
                    "similarity": round(similarity, 3)
                }
        except Exception as e:
            print(f"Error in authenticate_user: {str(e)}")
            return {
                "success": False,
                "message": f"Error processing {self.auth_type} authentication: {str(e)}",
                "auth_type": self.auth_type,
                "similarity": 0.0
            }

# Voice Authentication Implementation
class VoiceAuthenticator(BiometricAuthenticator):
    def __init__(self):
        super().__init__("voice", Config.VOICE_THRESHOLD)
        self.encoder = VoiceEncoder()
    
    def preprocess_data(self, data: bytes) -> np.ndarray:
        """Preprocess audio data"""
        print("Preprocessing voice data")
        y, _ = librosa.load(io.BytesIO(data), sr=Config.SAMPLE_RATE)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        return y_trimmed / np.max(np.abs(y_trimmed)) if np.max(np.abs(y_trimmed)) > 0 else y_trimmed
    
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract voice features using Resemblyzer"""
        print("Extracting voice features")
        return self.encoder.embed_utterance(data)

    # calculate_similarity is inherited from the base class for voice (cosine similarity)


# Updated Fingerprint Authentication Implementation
class FingerprintAuthenticator(BiometricAuthenticator):
    def __init__(self):
        super().__init__("fingerprint", Config.FINGERPRINT_THRESHOLD)
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.allowed_extensions = {'.jpg', '.jpeg', '.png'}  # Define allowed file extensions
    
    def get_file_extension(self, filename: str) -> str:
        """Extract the file extension from the filename"""
        return os.path.splitext(filename)[1].lower()
    
    def preprocess_data(self, fingerprint_file: Any) -> np.ndarray:
        """Preprocess fingerprint image data by reading the uploaded file"""
        try:
            # Validate file extension
            extension = self.get_file_extension(fingerprint_file.filename)
            if extension not in self.allowed_extensions:
                raise ValueError(f"Unsupported file format: {extension}. Use JPG or PNG.")
            
            print(f"Processing file: {fingerprint_file.filename}")
            # Read the file directly into a numpy array
            nparr = np.frombuffer(fingerprint_file.read(), np.uint8)
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                raise ValueError("Could not decode fingerprint image")
            
            return image, extension # Return image data and extension
        except Exception as e:
            print(f"Preprocessing error: {str(e)}")
            raise ValueError(f"Error preprocessing fingerprint image: {str(e)}")
    
    def extract_features(self, image_data: np.ndarray) -> dict:
        """Extract fingerprint features using ORB detector"""
        try:
            print("Extracting fingerprint features")
            # Detect keypoints and compute descriptors
            keypoints, descriptors = self.orb.detectAndCompute(image_data, None)
            
            if descriptors is None or len(descriptors) == 0:
                # If no descriptors, it means no discernible features were found
                # Return an empty array or None to indicate failure
                return {'keypoints': [], 'descriptors': np.array([]), 'image': image_data}
            
            return {
                'keypoints': keypoints,
                'descriptors': descriptors,
                'image': image_data  # Store the image for saving
            }
        except Exception as e:
            print(f"Error extracting fingerprint features: {str(e)}")
            raise ValueError(f"Error extracting fingerprint features: {str(e)}")
    
    def calculate_similarity(self, current_features: dict, stored_features: dict) -> float:
        """Calculate similarity between two fingerprints using ORB matching"""
        try:
            print("Calculating fingerprint similarity")
            des1 = current_features['descriptors']
            des2 = stored_features['descriptors']
            
            if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
                print("One or both descriptor sets are empty. Similarity 0.0")
                return 0.0
            
            # Match descriptors
            matches = self.bf.match(des1, des2)
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Return the number of matches as the similarity score
            return float(len(matches))
        except Exception as e:
            print(f"Error calculating fingerprint similarity: {e}")
            return 0.0
    
    def save_features(self, user_id: str, features: dict) -> bool:
        """Save the fingerprint image in the database"""
        cursor = None
        conn = None
        try:
            # Check if username exists, if so, update; otherwise, insert
            conn = db_connection()
            if not conn:
                raise ValueError("Could not establish DB connection")

            cursor = conn.cursor()
            
            check_query = "SELECT username FROM users_biometrics WHERE username = %s"
            cursor.execute(check_query, (user_id,))
            user_exists = cursor.fetchone()

            # Encode the image to bytes for storage
            success, buffer = cv2.imencode('.png', features['image'])
            if not success:
                raise ValueError("Failed to encode fingerprint image")
            image_bytes = buffer.tobytes()

            if user_exists:
                query = """
                    UPDATE users_biometrics
                    SET fingerprint_data = %s
                    WHERE username = %s
                """
                cursor.execute(query, (image_bytes, user_id))
            else:
                query = """
                    INSERT INTO users_biometrics (username, fingerprint_data)
                    VALUES (%s, %s)
                """
                cursor.execute(query, (user_id, image_bytes))
            
            conn.commit()
            return True

        except Exception as e:
            print(f"Error saving fingerprint image to database: {e}")
            return False

        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None and conn.is_connected():
                conn.close()

    def load_features(self, user_id: str) -> Optional[dict]:
        """Load the fingerprint image from DB and extract features"""
        cursor = None
        conn = None
        try:
            conn = db_connection()
            if not conn:
                raise ValueError("Could not establish DB connection")

            cursor = conn.cursor()
            query = "SELECT fingerprint_data FROM users_biometrics WHERE username = %s"
            cursor.execute(query, (user_id,))
            row = cursor.fetchone()

            if row is None or row[0] is None:
                print(f"No {self.auth_type} features found for user '{user_id}'")
                return None
            
            (row_data,) = row
            image_data = np.frombuffer(row_data, np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print("Failed to decode fingerprint image from DB")
                return None

            features = self.extract_features(image)
            return features

        except Exception as e:
            print(f"Error loading fingerprint image from database: {e}")
            return None

        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None and conn.is_connected():
                conn.close()


# Face Authentication Implementation (using pure OpenCV: Haar Cascades + LBPHFaceRecognizer)
class FaceAuthenticator(BiometricAuthenticator):
    def __init__(self):
        super().__init__("face", Config.FACE_THRESHOLD)
        try:
            self.face_cascade = cv2.CascadeClassifier(Config.HAARCASCADE_FRONTALFACE)
            if self.face_cascade.empty():
                raise IOError(f"Could not load Haar Cascade classifier from {Config.HAARCASCADE_FRONTALFACE}")
            
            self.lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()
            # We'll manage known faces for LBPH training within this class
            self.known_face_images: List[np.ndarray] = []
            self.known_face_labels: List[int] = []
            self.label_to_user_id: Dict[int, str] = {}
            self.user_id_to_label: Dict[str, int] = {}
            self.next_label = 0
            
            # Load existing face data for training at startup
            self._load_all_known_faces_from_db_and_retrain()

            print("OpenCV Haar Cascade and LBPHFaceRecognizer initialized.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to initialize plain OpenCV face components. Error: {e}")
            raise RuntimeError(f"Failed to load face components: {e}")
    
    def _get_face_roi(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detects a face in the image and returns the ROI (Region of Interest) as grayscale."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using Haar Cascade
        faces = self.face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            return None # No face detected

        # For simplicity, take the first detected face (or you could take the largest)
        x, y, w, h = faces[0]
        face_roi = gray_image[y:y+h, x:x+w]
        
        # Resize to a common size for LBPH
        face_roi_resized = cv2.resize(face_roi, (100, 100)) # Common size
        return face_roi_resized

    def preprocess_data(self, image_bytes: bytes) -> np.ndarray:
        """Decode image bytes to an OpenCV image."""
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image from bytes. Is it a valid image format?")
        return image
    
    def extract_features(self, image_data: np.ndarray) -> Optional[np.ndarray]:
        """Extract face ROI. For LBPH, the ROI itself is what's used for training/prediction."""
        face_roi = self._get_face_roi(image_data)
        if face_roi is None:
            print("No face detected in the provided image.")
            return np.array([]) # Return empty array to indicate no face
        return face_roi

    def save_features(self, user_id: str, face_roi: np.ndarray) -> bool:
        """Save the face ROI (image) to the database for training LBPH later."""
        conn = None
        cursor = None
        try:
            conn = db_connection()
            if not conn:
                raise ValueError("Database connection failed")
            
            cursor = conn.cursor()

            # Check if the user already has face data. If so, overwrite it or append (depending on policy).
            # For simplicity, we'll store one face image per user in the DB.
            # In a real system, you might store multiple for better training or re-enroll if new image is significantly different.
            check_query = "SELECT face_data FROM users_biometrics WHERE username = %s"
            cursor.execute(check_query, (user_id,))
            user_has_face_data = cursor.fetchone()

            # Encode the face ROI to bytes (e.g., PNG format)
            success, buffer = cv2.imencode('.png', face_roi)
            if not success:
                raise ValueError("Failed to encode face ROI image")
            face_image_bytes = buffer.tobytes()

            if user_has_face_data and user_has_face_data[0] is not None:
                # If existing data, update it
                query = "UPDATE users_biometrics SET face_data = %s WHERE username = %s"
                cursor.execute(query, (face_image_bytes, user_id))
            else:
                # If no existing face data or user, insert (or update if other biometric exists)
                insert_or_update_query = """
                    INSERT INTO users_biometrics (username, face_data) VALUES (%s, %s)
                    ON DUPLICATE KEY UPDATE face_data = %s
                """
                cursor.execute(insert_or_update_query, (user_id, face_image_bytes, face_image_bytes))

            conn.commit()
            print(f"Face image saved for {user_id}. Retraining LBPH recognizer.")
            
            # After saving, we need to retrain the LBPH recognizer with all known data
            self._load_all_known_faces_from_db_and_retrain()
            return True
        except Exception as e:
            print(f"Error saving face features: {e}")
            return False
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None and conn.is_connected():
                conn.close()

    def load_features(self, user_id: str) -> Optional[np.ndarray]:
        """
        For LBPH, loading "features" means getting the stored image (ROI) from DB
        and potentially just confirming its existence. The actual recognition
        happens by predicting against the trained model.
        This method is mainly for checking if a user has face data registered.
        """
        conn = None
        cursor = None
        try:
            conn = db_connection()
            if not conn:
                raise ValueError("Database connection failed")
            
            cursor = conn.cursor()
            query = "SELECT face_data FROM users_biometrics WHERE username = %s"
            cursor.execute(query, (user_id,))
            row = cursor.fetchone()

            if row is None or row[0] is None:
                return None # No face data found

            # Decode the stored image bytes back to an OpenCV image
            face_image_bytes = row[0]
            np_arr = np.frombuffer(face_image_bytes, np.uint8)
            face_image_roi = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            
            if face_image_roi is None:
                print(f"Could not decode stored face image for user {user_id}")
                return None

            # Resize to the standard size expected by LBPH
            face_image_roi = cv2.resize(face_image_roi, (100, 100))
            return face_image_roi # Return the image data itself

        except Exception as e:
            print(f"Error loading face features from DB for {user_id}: {e}")
            return None
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None and conn.is_connected():
                conn.close()

    def _load_all_known_faces_from_db_and_retrain(self):
        """Loads all registered face images from the database and retrains the LBPH recognizer."""
        self.known_face_images = []
        self.known_face_labels = []
        self.label_to_user_id = {}
        self.user_id_to_label = {}
        self.next_label = 0

        conn = None
        cursor = None
        try:
            conn = db_connection()
            if not conn:
                print("Could not connect to database for retraining LBPH.")
                return

            cursor = conn.cursor()
            query = "SELECT username, face_data FROM users_biometrics WHERE face_data IS NOT NULL"
            cursor.execute(query)
            rows = cursor.fetchall()

            if not rows:
                print("No face data in DB to train LBPH recognizer.")
                # Clear existing model if no data
                self.lbph_recognizer = cv2.face.LBPHFaceRecognizer_create() 
                return

            for username, face_data_bytes in rows:
                np_arr = np.frombuffer(face_data_bytes, np.uint8)
                face_image_roi = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
                
                if face_image_roi is None:
                    print(f"Warning: Could not decode stored face image for user {username}. Skipping.")
                    continue

                face_image_roi = cv2.resize(face_image_roi, (100, 100)) # Ensure consistent size

                if username not in self.user_id_to_label:
                    self.user_id_to_label[username] = self.next_label
                    self.label_to_user_id[self.next_label] = username
                    self.next_label += 1
                
                self.known_face_images.append(face_image_roi)
                self.known_face_labels.append(self.user_id_to_label[username])
            
            if len(self.known_face_images) > 0 and len(set(self.known_face_labels)) > 0:
                self.lbph_recognizer.train(self.known_face_images, np.array(self.known_face_labels))
                print(f"LBPHFaceRecognizer retrained with {len(self.known_face_images)} images for {len(set(self.known_face_labels))} users.")
            else:
                print("Not enough unique faces/labels to train LBPHFaceRecognizer after loading from DB.")
                # Clear recognizer if training failed
                self.lbph_recognizer = cv2.face.LBPHFaceRecognizer_create() 


        except Exception as e:
            print(f"Error during LBPH retraining from DB: {e}")
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None and conn.is_connected():
                conn.close()

    def calculate_similarity(self, current_face_roi: np.ndarray, stored_face_image_dummy: Any) -> float:
        """
        For LBPH, "similarity" is actually a distance. Lower values indicate better matches.
        We don't directly compare two ROIs here. Instead, we predict against the trained model.
        The stored_face_image_dummy parameter is just to match the signature; it's not directly used
        for the comparison itself, as the comparison is against the internal lbph_recognizer state.
        """
        if current_face_roi is None or current_face_roi.size == 0:
            return float('inf') # Indicate a very bad match if no face is detected
        
        try:
            # Predict returns (label, confidence/distance)
            predicted_label, confidence = self.lbph_recognizer.predict(current_face_roi)
            
            # Check if the predicted label corresponds to a known user_id
            predicted_user_id = self.label_to_user_id.get(predicted_label)
            
            # If the predicted user_id matches the one we're trying to authenticate, return its confidence.
            # Otherwise, return a very high distance to signify no match.
            # We will handle the actual thresholding in the parent's authenticate_user
            # The returned value is the distance. Lower is better.
            return confidence if predicted_user_id is not None else float('inf')

        except Exception as e:
            print(f"Error during LBPH prediction: {e}")
            return float('inf') # Return a very high distance on error

    def delete_features(self, user_id: str) -> bool:
        """Delete face data for the user from the database and retrain LBPH."""
        conn = None
        cursor = None
        try:
            conn = db_connection()
            if not conn:
                raise ValueError("Could not connect to database")

            cursor = conn.cursor()
            
            # Step 1: Nullify face_data
            update_query = """
                UPDATE users_biometrics
                SET face_data = NULL
                WHERE username = %s
            """
            cursor.execute(update_query, (user_id,))
            conn.commit()

            # Step 2: Retrain LBPH recognizer after deletion
            self._load_all_known_faces_from_db_and_retrain()

            # Step 3: Check if all biometrics are now NULL and delete user entry if so
            check_query = """
                SELECT voice_data, face_data, fingerprint_data
                FROM users_biometrics
                WHERE username = %s
            """
            cursor.execute(check_query, (user_id,))
            result = cursor.fetchone()

            if result and all(field is None for field in result):
                print(f"All biometrics null for '{user_id}'. Deleting user entry.")
                delete_query = "DELETE FROM users_biometrics WHERE username = %s"
                cursor.execute(delete_query, (user_id,))
                conn.commit()

            return True

        except Exception as e:
            print(f"Error deleting face data for user '{user_id}': {e}")
            return False

        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None and conn.is_connected():
                conn.close()


# Biometric Manager
class BiometricManager:
    def __init__(self):
        self.authenticators = {
            "voice": VoiceAuthenticator(),
            "face": FaceAuthenticator(),
            "fingerprint": FingerprintAuthenticator()
        }
    
    def get_authenticator(self, auth_type: str) -> Optional[BiometricAuthenticator]:
        return self.authenticators.get(auth_type)
    
    def get_available_methods(self) -> list:
        """Return list of available authentication methods"""
        available = []
        for method, authenticator in self.authenticators.items():
            try:
                # All methods are generally "available" if their initializers don't fail.
                # The check for registered data is done during login/get_user_registered_methods.
                available.append(method)
            except Exception:
                # If an authenticator failed to initialize, it won't be truly "available"
                pass 
        return available

# Initialize biometric manager
biometric_manager = BiometricManager()

# Helper functions
def decode_base64(b64_string: str) -> bytes:
    """Decode base64 string to bytes"""
    try:
        # Add padding if necessary
        missing_padding = len(b64_string) % 4
        if missing_padding:
            b64_string += '=' * (4 - missing_padding)
        return base64.b64decode(b64_string)
    except Exception:
        raise ValueError("Invalid base64 string")

def validate_request_data(data: dict, required_fields: list) -> tuple:
    """Validate request data"""
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    return True, ""

# API Routes
@app.route('/')
def serve_static():
    """Serve static files"""
    print("Serving static file: index.html")
    return send_from_directory(Config.STATIC_DIR, 'index.html')

@app.route('/static/<path:path>')
def serve_static_files(path):
    print(f"Serving static file: {path}")
    return send_from_directory(Config.STATIC_DIR, path)

@app.route('/api/methods', methods=['GET'])
def get_available_methods_api():
    """Get available authentication methods"""
    print("Handling /api/methods request")
    return jsonify({
        "available_methods": biometric_manager.get_available_methods(),
        "all_methods": list(biometric_manager.authenticators.keys())
    })

@app.route('/api/register', methods=['POST'])
def register():
    """Register user with biometric data"""
    print("Received /api/register request")
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Map frontend field names to backend names
        user_id = data.get('username') or data.get('user_id')
        auth_type = data.get('auth_method') or data.get('auth_type')
        
        raw_data = None
        # Get the appropriate data based on auth_type
        if auth_type == 'voice':
            data_b64 = data.get('voice_data') or data.get('data_b64')
            if not data_b64:
                return jsonify({
                    "success": False,
                    "message": "Missing voice biometric data"
                }), 400
            raw_data = decode_base64(data_b64)
        elif auth_type == 'face':
            data_b64 = data.get('face_data') or data.get('data_b64')
            if not data_b64:
                return jsonify({
                    "success": False,
                    "message": "Missing face biometric data"
                }), 400
            raw_data = decode_base64(data_b64)
        elif auth_type == 'fingerprint':
            if 'fingerprint' not in request.files:
                return jsonify({
                    "success": False,
                    "message": "Missing fingerprint file"
                }), 400
            fingerprint_file = request.files['fingerprint']
            if not fingerprint_file.filename:
                return jsonify({
                    "success": False,
                    "message": "No fingerprint file selected"
                }), 400
            # Extension validation is now handled inside FingerprintAuthenticator's preprocess_data
            raw_data = fingerprint_file
        else:
            return jsonify({
                "success": False,
                "message": "No biometric data provided or unsupported type"
            }), 400
        
        # Validate required fields
        if not all([user_id, auth_type]):
            missing = []
            if not user_id: missing.append('username/user_id')
            if not auth_type: missing.append('auth_method/auth_type')
            return jsonify({
                "success": False,
                "message": f"Missing required fields: {', '.join(missing)}"
            }), 400
        
        # Get authenticator
        authenticator = biometric_manager.get_authenticator(auth_type)
        if not authenticator:
            return jsonify({
                "success": False,
                "message": f"Unsupported authentication type: {auth_type}"
            }), 400
        
        # Register user
        result = authenticator.register_user(user_id, raw_data)
        
        print(f"Register response: {result}")
        status_code = 200 if result["success"] else 400
        return jsonify(result), status_code
        
    except ValueError as ve:
        print(f"Validation error in /api/register: {str(ve)}")
        return jsonify({
            "success": False,
            "message": f"Invalid request data: {str(ve)}"
        }), 400
    except Exception as e:
        print(f"Error in /api/register: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}"
        }), 500

@app.route('/api/login', methods=['POST'])
def login():
    """Authenticate user with biometric data"""
    print("Received /api/login request")
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Map frontend field names to backend names
        user_id = data.get('username') or data.get('user_id')
        auth_type = data.get('auth_method') or data.get('auth_type')
        
        print(f"Login attempt: {user_id} using {auth_type}")
        
        raw_data = None
        # Get the appropriate data
        if auth_type == 'voice':
            data_b64 = data.get('voice_data') or data.get('data_b64')
            if not data_b64:
                return jsonify({
                    "success": False,
                    "message": "Missing voice biometric data"
                }), 400
            raw_data = decode_base64(data_b64)
        elif auth_type == 'face':
            data_b64 = data.get('face_data') or data.get('data_b64')
            if not data_b64:
                return jsonify({
                    "success": False,
                    "message": "Missing face biometric data"
                }), 400
            raw_data = decode_base64(data_b64)
        elif auth_type == 'fingerprint':
            if 'fingerprint' not in request.files:
                return jsonify({
                    "success": False,
                    "message": "Missing fingerprint file"
                }), 400
            fingerprint_file = request.files['fingerprint']
            if not fingerprint_file.filename:
                return jsonify({
                    "success": False,
                    "message": "No fingerprint file selected"
                }), 400
            # Extension validation handled within FingerprintAuthenticator
            raw_data = fingerprint_file
        else:
            return jsonify({
                "success": False,
                "message": "No biometric data provided or unsupported type"
            }), 400
        
        # Validate required fields
        if not all([user_id, auth_type]):
            missing = []
            if not user_id: missing.append('username/user_id')
            if not auth_type: missing.append('auth_method/auth_type')
            return jsonify({
                "success": False,
                "message": f"Missing required fields: {', '.join(missing)}"
            }), 400
        
        # Get authenticator
        authenticator = biometric_manager.get_authenticator(auth_type)
        if not authenticator:
            return jsonify({
                "success": False,
                "message": f"Unsupported authentication type: {auth_type}"
            }), 400
        
        # Authenticate user
        result = authenticator.authenticate_user(user_id, raw_data)
        
        print(f"Login response: {result}")
        status_code = 200 if result["success"] else 401
        return jsonify(result), status_code
        
    except ValueError as ve:
        print(f"Validation error in /api/login: {str(ve)}")
        return jsonify({
            "success": False,
            "message": f"Invalid request data: {str(ve)}"
        }), 400
    except Exception as e:
        print(f"Error in /api/login: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Server error: {str(e)}"
        }), 500

@app.route('/api/user/<user_id>/methods', methods=['GET'])
def get_user_registered_methods(user_id):
    """Get authentication methods registered for a specific user"""
    print(f"Received /api/user/{user_id}/methods request")
    registered_methods = []
    
    conn = None
    cursor = None
    try:
        conn = db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True) # Get results as dictionaries
            query = "SELECT voice_data, face_data, fingerprint_data FROM users_biometrics WHERE username = %s"
            cursor.execute(query, (user_id,))
            result = cursor.fetchone()

            if result:
                if result['voice_data'] is not None:
                    registered_methods.append('voice')
                if result['face_data'] is not None:
                    registered_methods.append('face')
                if result['fingerprint_data'] is not None:
                    registered_methods.append('fingerprint')
    except Exception as e:
        print(f"Error fetching registered methods for user {user_id}: {e}")
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()

    return jsonify({
        "user_id": user_id,
        "registered_methods": registered_methods
    })

@app.route('/api/user/<user_id>/delete', methods=['DELETE'])
def delete_user_data(user_id):
    """Delete all biometric data for a user"""
    print(f"Received /api/user/{user_id}/delete request")
    deleted_methods = []
    errors = []

    for method_name, authenticator in biometric_manager.authenticators.items():
        try:
            success = authenticator.delete_features(user_id)
            if success:
                deleted_methods.append(method_name)
            else:
                # Note: delete_features might return False if data wasn't found, which isn't an "error" necessarily.
                # Adjusting this logic to be more precise based on what delete_features returns.
                # For now, it's fine as the authenticator's internal print will indicate if data wasn't found.
                errors.append(f"Failed to delete or data not found for {method_name} for user '{user_id}'")
        except Exception as e:
            errors.append(f"Error deleting {method_name} data: {str(e)}")

    # Check if any data was actually deleted
    if not deleted_methods and errors:
        return jsonify({
            "success": False,
            "message": "Failed to delete any biometric data or data not found.",
            "deleted_methods": [],
            "errors": errors
        }), 404 # Or 500 if it was a server-side error
    elif errors:
         return jsonify({
            "success": False,
            "message": "Some data could not be deleted",
            "deleted_methods": deleted_methods,
            "errors": errors
        }), 500

    return jsonify({
        "success": True,
        "message": f"All biometric data deleted for user '{user_id}'",
        "deleted_methods": deleted_methods
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    print("404 error occurred")
    return jsonify({"success": False, "message": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    print("500 error occurred")
    import traceback
    traceback.print_exc() # Print full traceback for debugging
    return jsonify({"success": False, "message": "Internal server error"}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Available authentication methods:", biometric_manager.get_available_methods())
    app.run(debug=True, host='0.0.0.0', port=5000)