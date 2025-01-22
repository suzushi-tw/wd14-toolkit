import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import math

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.2,  # Even lower threshold for anime
            min_tracking_confidence=0.2
        )
        
        # Keep your existing gestures and thresholds
        self.gestures = {
            'v_sign': [(8,6,5), (12,6,5)],
            'double_v': [(8,6,5), (12,6,5), (16,14,13), (20,18,17)],
            'pointing_at_viewer': [(8,6,5)],
            'hands_up': [(8,6,5), (12,10,9), (16,14,13), (20,18,17)],
            'rabbit_pose': [(8,6,5), (12,10,9)],
            'shushing': [(8,6,5), (4,3,2)],
            'wave': [(8,6,5), (12,10,9), (16,14,13), (20,18,17)],
            'peace': [(8,6,5), (12,10,9)]
        }

        self.gesture_thresholds = {
            'v_sign': 130,
            'double_v': 130,
            'pointing_at_viewer': 150,
            'hands_up': 140,
            'rabbit_pose': 130,
            'shushing': 140,
            'wave': 120,
            'peace': 130
        }

    def preprocess_image(self, image):
        # Convert to grayscale and back to help with feature detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Enhanced preprocessing pipeline
        # 1. Increase contrast
        lab = cv2.cvtColor(gray, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # 2. Sharpen
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        # 3. Bilateral filter to preserve edges while reducing noise
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # 4. Additional contrast adjustment
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=0)

        return enhanced

    def detect_hands(self, pil_image):
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Try multiple preprocessing variations
        preprocessed_versions = [
            image,  # Original
            self.preprocess_image(image),  # Enhanced
            cv2.convertScaleAbs(image, alpha=1.5, beta=0),  # Simple contrast boost
            cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)  # Grayscale
        ]
        
        for processed in preprocessed_versions:
            results = self.hands.process(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                break
        
        if results.multi_hand_landmarks:
            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS)
            
            annotated_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            
            info = []
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                gesture = self.detect_gesture(hand_landmarks)
                hand_type = "Right hand" if idx == 0 else "Left hand"  # Simplified hand type detection
                info.append(f"{hand_type}: {gesture}")
            
            return annotated_pil, "\n".join(info)
        
        return pil_image, "No hands detected"

    # Keep your existing calculate_angle and detect_gesture methods
    
    def __del__(self):
        self.hands.close()