import mediapipe as mp
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Dict, Any, List

class AnimeHandDetector:
    def __init__(self, min_detection_confidence=0.3):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2, 
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )

    def _calculate_finger_state(self, hand_landmarks) -> List[bool]:
        """Calculate whether each finger is extended"""
        
        # Get landmark coordinates
        points = []
        for landmark in hand_landmarks.landmark:
            points.append([landmark.x, landmark.y, landmark.z])
        points = np.array(points)
        
        # Define finger indices
        finger_indices = [
            [8, 7, 6],  # Index finger
            [12, 11, 10],  # Middle finger
            [16, 15, 14],  # Ring finger
            [20, 19, 18]  # Pinky
        ]

        # Check if fingers are extended
        fingers = []
        
        # Thumb (special case)
        thumb_tip = points[4]
        thumb_base = points[2]
        if thumb_tip[0] < thumb_base[0]:  # For right hand
            fingers.append(True)
        else:
            fingers.append(False)
            
        # Other fingers
        for finger in finger_indices:
            if points[finger[0]][1] < points[finger[1]][1]:
                fingers.append(True)
            else:
                fingers.append(False)
                
        return fingers

    def _determine_gesture(self, fingers: List[bool]) -> Tuple[str, float]:
        """Determine gesture based on finger states"""
        
        gestures = {
            (True, True, True, True, True): ("open_palm", 0.9),
            (False, False, False, False, False): ("fist", 0.9),
            (False, True, True, False, False): ("peace", 0.8),
            (True, True, False, False, False): ("pointer", 0.8),
            (True, False, False, False, True): ("rock_on", 0.8),
            (True, True, True, False, False): ("three", 0.8),
            (True, True, True, True, False): ("four", 0.8),
            (False, True, False, False, False): ("one", 0.8),
            (True, False, False, False, False): ("thumbs_up", 0.8),
        }
        
        finger_state = tuple(fingers)
        return gestures.get(finger_state, ("unknown", 0.5))

    def detect_gesture(self, image: Image.Image, debug: bool = False) -> Tuple[Dict[str, Any], Dict[str, float], Image.Image]:
        """
        Detect hand gestures in the given image
        """
        if image is None:
            return {"error": "No image provided"}, {"confidence": 0.0}, None
            
        # Convert PIL image to RGB numpy array
        image_np = np.array(image.convert('RGB'))
        
        # Process the image
        results = self.hands.process(image_np)
        
        if not results.multi_hand_landmarks:
            return {"gesture": "no_hands_detected"}, {"confidence": 1.0}, image
            
        gestures = []
        confidences = {}
        
        # Create debug image
        debug_image = image_np.copy()
        
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get finger states
            fingers = self._calculate_finger_state(hand_landmarks)
            
            # Determine gesture
            gesture, confidence = self._determine_gesture(fingers)
            
            gestures.append(gesture)
            confidences[f"hand_{idx+1}"] = confidence
            
            # Draw landmarks
            if debug:
                self.mp_draw.draw_landmarks(
                    debug_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
        # Prepare return format
        gesture_dict = {
            "num_hands_detected": len(gestures),
            "gestures": gestures
        }
        
        output_image = Image.fromarray(debug_image) if debug else image
        
        return gesture_dict, confidences, output_image