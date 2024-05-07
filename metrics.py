import numpy as np
from scipy.signal import find_peaks
from fer import FER

import threading
from collections import deque

from config import Config
from image_processing import check_hand_on_face, get_aspect_ratio, get_avg_gaze, get_is_blinking, get_cheek_intensity

class MetricsCalculator:
  def __init__(self):
    self.initialize_metrics()
    self.emotion_detector = FER(mtcnn=True)
    self.current_emotion = 6
    self.mood_thread = None
    
  def initialize_metrics(self):
    self.cheek_color_values = deque(maxlen=Config.MAX_FRAMES)
    self.gaze_values = deque(maxlen=Config.MAX_FRAMES)
    self.blinks = deque(maxlen=Config.MAX_FRAMES)

  def collect_metrics(self, image, face_landmarks, hands_landmarks):
    face_landmarks = face_landmarks.landmark
    
    cheeks_intensity = get_cheek_intensity(image, face_landmarks)
    self.cheek_color_values.append(cheeks_intensity) # TODO: update outside of collect metrics for better encapsulation
    
    if self.mood_thread is None or not self.mood_thread.is_alive():
      self.mood_thread = threading.Thread(target=self.async_get_emotion, args=(image,))
      self.mood_thread.start()
    
    is_hand_on_face = check_hand_on_face(hands_landmarks, face_landmarks)
    
    lip_compression_ratio = self.get_lip_ratio(face_landmarks)
    
    avg_gaze = get_avg_gaze(face_landmarks)
    self.gaze_values.append(avg_gaze)
    gaze_change = self.detect_gaze_change(avg_gaze)
    
    is_blinking = get_is_blinking(face_landmarks)
    self.blinks.append(is_blinking) # TODO: update outside of collect metrics for better encapsulation
    
    return [self.get_bpm(), self.current_emotion, is_hand_on_face,
                lip_compression_ratio, avg_gaze, self.get_blink_rate()]
       
  def get_bpm(self):
    peak_frames, _ = find_peaks(list(self.cheek_color_values),
                              threshold=0.1, distance=5, prominence=0.5, wlen=10)

    bpms = 60 * np.diff(peak_frames) / Config.FPS
    valid_bpms = bpms[(bpms > 50) & (bpms < 150)]
    if valid_bpms.size == 0:
      return None
    return int(np.mean(valid_bpms))
  
  def async_get_emotion(self, image):
    emotions = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6
    }
    detected_emotion, score = self.emotion_detector.top_emotion(image)
    if score and (score > .4 or detected_emotion == 'neutral'):
      self.current_emotion = emotions.get(detected_emotion)
    else:
      self.current_emotion = 6

  def get_lip_ratio(self, face_landmarks):
    lip_points = [face_landmarks[idx] for idx in [0, 17, 61, 291]]
    return get_aspect_ratio(lip_points)
  
  def detect_gaze_change(self, avg_gaze):
    return self.gaze_values.count(avg_gaze) / len(self.gaze_values) < 0.01
    
  def get_blink_rate(self):
    return sum(self.blinks) / len(self.blinks)
