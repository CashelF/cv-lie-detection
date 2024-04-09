import numpy as np
from scipy.signal import find_peaks
from fer import FER

import threading

from config import Config
from video_processing import check_hand_on_face, get_aspect_ratio, get_avg_gaze, get_is_blinking, get_cheeks_intensity_without_blue_avg

class MetricsCalculator:
  def __init__(self):
    self.cheek_color_values = [9999] * Config.MAX_FRAMES
    self.gaze_values = [0] * Config.MAX_FRAMES
    self.blinks = [False] * Config.MAX_FRAMES
    self.emotion_detector = FER(mtcnn=True)
    self.current_emotion = None
    self.mood_thread = None

  def collect_metrics(self, image, face_landmarks, hands_landmarks):
    face = face_landmarks.landmark
    
    cheeks_intensity_withoutBlue = get_cheeks_intensity_without_blue_avg(image, face)
    self.update_cheek_color_values(cheeks_intensity_withoutBlue) # TODO: update outside of collect metrics for better encapsulation
    bpm = self.get_bpm()
    
    if self.mood_thread is None or not self.mood_thread.is_alive():
      self.mood_thread = threading.Thread(target=self.async_get_emotion, args=(image,))
      self.mood_thread.start()
    
    is_hand_on_face = self.get_hand_on_face(hands_landmarks, face)
    
    lip_compression_ratio = self.get_lip_ratio(face)
    
    avg_gaze = get_avg_gaze(face)
    gaze_change = self.detect_gaze_change(avg_gaze)
    
    is_blinking = get_is_blinking(face)
    self.update_blinks(is_blinking) # TODO: update outside of collect metrics for better encapsulation
    blink_rate = self.get_blink_rate()
    
    return bpm, self.current_emotion, is_hand_on_face, lip_compression_ratio, gaze_change, blink_rate
    
  def update_cheek_color_values(self, cheeks_color_value):
    self.cheek_color_values = self.cheek_color_values[1:] + [cheeks_color_value]
    
  def get_bpm(self):
    peak_frames, _ = find_peaks(self.cheek_color_values,
      threshold=.1,
      distance=5,
      prominence=.5,
      wlen=10,
    )

    bpms = 60 * np.diff(peak_frames) / Config.FPS
    bpms = bpms[(bpms > 50) & (bpms < 150)] # filter to reasonable BPM range
    recent_bpms = bpms[(-3 * Config.RECENT_FRAMES):] # HR slower signal than other tells
    
    if len(recent_bpms) == 0:
      return None

    recent_avg_bpm = np.average(recent_bpms).astype(int)

    return recent_avg_bpm
  
  def async_get_emotion(self, image):
    detected_emotion, score = self.emotion_detector.top_emotion(image)
    if score and (score > .4 or detected_emotion == 'neutral'):
      self.current_emotion = detected_emotion

  def get_hand_on_face(self, hands_landmarks, face):
    return check_hand_on_face(hands_landmarks, face)

  def get_lip_ratio(self, face):
    return get_aspect_ratio(face[0], face[17], face[61], face[291])
  
  def detect_gaze_change(self, avg_gaze):
    self.gaze_values = self.gaze_values[1:] + [avg_gaze]
    gaze_relative_matches = 1.0 * self.gaze_values.count(avg_gaze) / Config.MAX_FRAMES
    if gaze_relative_matches < .01: # looking in a new direction
      return True
    return False
  
  def update_blinks(self, is_blinking):
    self.blinks = self.blinks[1:] + [is_blinking]
    
  def get_blink_rate(self):
    return sum(self.blinks) / len(self.blinks)
