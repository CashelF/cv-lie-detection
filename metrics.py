import numpy as np
from scipy.signal import find_peaks
from fer import FER

import threading

from config import Config
from video_processing import crop_image, find_face_and_hands, check_hand_on_face

class MetricsCalculator:
  def __init__(self):
    self.hr_values = [9999] * Config.MAX_FRAMES
    self.emotion_detector = FER(mtcnn=True)
    self.current_emotion = None
    self.mood_thread = None

  def collect_metrics(self, image, face_mesh, hands):
    face_landmarks, hands_landmarks = find_face_and_hands(image, face_mesh, hands)
    if not face_landmarks:
      return None
    face = face_landmarks.landmark
    
    bpm = self.get_bpm(image, face) 
    
    if self.mood_thread is None or not self.mood_thread.is_alive():
      self.mood_thread = threading.Thread(target=self.async_get_emotion, args=(image,))
      self.mood_thread.start()
    
    is_hand_on_face = self.get_hand_on_face(hands_landmarks, face)
    
    return bpm, self.current_emotion, is_hand_on_face
    
  def get_bpm(self, image, face):
    cheekL = crop_image(image, topL=face[449], topR=face[350], bottomR=face[429], bottomL=face[280])
    cheekR = crop_image(image, topL=face[121], topR=face[229], bottomR=face[50], bottomL=face[209])

    cheekLwithoutBlue = np.average(cheekL[:, :, 1:3])
    cheekRwithoutBlue = np.average(cheekR[:, :, 1:3])
    self.hr_values = self.hr_values[1:] + [cheekLwithoutBlue + cheekRwithoutBlue]

    peak_frames, _ = find_peaks(self.hr_values,
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
