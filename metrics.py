import numpy as np
from scipy.signal import find_peaks

import time

from config import Config
from video_processing import crop_image, find_face_and_hands

class MetricsCalculator:
  def __init__(self):
    self.hr_values = [9999] * Config.MAX_FRAMES

  def collect_metrics(self, image, face_mesh, hands):
    face_landmarks, hands_landmarks = find_face_and_hands(image, face_mesh, hands)
    if not face_landmarks:
      return None
    face = face_landmarks.landmark
    
    bpm = self.get_bpm(image, face)
    
    return bpm
    
    
    
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


