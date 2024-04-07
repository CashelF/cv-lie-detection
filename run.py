import argparse

import cv2
import mediapipe as mp

from config import Config
from metrics import MetricsCalculator

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', '-i', nargs='*', default=['0'], help='Input video device (number or path), file, or screen dimensions (x y width height), defaults to 0')
  parser.add_argument('--mesh', '-m', action='store_true', help='Enable drawing of face and hand mesh')

  args = parser.parse_args()

  if len(args.input) == 1:
      INPUT = int(args.input[0])
  elif len(args.input) != 4:
      return print("Wrong number of values for 'input' argument; should be 0, 1, or 4.")

  DRAW_MESH = args.mesh
  
  
  
  cap = cv2.VideoCapture(INPUT)
  Config.FPS = cap.get(cv2.CAP_PROP_FPS) # time tell shows should be 1 second
  if isinstance(INPUT, str) and INPUT.find('.') == -1: # from camera, not file
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)


  face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
  hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
  
  try:
    metrics_calculator = MetricsCalculator()
    while cap.isOpened():
      success, image = cap.read()
      if not success: break
      # calibration_frames += process(image, face_mesh, hands, calibrated, DRAW_MESH, fps)
      # calibrated = (calibration_frames >= Config.MAX_FRAMES)
      metrics = metrics_calculator.collect_metrics(image, face_mesh, hands)
      print("Metrics:", metrics)
      
      cv2.imshow('face', image)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
      
  finally:
    cap.release()
    face_mesh.close()
    hands.close()
    cv2.destroyAllWindows()
  
  
  
if __name__ == '__main__':
  main()