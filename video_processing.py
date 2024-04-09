import cv2
import numpy as np
from scipy.spatial import distance
import mediapipe as mp

FACEMESH_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]

def crop_image(image, topL, topR, bottomR, bottomL):
  topY = int((topR.y+topL.y)/2 * image.shape[0])
  botY = int((bottomR.y+bottomL.y)/2 * image.shape[0])
  leftX = int((topL.x+bottomL.x)/2 * image.shape[1])
  rightX = int((topR.x+bottomR.x)/2 * image.shape[1])

  return image[topY:botY, rightX:leftX]

def find_face_and_hands(image_original, face_mesh, hands):
  image = image_original.copy()
  image.flags.writeable = False # pass by reference to improve speed
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  faces = face_mesh.process(image)
  hands_landmarks = hands.process(image).multi_hand_landmarks

  face_landmarks = None
  if faces.multi_face_landmarks and len(faces.multi_face_landmarks) > 0:
    face_landmarks = faces.multi_face_landmarks[0] # use first face found

  return face_landmarks, hands_landmarks

def check_hand_on_face(hands_landmarks, face):
  if hands_landmarks:
    face_landmarks = [face[p] for p in FACEMESH_FACE_OVAL]
    face_points = [[[p.x, p.y] for p in face_landmarks]]
    face_contours = np.array(face_points).astype(np.single)

    for hand_landmarks in hands_landmarks:
      hand = []
      for point in hand_landmarks.landmark:
        hand.append( (point.x, point.y) )

      for finger in [4, 8, 20]:
        overlap = cv2.pointPolygonTest(face_contours, hand[finger], False)
        if overlap != -1:
          return True
  return False

def get_aspect_ratio(top, bottom, right, left):
  height = distance.euclidean([top.x, top.y], [bottom.x, bottom.y])
  width = distance.euclidean([right.x, right.y], [left.x, left.y])
  return height / width

def draw_landmarks_on_frame(image, face_landmarks, hands_landmarks):
  mp.solutions.drawing_utils.draw_landmarks(
      image,
      face_landmarks,
      mp.solutions.face_mesh.FACEMESH_CONTOURS,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp.solutions.drawing_styles
      .get_default_face_mesh_contours_style())
  mp.solutions.drawing_utils.draw_landmarks(
      image,
      face_landmarks,
      mp.solutions.face_mesh.FACEMESH_IRISES,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp.solutions.drawing_styles
      .get_default_face_mesh_iris_connections_style())
  for hand_landmarks in (hands_landmarks or []):
    mp.solutions.drawing_utils.draw_landmarks(
        image,
        hand_landmarks,
        mp.solutions.hands.HAND_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
        mp.solutions.drawing_styles.get_default_hand_connections_style())
    
def get_gaze(face, iris_L_side, iris_R_side, eye_L_corner, eye_R_corner):
  iris = (
    face[iris_L_side].x + face[iris_R_side].x,
    face[iris_L_side].y + face[iris_R_side].y,
  )
  eye_center = (
    face[eye_L_corner].x + face[eye_R_corner].x,
    face[eye_L_corner].y + face[eye_R_corner].y,
  )

  gaze_dist = distance.euclidean(iris, eye_center)
  eye_width = abs(face[eye_R_corner].x - face[eye_L_corner].x)
  gaze_relative = gaze_dist / eye_width

  if (eye_center[0] - iris[0]) < 0: # flip along x for looking L vs R
    gaze_relative *= -1

  return gaze_relative

def get_avg_gaze(face):
  gaze_left = get_gaze(face, 476, 474, 263, 362)
  gaze_right = get_gaze(face, 471, 469, 33, 133)
  return round((gaze_left + gaze_right) / 2, 1)
