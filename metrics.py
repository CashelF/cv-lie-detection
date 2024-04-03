import cv2
import numpy as np
from scipy.signal import find_peaks
import time
from scipy.spatial import distance as dist
from fer import FER

MAX_FRAMES = 120 # modify this to affect calibration period and amount of "lookback"
EPOCH = time.time()
RECENT_FRAMES = int(MAX_FRAMES / 10) # modify to affect sensitivity to recent changes
SIGNIFICANT_BPM_CHANGE = 8
EYE_BLINK_HEIGHT = .15 # threshold may depend on relative face shape
FACEMESH_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]

hr_times = list(range(0, MAX_FRAMES))
hr_values = [400] * MAX_FRAMES
avg_bpms = [0] * MAX_FRAMES

emotion_detector = FER(mtcnn=True)

# for BPM chart
ax = None
line = None
peakpts = None

def get_aspect_ratio(top, bottom, right, left):
  height = dist.euclidean([top.x, top.y], [bottom.x, bottom.y])
  width = dist.euclidean([right.x, right.y], [left.x, left.y])
  return height / width


def get_area(image, draw, topL, topR, bottomR, bottomL):
  topY = int((topR.y+topL.y)/2 * image.shape[0])
  botY = int((bottomR.y+bottomL.y)/2 * image.shape[0])
  leftX = int((topL.x+bottomL.x)/2 * image.shape[1])
  rightX = int((topR.x+bottomR.x)/2 * image.shape[1])

  if draw:
    image = cv2.circle(image, (leftX,topY), 2, (255,0,0), 2)
    image = cv2.circle(image, (leftX,botY), 2, (255,0,0), 2)
    image = cv2.circle(image, (rightX,topY), 2, (255,0,0), 2)
    image = cv2.circle(image, (rightX,botY), 2, (255,0,0), 2)

  return image[topY:botY, rightX:leftX]


def get_bpm_tells(cheekL, cheekR, fps, bpm_chart):
  global hr_values, hr_times, avg_bpms
    
  cheekLwithoutBlue = np.average(cheekL[:, :, 1:3])
  cheekRwithoutBlue = np.average(cheekR[:, :, 1:3])
  hr_values = hr_values[1:] + [cheekLwithoutBlue + cheekRwithoutBlue]

  if not fps:
    hr_times = hr_times[1:] + [time.time() - EPOCH]

  if bpm_chart:
    line.set_data(hr_times, hr_values)
    ax.relim()
    ax.autoscale()

  peaks, _ = find_peaks(hr_values,
    threshold=.1,
    distance=5,
    prominence=.5,
    wlen=10,
  )

  peak_times = [hr_times[i] for i in peaks]

  if bpm_chart:
    peakpts.set_data(peak_times, [hr_values[i] for i in peaks])

  bpms = 60 * np.diff(peak_times) / (fps or 1)
  bpms = bpms[(bpms > 50) & (bpms < 150)] # filter to reasonable BPM range
  recent_bpms = bpms[(-3 * RECENT_FRAMES):] # HR slower signal than other tells

  recent_avg_bpm = 0
  bpm_display = "BPM: ..."
  if recent_bpms.size > 1:
    recent_avg_bpm = int(np.average(recent_bpms))
    bpm_display = "BPM: {} ({})".format(recent_avg_bpm, len(recent_bpms))

  avg_bpms = avg_bpms[1:] + [recent_avg_bpm]

  bpm_delta = 0
  bpm_change = ""

  if len(recent_bpms) > 2:
    all_bpms = list(filter(lambda bpm: bpm != '-', avg_bpms))
    all_avg_bpm = sum(all_bpms) / len(all_bpms)
    avg_recent_bpm = sum(recent_bpms) / len(recent_bpms)
    bpm_delta = avg_recent_bpm - all_avg_bpm

    if bpm_delta > SIGNIFICANT_BPM_CHANGE:
      bpm_change = "Heart rate increasing"
    elif bpm_delta < -SIGNIFICANT_BPM_CHANGE:
      bpm_change = "Heart rate decreasing"

  return bpm_display, bpm_change


def is_blinking(face):
  eyeR = [face[p] for p in [159, 145, 133, 33]]
  eyeR_ar = get_aspect_ratio(*eyeR)

  eyeL = [face[p] for p in [386, 374, 362, 263]]
  eyeL_ar = get_aspect_ratio(*eyeL)

  eyeA_ar = (eyeR_ar + eyeL_ar) / 2
  return eyeA_ar < EYE_BLINK_HEIGHT


def get_blink_tell(blinks):
  if sum(blinks[:RECENT_FRAMES]) < 3: # not enough blinks for valid comparison
    return None

  recent_closed = 1.0 * sum(blinks[-RECENT_FRAMES:]) / RECENT_FRAMES
  avg_closed = 1.0 * sum(blinks) / MAX_FRAMES

  if recent_closed > (20 * avg_closed):
    return "Increased blinking"
  elif avg_closed >  (20 * recent_closed):
    return "Decreased blinking"
  else:
    return None


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


def get_avg_gaze(face):
  gaze_left = get_gaze(face, 476, 474, 263, 362)
  gaze_right = get_gaze(face, 471, 469, 33, 133)
  return round((gaze_left + gaze_right) / 2, 1)


def get_gaze(face, iris_L_side, iris_R_side, eye_L_corner, eye_R_corner):
  iris = (
    face[iris_L_side].x + face[iris_R_side].x,
    face[iris_L_side].y + face[iris_R_side].y,
  )
  eye_center = (
    face[eye_L_corner].x + face[eye_R_corner].x,
    face[eye_L_corner].y + face[eye_R_corner].y,
  )

  gaze_dist = dist.euclidean(iris, eye_center)
  eye_width = abs(face[eye_R_corner].x - face[eye_L_corner].x)
  gaze_relative = gaze_dist / eye_width

  if (eye_center[0] - iris[0]) < 0: # flip along x for looking L vs R
    gaze_relative *= -1

  return gaze_relative


def detect_gaze_change(avg_gaze):
  gaze_values = [0] * MAX_FRAMES
  gaze_values = gaze_values[1:] + [avg_gaze]
  gaze_relative_matches = 1.0 * gaze_values.count(avg_gaze) / MAX_FRAMES
  if gaze_relative_matches < .01: # looking in a new direction
    return gaze_relative_matches
  return 0


def get_lip_ratio(face):
  return get_aspect_ratio(face[0], face[17], face[61], face[291])


def get_mood(image):
  global mood
  detected_mood, score = emotion_detector.top_emotion(image)
  if score and (score > .4 or detected_mood == 'neutral'):
    mood = detected_mood
    
def get_face_relative_area(face):
  face_width = abs(max(face[454].x, 0) - max(face[234].x, 0))
  face_height = abs(max(face[152].y, 0) - max(face[10].y, 0))
  return face_width * face_height


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