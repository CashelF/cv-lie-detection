from metrics import *
import threading

from view import *

LIP_COMPRESSION_RATIO = .35 # from testing, ~universal

def process(image, face_mesh, hands, calibrated=False, draw=False, bpm_chart=False, flip=False, fps=None):
  global tells, calculating_mood
  global blinks, hand_on_face, face_area_size

  tells = decrement_tells(tells)

  face_landmarks, hands_landmarks = find_face_and_hands(image, face_mesh, hands)
  if face_landmarks:
    face = face_landmarks.landmark
    face_area_size = get_face_relative_area(face)

    if not calculating_mood:
      emothread = threading.Thread(target=get_mood, args=(image,))
      emothread.start()
      calculating_mood = True

    # TODO check cheek visibility?
    cheekL = get_area(image, draw, topL=face[449], topR=face[350], bottomR=face[429], bottomL=face[280])
    cheekR = get_area(image, draw, topL=face[121], topR=face[229], bottomR=face[50], bottomL=face[209])

    avg_bpms, bpm_change = get_bpm_tells(cheekL, cheekR, fps, bpm_chart)
    tells['avg_bpms'] = new_tell(avg_bpms) # always show "..." if BPM missing
    if len(bpm_change):
      tells['bpm_change'] = new_tell(bpm_change)

    # Blinking
    blinks = blinks[1:] + [is_blinking(face)]
    recent_blink_tell = get_blink_tell(blinks)
    if recent_blink_tell:
      tells['blinking'] = new_tell(recent_blink_tell)

    # Hands on face
    recent_hand_on_face = check_hand_on_face(hands_landmarks, face)
    hand_on_face = hand_on_face[1:] + [recent_hand_on_face]
    if recent_hand_on_face:
      tells['hand'] = new_tell("Hand covering face")

    # Gaze tracking
    avg_gaze = get_avg_gaze(face)
    if detect_gaze_change(avg_gaze):
      tells['gaze'] = new_tell("Change in gaze")

    # Lip compression
    if get_lip_ratio(face) < LIP_COMPRESSION_RATIO:
      tells['lips'] = new_tell("Lip compression")

    if bpm_chart: # update chart
      fig.canvas.draw()
      fig.canvas.flush_events()

    if draw: # overlay face and hand landmarks
      draw_on_frame(image, face_landmarks, hands_landmarks)

  if flip:
    image = cv2.flip(image, 1) # flip image horizontally

  add_text(image, tells, calibrated)
  add_truth_meter(image, len(tells))

  return 1 if (face_landmarks and not calibrated) else 0


def mirror_compare(first, second, rate, less, more):
  if (rate * first) < second:
    return less
  elif first > (rate * second):
    return more
  return None

def get_blink_comparison(blinks1, blinks2):
  return mirror_compare(sum(blinks1), sum(blinks2), 1.8, "Blink less", "Blink more")

def get_hand_face_comparison(hand1, hand2):
  return mirror_compare(sum(hand1), sum(hand2), 2.1, "Stop touching face", "Touch face more")

def get_face_size_comparison(ratio1, ratio2):
  return mirror_compare(ratio1, ratio2, 1.5, "Too close", "Too far")


# process optional second input for mirroring
def process_second(cap, image, face_mesh, hands):
  global blinks, blinks2
  global hand_on_face, hand_on_face2
  global face_area_size

  success2, image2 = cap.read()
  if success2:
    face_landmarks2, hands_landmarks2 = find_face_and_hands(image2, face_mesh, hands)

    if face_landmarks2:
      face2 = face_landmarks2.landmark

      blinks2 = blinks2[1:] + [is_blinking(face2)]
      blink_mirror = get_blink_comparison(blinks, blinks2)

      hand_on_face2 = hand_on_face2[1:] + [check_hand_on_face(hands_landmarks2, face2)]
      hand_face_mirror = get_hand_face_comparison(hand_on_face, hand_on_face2)

      face_area_size2 = get_face_relative_area(face2)
      face_ratio_mirror = get_face_size_comparison(face_area_size, face_area_size2)

      text_y = 2 * TEXT_HEIGHT # show prompts below 'mood' on right side
      for comparison in [blink_mirror, hand_face_mirror, face_ratio_mirror]:
        if comparison:
          write(comparison, image, int(.75 * image.shape[1]), text_y)
          text_y += TEXT_HEIGHT
          
def decrement_tells(tells):
  for key, tell in tells.copy().items():
    if 'ttl' in tell:
      tell['ttl'] -= 1
      if tell['ttl'] <= 0:
        del tells[key]
  return tells

def new_tell(result):
  global TELL_MAX_TTL

  return {
    'text': result,
    'ttl': TELL_MAX_TTL
  }