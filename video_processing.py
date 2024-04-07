import cv2



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