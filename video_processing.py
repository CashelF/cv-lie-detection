import cv2
import numpy as np
from scipy.spatial import distance
import mediapipe as mp

FACEMESH_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
EYE_BLINK_HEIGHT = .15 # threshold may depend on relative face shape

def crop_image(image, top_left, top_right, bottom_right, bottom_left):
    """Crop the image based on the provided top and bottom landmark coordinates."""
    top_y = int((top_right.y + top_left.y) / 2 * image.shape[0])
    bottom_y = int((bottom_right.y + bottom_left.y) / 2 * image.shape[0])
    left_x = int((top_left.x + bottom_left.x) / 2 * image.shape[1])
    right_x = int((top_right.x + bottom_right.x) / 2 * image.shape[1])
    
    return image[top_y:bottom_y, right_x:left_x]

def process_face_and_hands(image, face_mesh, hands_detector):
    """Process the image to detect face and hand landmarks using MediaPipe."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_mesh.process(image_rgb)
    hands_landmarks = hands_detector.process(image_rgb).multi_hand_landmarks
    face_landmarks = faces.multi_face_landmarks[0] if faces.multi_face_landmarks else None
    return face_landmarks, hands_landmarks

def get_cheek_intensity(image, face):
    """Calculate average intensity of the cheeks without the blue channel."""
    cheek_left = crop_image(image, face[449], face[350], face[429], face[280])
    cheek_right = crop_image(image, face[121], face[229], face[50], face[209])
    return np.mean(cheek_left[:, :, 1:3]) + np.mean(cheek_right[:, :, 1:3])

def check_hand_on_face(hands_landmarks, face):
    """Check if any hand is on the face based on predefined face mesh oval."""
    if hands_landmarks:
        face_contours = np.array([[[landmark.x, landmark.y] for landmark in (face[p] for p in FACEMESH_FACE_OVAL)]]).astype(np.single)
        for hand_landmarks in hands_landmarks:
            for finger_index in [4, 8, 20]:  # Check specific fingers for overlap
                if cv2.pointPolygonTest(face_contours, (hand_landmarks.landmark[finger_index].x, hand_landmarks.landmark[finger_index].y), False) != -1:
                    return True
    return False

def get_aspect_ratio(landmarks):
    """Calculate the aspect ratio of a rectangle defined by four landmarks."""
    height = distance.euclidean((landmarks[0].x, landmarks[0].y), (landmarks[1].x, landmarks[1].y))
    width = distance.euclidean((landmarks[2].x, landmarks[2].y), (landmarks[3].x, landmarks[3].y))
    return height / width

def draw_landmarks(image, face_landmarks, hand_landmarks):
    """Draw facial and hand landmarks on the image using MediaPipe styles."""
    if face_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image, face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None, connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
        mp.solutions.drawing_utils.draw_landmarks(
            image, face_landmarks, mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None, connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())
    for hand_landmarks in hand_landmarks or []:
        mp.solutions.drawing_utils.draw_landmarks(
            image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style())
    
def get_gaze(face, iris_left_idx, iris_right_idx, eye_corner_left_idx, eye_corner_right_idx):
    """Calculate the relative gaze direction based on iris and eye corner landmarks."""
    iris_center = (
        (face[iris_left_idx].x + face[iris_right_idx].x) / 2,
        (face[iris_left_idx].y + face[iris_right_idx].y) / 2
    )
    eye_center = (
        (face[eye_corner_left_idx].x + face[eye_corner_right_idx].x) / 2,
        (face[eye_corner_left_idx].y + face[eye_corner_right_idx].y) / 2
    )

    gaze_distance = distance.euclidean(iris_center, eye_center)
    eye_width = abs(face[eye_corner_right_idx].x - face[eye_corner_left_idx].x)
    gaze_ratio = gaze_distance / eye_width

    # Adjust gaze ratio for left vs right direction
    if iris_center[0] < eye_center[0]:
        gaze_ratio *= -1

    return gaze_ratio

def get_avg_gaze(face):
    """Compute the average gaze direction combining left and right eye data."""
    gaze_left = get_gaze(face, 476, 474, 263, 362)
    gaze_right = get_gaze(face, 471, 469, 33, 133)
    avg_gaze = (gaze_left + gaze_right) / 2
    return round(avg_gaze, 1)

def get_is_blinking(face):
    """Determine if blinking by calculating the eye aspect ratio below a set threshold."""
    eye_right_points = [face[idx] for idx in [159, 145, 133, 33]]
    eye_left_points = [face[idx] for idx in [386, 374, 362, 263]]

    eye_right_aspect_ratio = get_aspect_ratio(eye_right_points)
    eye_left_aspect_ratio = get_aspect_ratio(eye_left_points)

    average_eye_aspect_ratio = (eye_right_aspect_ratio + eye_left_aspect_ratio) / 2
    return average_eye_aspect_ratio < EYE_BLINK_HEIGHT
