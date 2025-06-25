import cv2
import time
from .utils.face_detection import detect_faces, face_size, is_head_pose_acceptable
from .utils.parsing import parse_face, are_glasses_transparent, check_face_features, check_occlusions
from .logs import error, ok
from .visualize import visualize_face_detections
from .models import load_models

MODELS = load_models()
DEVICE = MODELS["device"]

def process_image(path, visualize=False):
    start = time.time()

    # === 1. Load image ===
    image = cv2.imread(str(path))
    if image is None:
        return error("Failed to load image. Please ensure the file exists.", start)

    # === 2. Face detection ===
    boxes = detect_faces(image, MODELS["insight_model"])
    if visualize:
        visualize_face_detections(path, boxes)

    if not boxes:
        return error("No face detected. Please ensure your face is clearly visible.", start)
    elif len(boxes) != 1:
        return error("Only one person should be in the photo. Please ensure you are alone in the image.", start)

    # === 3. Check face size ===
    box, conf, pose = boxes[0]
    size = face_size(box, image.shape)
    if size <= 0.1:
        return error("Move closer to the camera so your face occupies more of the image.", start)
    if size >= 0.7:
        return error("Move away from the camera so your face occupies less of the image.", start)

    # === 3. Check head pose angle ===
    if (msg := is_head_pose_acceptable(pose)):
        return error(msg, start)
    
    # === 4. Assess image quality ===
    if conf < 0.75:
        return error("Take a photo with better lighting and ensure image sharpness.", start)

    # === 5. Face segmentation and analysis ===
    face_crop = image[box[1]:box[3], box[0]:box[2]]
    parsing_map, img_input = parse_face(face_crop, MODELS["parsing_session"], visualize, device=DEVICE)

    glasses_transparent = are_glasses_transparent(parsing_map, img_input, threshold=50)

    if (msg := check_occlusions(parsing_map, glasses_transparent)):
        return error(msg, start)

    if (msg := check_face_features(parsing_map)):
        return error(msg, start)

    return ok("Photo accepted. Thank you!", start)
