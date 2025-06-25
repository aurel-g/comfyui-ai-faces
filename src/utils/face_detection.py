def detect_faces(image, insight_model):
    results = insight_model.get(image)
    boxes = []
    for face in results:
        x1, y1, x2, y2 = map(int, face.bbox)
        conf = float(face.det_score)
        pose = face.pose
        boxes.append(((x1, y1, x2, y2), conf, pose))
    return boxes

def face_size(box, image_shape):
    img_area = image_shape[0] * image_shape[1]
    face_area = (box[2] - box[0]) * (box[3] - box[1])
    return face_area / img_area 

def is_head_pose_acceptable(pose, thresh=30):
    if pose is None:
        return "Unable to determine head position."

    pitch, yaw, roll = map(abs, pose)
    if yaw > thresh:
        return "Turn your head more towards the camera (too much side rotation)."
    if pitch > thresh:
        return "Don't tilt your head too much up or down."
    if roll > thresh:
        return "Don't tilt your head to the side."

    return None
