import cv2
import torch
import numpy as np
from ..visualize import visualize_parsing

def parse_face(face_crop, parsing_session, visualize=False, device='cpu'):
    img_input = cv2.resize(face_crop, (512, 512))
    img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        out, _, _ = parsing_session(img_tensor)
    parsing = out.squeeze(0).argmax(0).cpu().numpy()
    if visualize:
        visualize_parsing(parsing, img_input)
    return parsing, img_input

def are_glasses_transparent(parsing_map, img_input, threshold=50):
    glasses_mask = parsing_map == 6
    if not np.any(glasses_mask):
        return True 

    brightness = img_input[glasses_mask].mean(axis=1)
    mean_brightness = brightness.mean()

    return mean_brightness > threshold

def has_feature(parsing_map, feature_idx):
    return feature_idx in parsing_map

def check_face_features(parsing_map):
    min_required = {
        2: ("left eyebrow", 0.4),
        3: ("right eyebrow", 0.4),
        4: ("left eye", 0.3),
        5: ("right eye", 0.3),
        10: ("nose", 2.0),
        12: ("upper lip", 0.4),
        13: ("lower lip", 0.4),
    }

    if has_feature(parsing_map, 6):
        check_indices = [2, 3, 10, 12, 13] # skip eyes
    else:
        check_indices = [2, 3, 4, 5, 10, 12, 13]  

    total_pixels = parsing_map.size
    missing = []

    for idx in check_indices:
        name, threshold = min_required[idx]
        area_percent = (np.sum(parsing_map == idx) / total_pixels) * 100
        if area_percent < threshold:
            missing.append(name)

    if missing:
        if len(missing) == 1:
            return f"Please show your {missing[0]} more clearly"
        elif len(missing) == 2:
            return f"Please show your {missing[0]} and {missing[1]} more clearly"
        else:
            return f"Please show your {', '.join(missing[:-1])} and {missing[-1]} more clearly"
    else:
        return None
    
def check_occlusions(parsing_map, glasses_transparent):
    occlusions = []

    if has_feature(parsing_map, 6) and not glasses_transparent:
        occlusions.append("glasses")

    if has_feature(parsing_map, 18):
        occlusions.append("hat or cap")

    if occlusions:
        if len(occlusions) == 1:
            return f"Please remove your {occlusions[0]}"
        else:
            return f"Please remove your {occlusions[0]} and {occlusions[1]}"
    return None
