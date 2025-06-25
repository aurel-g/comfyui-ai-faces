import torch
import os
import requests 
import insightface
from src.model_bisenet import BiSeNet
import onnxruntime as ort

so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1

def load_models():
    # Sprawdzenie dostępności GPU
    device = "cpu"
    print(f"Używane urządzenie: {device}")

    # 1. BiseNet
    bisenet_weights_url = "https://github.com/zllrunning/face-makeup.PyTorch/raw/master/cp/79999_iter.pth"
    bisenet_weights_path = "/net/pr2/projects/plgrid/plggaigraphicsk46/models/79999_iter.pth"

    if not os.path.exists(bisenet_weights_path):
        with open(bisenet_weights_path, "wb") as f:
            f.write(requests.get(bisenet_weights_url).content)

    bisenet = BiSeNet(n_classes=19)
    bisenet.load_state_dict(torch.load(bisenet_weights_path, map_location=device))
    bisenet.to(device)
    bisenet.eval()

    # 2. InsightFace
    providers = ['CPUExecutionProvider']
    insight_model = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers, allowed_modules=['detection', 'landmark_3d_68'])
    insight_model.prepare(ctx_id=0 if 'CUDAExecutionProvider' in providers else -1)

    print("All models successfully loaded.")

    return {
        "parsing_session": bisenet,
        "insight_model": insight_model,
        "device": device  
    }
