import cv2
import torch

# 1. Patch Meta's internal URL issue
_orig = torch.hub.load_state_dict_from_url
torch.hub.load_state_dict_from_url = lambda u, *a, **kw: _orig(u.replace("http://localhost:8300", "https://dl.fbaipublicfiles.com/vjepa2"), *a, **kw)

# 2. Read 16 frames from interaction.mp4 and resize to 224x224
cap = cv2.VideoCapture("interaction.mp4")
frames = []
for _ in range(16):
    ret, frame = cap.read()
    if ret:
        frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224)))
cap.release()

# Pad if the video is shorter than 16 frames
while len(frames) < 16:
    frames.append(frames[-1])

# 3. Shape tensor to V-JEPA format: [Batch(1), Channels(3), Frames(16), H(224), W(224)]
video_tensor = torch.tensor(frames).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0

# 4. Load V-JEPA encoder and extract features
encoder, _ = torch.hub.load('./vjepa2', 'vjepa2_1_vit_base_384', source='local')
encoder.eval()

with torch.no_grad():
    embeddings = encoder(video_tensor)

print("V-JEPA Latent Shape:", embeddings.shape)