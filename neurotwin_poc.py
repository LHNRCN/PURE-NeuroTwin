import cv2
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# extracts V-JEPA latent tensor from the video (AI generated)
def extract_jepa_features(video_filename):
    # 1. Read and resize 16 frames
    cap = cv2.VideoCapture(video_filename)
    frames = []
    for _ in range(16):
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224)))
    cap.release()

    # 2. Pad to ensure exactly 16 frames
    while len(frames) < 16:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

    # 3. Format into the exact 5D Tensor V-JEPA expects
    video_tensor = torch.tensor(np.array(frames)).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0

    # 4. Extract the latent representations
    encoder, _ = torch.hub.load('./vjepa2', 'vjepa2_1_vit_base_384', source='local')
    encoder.eval()

    with torch.no_grad():
        return encoder(video_tensor)

#gives a discrete progress value (0.0 or 10.0) based on the V-JEPA latent tensor
def progress_from_jepa(jepa_latent):
    feature_activation = jepa_latent.mean().item() #AI suggested
    
    if feature_activation > 0.0:
        return 10.0
    else:
        return 0.0

# decides the action (hint / simplify / continue) based on the current state of the learner
def dreamer_action_policy(current_state):
    progress, errors, hints, time = current_state
    if errors > 2 and hints > 1: # repeated errors
        return "Simplify"
    elif time > 60 and progress < 50: # excessive time spent with low progress
        return "Hint"
    else:
        return "Continue"

#calculates the next state and reward based on the current state and action taken
def step_environment(current_state, action, video_filename):
    progress, errors, hints, time = current_state
    reward = 0
    time += 15.0 # 15 seconds pass per action step

    # extratcted output from JEPA model
    jepa_latent = extract_jepa_features(video_filename) # Extract V-JEPA latent tensor from the video

    # mimicing the XR data
    if action == "Continue":
        current_progress = progress_from_jepa(jepa_latent) # interperate the V-JEPA latent tensor to determine progress
        progress += current_progress
        if current_progress == 0: # No progress made, repeated errors
            errors += 1
            reward -= 1
        else:
            reward += 1
    elif action == "Hint":
        hints += 1
        progress += 15
    elif action == "Simplify":
        progress += 20
        
    if time > 120: # Excessive time
        reward -= 1
        
    new_state = [progress, errors, hints, time]
    return new_state, reward

# ----main part---- #

print("--- NeuroTwin PoC (Isolated) ---")

# State: [task_progress, number_of_errors, number_of_hints, time_on_task]
current_state = [0.0, 0.0, 0.0, 0.0]
print("Initial Learner State:", current_state, "\n")

number_of_iterations = 5
for i in range(number_of_iterations):
    print("\nIteration ", i+1, ":", sep = "")

    #--- Current Learner State ---#
    print("Current Learner State   : ", current_state)

    #--- Selecting adaptation policy according to possible learner needs ---#
    action = dreamer_action_policy(current_state)
    print("  Selected Action       : ", action)

    #--- updated states and rewards---#
    video_filename = "interaction.mp4" # Placeholder for the video filename
    current_state, reward = step_environment(current_state, action, video_filename) #calculated via the environment model
    print("  Resulting Reward      : ", reward)
