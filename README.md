This repository contains the isolated Proof of Concept (PoC) for the NeuroTwin research framework. It models an adaptive virtual reality learning application using multimodal learner-state estimation to intelligently tailor educational interventions.

# Used repositories:

https://github.com/danijar/dreamerv3

https://github.com/facebookresearch/jepa

## Overview
This pipeline bridges two state-of-the-art AI paradigms to simulate an intelligent XR tutor:
*   **V-JEPA (Vision Joint-Embedding Predictive Architecture):** Extracts spatial-temporal behavioral embeddings directly from raw XR interaction videos.
*   **DreamerV3 (World Model RL):** Simulates the decision-making engine that observes the extracted learner state and selects optimal interventions.
# Command Line (To run locally):


```
conda activate neurotwin_poc
```

```
python neurotwin_poc.py
```

## Execution
| Functions | Corresponding action | 
| -------- | -------- | 
| extract_jepa_features  | JEPA latent representation   |
|   progress_from_jepa   | Progress of user according to video clip   |
| dreamer_action_policy  | Predicting future state and selecting hint/simplify/continue   | 
|   step_environment     | calculates next state and reward   |    

# Execution Process Diagram

<img width="855" height="1308" alt="diagram" src="https://github.com/user-attachments/assets/0390a176-e177-4da8-96d0-442a35753e47" />


