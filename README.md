# Used repositories:

https://github.com/danijar/dreamerv3

https://github.com/facebookresearch/jepa

# Command Line:

```
conda activate neurotwin_poc
```

```
python neurotwin_poc.py
```

| Functions | Corresponding action | 
| -------- | -------- | 
| extract_jepa_features  | JEPA latent representation   |
|   progress_from_jepa   | Progress of user according to video clip   |
| dreamer_action_policy  | Predicting future state and selecting hint/simplify/continue   | 
|   step_environment     | calculates next state and reward   |    

# Execution Process Diagram

<img width="855" height="1308" alt="diagram" src="https://github.com/user-attachments/assets/0390a176-e177-4da8-96d0-442a35753e47" />


