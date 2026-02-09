def compute_reward(prev_wer, curr_wer, status, step_penalty=0.05):
    reward = (prev_wer - curr_wer) - step_penalty

    if status == "oom":
        reward -= 1.0
    elif status == "nan":
        reward -= 2.0
    elif status == "diverged":
        reward -= 2.5

    return reward
