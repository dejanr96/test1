import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_latest_metrics(log_dir):
    # Find all PPO directories
    ppo_dirs = glob.glob(os.path.join(log_dir, "PPO_*"))
    if not ppo_dirs:
        print("No PPO directories found.")
        return

    # Sort by creation time or number
    # Assuming PPO_N format, sort by N
    try:
        latest_dir = sorted(ppo_dirs, key=lambda x: int(x.split("_")[-1]))[-1]
    except:
        latest_dir = max(ppo_dirs, key=os.path.getmtime)
        
    print(f"Reading metrics from: {latest_dir}")
    
    # Find event file
    event_files = glob.glob(os.path.join(latest_dir, "events.out.tfevents*"))
    if not event_files:
        print("No event file found.")
        return
        
    event_file = event_files[0]
    
    # Load events
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    # Extract scalars
    tags = ea.Tags()['scalars']
    
    metrics = {}
    interesting_tags = ['rollout/ep_rew_mean', 'train/explained_variance', 'train/learning_rate', 'rollout/trades_taken']
    
    for tag in interesting_tags:
        if tag in tags:
            events = ea.Scalars(tag)
            if events:
                latest_event = events[-1]
                metrics[tag] = latest_event.value
                print(f"{tag}: {latest_event.value:.4f} (Step {latest_event.step})")
        else:
            print(f"{tag}: Not found")

if __name__ == "__main__":
    get_latest_metrics("c:/Users/666/Desktop/HFT/tensorboard_logs")
