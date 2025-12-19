import pandas as pd
import numpy as np
import random
from pathlib import Path
from tqdm.notebook import tqdm
import ipywidgets as widgets
from IPython.display import display
import json

def generate_events(matrix, feature_event_types=None):
    events = []
    
    features = matrix.columns
    
    for feat in features:
        col = matrix[feat]
        
        # Determine which types to generate
        if feature_event_types is not None:
            selected_types = feature_event_types.get(feat, [])
        else:
            selected_types = ['exists', 'threshold', 'exact']  # All types if not specified
        
        # 1. Existence: "Contains a [Feature]"
        if 'exists' in selected_types:
            mask = col > 0
            prob = mask.mean()
            if 0.001 <= prob <= 0.95:
                events.append({
                    "description": f"Contains {feat}",
                    "type": "exists",
                    "feature": feat,
                    "condition": lambda c: c > 0,
                    "mask": mask.values,
                    "probability": prob
                })
            
        # 2. Thresholds: "More than N [Feature]s"
        max_val = col.max()
        if 'threshold' in selected_types and max_val > 1:
            for n in range(1, min(int(max_val), 6)):
                mask = col > n
                prob = mask.mean()
                if 0.05 <= prob <= 0.95:
                    events.append({
                        "description": f"More than {n} {feat}s",
                        "type": "threshold",
                        "feature": feat,
                        "condition": lambda c, n=n: c > n,
                        "mask": mask.values,
                        "probability": prob
                    })

        # 3. Exact Counts: "Exactly N [Feature]s"
        if 'exact' in selected_types and max_val >= 1:
            for n in range(1, min(int(max_val) + 1, 6)):
                mask = col == n
                prob = mask.mean()
                if 0.05 <= prob <= 0.95:
                    events.append({
                        "description": f"Exactly {n} {feat}{'s' if n>1 else ''}",
                        "type": "exact",
                        "feature": feat,
                        "condition": lambda c, n=n: c == n,
                        "mask": mask.values,
                        "probability": prob
                    })
                    
    return pd.DataFrame(events)

def calculate_turns_to_win(card_indices, truth_matrix, n_simulations=1000000):
    """
    Simulates the game n_simulations times for a given card.
    Returns the average number of turns (images drawn) to complete the card.
    """
    n_images, n_events = truth_matrix.shape
    card_mask = truth_matrix[:, card_indices] # Shape: (n_images, 10)
    
    # If a card has an event that is NEVER satisfied by any image, it's impossible.
    if np.any(card_mask.sum(axis=0) == 0):
        return float('inf')
    
    turns_needed = []
    
    # Create an array of image indices [0, 1, ... N-1]
    deck = np.arange(n_images)
    
    for _ in range(n_simulations):
        np.random.shuffle(deck)
        
        # Reorder the card_mask according to the shuffled deck
        shuffled_mask = card_mask[deck] # Shape: (n_images, 10)
        
        # Cumulative sum (or rather, cumulative OR)
        # We want to know when we have seen a True for every column.
        covered_cum = np.maximum.accumulate(shuffled_mask, axis=0) # Shape: (n_images, 10)
        
        # Check if all 10 are covered at each step
        all_covered = covered_cum.all(axis=1) # Shape: (n_images,)
        
        # Find the first index where all_covered is True
        if not all_covered.any():
             turns = n_images 
        else:
            turns = np.argmax(all_covered) + 1
            
        turns_needed.append(turns)
        
    return np.mean(turns_needed)

def create_cards_interactive(stats_path, output_path, num_cards=50, card_size=10, tolerance=1, target_difficulty=None):
    stats_path = Path(stats_path)
    output_path = Path(output_path)
    
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    df = pd.read_csv(stats_path)
    df["n_objects"] = df["n_objects"].fillna(0).astype(int)
    
    counts_matrix = df.pivot_table(index="image", columns="feature", values="n_objects", fill_value=0)
    features = list(counts_matrix.columns)

    # Create a widget for each feature
    feature_widgets = {}
    for feat in features:
        # Checkboxes for event types
        exists_cb = widgets.Checkbox(value=True, description='Contains')
        threshold_cb = widgets.Checkbox(value=False, description='More than N')
        exact_cb = widgets.Checkbox(value=False, description='Exactly N')
        
        # Group them in a VBox
        box = widgets.VBox([widgets.Label(f"Feature: {feat}"), exists_cb, threshold_cb, exact_cb])
        feature_widgets[feat] = box  # Store the VBox

    # Display all widgets
    widgets_list = [feature_widgets[feat] for feat in features]  # Use the VBox directly
    accordion = widgets.Accordion(children=widgets_list, titles=features)
    display(accordion)

    # Button to confirm selection
    confirm_button = widgets.Button(description="Confirm Selections & Generate Cards")
    output = widgets.Output()

    def on_confirm_clicked(b):
        with output:
            output.clear_output()
            feature_event_types = {}
            for feat, box in feature_widgets.items():
                # box.children: [Label, exists_cb, threshold_cb, exact_cb]
                exists_val = box.children[1].value
                threshold_val = box.children[2].value
                exact_val = box.children[3].value
                selected = []
                if exists_val:
                    selected.append('exists')
                if threshold_val:
                    selected.append('threshold')
                if exact_val:
                    selected.append('exact')
                feature_event_types[feat] = selected
            
            print("Generating candidate events...")
            events_df = generate_events(counts_matrix, feature_event_types)
            print(f"Generated {len(events_df)} candidate events.")
            
            # Compute Truth Matrix
            truth_matrix = np.stack(events_df["mask"].values).T
            
            # --- Card Generation Logic ---
            NUM_CARDS = num_cards
            CARD_SIZE = card_size
            TOLERANCE = tolerance
            
            feature_groups = events_df.groupby("feature").indices
            unique_features = list(feature_groups.keys())

            if len(unique_features) < CARD_SIZE:
                print(f"Error: Cannot create cards of size {CARD_SIZE}: Only {len(unique_features)} unique features available.")
                return

            if target_difficulty is None:
                print("Estimating baseline difficulty...")
                sample_difficulties = []
                for _ in range(50):
                    feats = random.sample(unique_features, CARD_SIZE)
                    idxs = [random.choice(feature_groups[f]) for f in feats]
                    d = calculate_turns_to_win(idxs, truth_matrix, n_simulations=100)
                    if d != float('inf'):
                        sample_difficulties.append(d)

                target_difficulty = np.median(sample_difficulties)
                print(f"Calculated Target Average Turns to Win (Median): {target_difficulty:.2f}")
            else:
                print(f"Using Manual Target Average Turns to Win: {target_difficulty:.2f}")

            final_cards = []
            final_stats = []

            print(f"Generating {NUM_CARDS} fair cards...")
            pbar = tqdm(total=NUM_CARDS)

            attempts = 0
            MAX_ATTEMPTS = NUM_CARDS * 5000 

            while len(final_cards) < NUM_CARDS:
                attempts += 1
                # 1. Pick N unique features
                selected_features = random.sample(unique_features, CARD_SIZE)
                
                found_config = False
                for _ in range(100): 
                    card_idxs = []
                    for feat in selected_features:
                        possible_idxs = feature_groups[feat]
                        card_idxs.append(random.choice(possible_idxs))
                        
                    # Check Difficulty
                    est_diff = calculate_turns_to_win(card_idxs, truth_matrix, n_simulations=50)
                    
                    if abs(est_diff - target_difficulty) < TOLERANCE * 2:
                        precise_diff = calculate_turns_to_win(card_idxs, truth_matrix, n_simulations=50000)
                        
                        if abs(precise_diff - target_difficulty) < TOLERANCE:
                            final_cards.append(card_idxs)
                            final_stats.append(precise_diff)
                            pbar.update(1)
                            found_config = True
                            break 
                
                if attempts > MAX_ATTEMPTS:
                    print("Warning: Difficulty finding cards. Try widening tolerance or changing target.")
                    break
                    
            pbar.close()
            
            # Export
            cards_data = []
            for i, idxs in enumerate(final_cards):
                card_events = events_df.iloc[idxs]["description"].tolist()
                cards_data.append({
                    "card_id": i + 1,
                    "events": card_events,
                    "avg_turns_to_win": final_stats[i]
                })
                
            with open(output_path, "w") as f:
                json.dump(cards_data, f, indent=2)
                
            print(f"Saved {len(cards_data)} cards to {output_path}")

    confirm_button.on_click(on_confirm_clicked)
    display(confirm_button, output)
