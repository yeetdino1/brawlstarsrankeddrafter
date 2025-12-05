#%%
import streamlit as st
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
#%%
# --- Page Config ---
st.set_page_config(page_title="Brawl Stars AI Drafter", layout="wide")

# --- Constants & Config ---
MODEL_FILES = {
    'judge': 'brawl_win_predictor.keras',
    'fp_agent': 'agent_first_pick.keras',
    'lp_agent': 'agent_last_pick.keras',
    'vocabs': 'brawl_vocabs.json'
}

# Standard Draft Order used in Training
# Actor 0 = First Pick Team, Actor 1 = Last Pick Team
DRAFT_ORDER = [
    (0, 'ban', 1), (0, 'ban', 2), (0, 'ban', 3), # FP Bans
    (1, 'ban', 1), (1, 'ban', 2), (1, 'ban', 3), # LP Bans
    (0, 'pick', 1),                              # FP Pick 1
    (1, 'pick', 1), (1, 'pick', 2),              # LP Pick 1, 2
    (0, 'pick', 2), (0, 'pick', 3),              # FP Pick 2, 3
    (1, 'pick', 3)                               # LP Pick 3
]

# --- Helper Functions ---

@st.cache_resource
def load_resources():
    """Loads models and vocabs once to improve performance."""
    resources = {}
    
    # Load Vocabs
    try:
        with open(MODEL_FILES['vocabs'], 'r') as f:
            resources['vocabs'] = json.load(f)
            # Create reverse lookups
            resources['id_to_brawler'] = {v: k for k, v in resources['vocabs']['brawler'].items()}
            resources['id_to_mode'] = {v: k for k, v in resources['vocabs']['mode'].items()}
            resources['id_to_map'] = {v: k for k, v in resources['vocabs']['map'].items()}
    except FileNotFoundError:
        st.error(f"Could not find {MODEL_FILES['vocabs']}. Please upload it.")
        return None

    # Load Models
    try:
        resources['judge'] = load_model(MODEL_FILES['judge'])
        resources['agent_fp'] = load_model(MODEL_FILES['fp_agent'])
        resources['agent_lp'] = load_model(MODEL_FILES['lp_agent'])
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None
        
    return resources

def get_state_vector(resources, mode, map_name, fp_bans, lp_bans, fp_picks, lp_picks):
    """
    Reconstructs the 14-integer state vector expected by the RL agents.
    Order: [Mode, Map, FP_Ban1-3, LP_Ban1-3, FP_Pick1-3, LP_Pick1-3]
    """
    vocabs = resources['vocabs']
    
    # Helper to get ID safe
    def get_id(name, type_):
        return vocabs[type_].get(name, 0) # 0 is usually 'none' or unknown

    vec = [get_id(mode, 'mode'), get_id(map_name, 'map')]
    
    # Pad lists to ensure fixed size
    f_b = (fp_bans + ['none']*3)[:3]
    l_b = (lp_bans + ['none']*3)[:3]
    f_p = (fp_picks + ['none']*3)[:3]
    l_p = (lp_picks + ['none']*3)[:3]
    
    # Convert names to IDs
    for item in f_b: vec.append(get_id(item, 'brawler'))
    for item in l_b: vec.append(get_id(item, 'brawler'))
    for item in f_p: vec.append(get_id(item, 'brawler'))
    for item in l_p: vec.append(get_id(item, 'brawler'))
    
    return np.array(vec)

def predict_model_move(agent, state_vector, available_mask):
    """Uses the RL agent to predict the next best move."""
    # Reshape for model input (1, 14)
    state_input = state_vector.reshape(1, 14)
    
    # Get Q-values
    q_values = agent.predict(state_input, verbose=0)[0]
    
    # Apply Mask (Set unavailable to -infinity)
    # We use a very large negative number instead of -inf to avoid NaN issues in some TF versions
    masked_q_values = np.where(available_mask == 1, q_values, -1e9)
    
    # Get best action ID
    action_id = np.argmax(masked_q_values)
    return action_id

def get_judge_prediction(model, resources, mode, map_name, fp_bans, lp_bans, fp_picks, lp_picks):
    """Uses the Trainer/Judge model to give final win probability."""
    vocabs = resources['vocabs']
    
    # The Judge expects picks in draft order: P1(FP), P2(LP), P3(LP), P4(FP), P5(FP), P6(LP)
    # We have them separated by team in our state, so we must interleave them.
    
    # Pad picks to ensure we can access indices
    fp = (fp_picks + ['none']*3)
    lp = (lp_picks + ['none']*3)
    
    # Interleave Logic
    final_picks_order = [
        fp[0], lp[0], lp[1], fp[1], fp[2], lp[2]
    ]
    
    # Create the input dict structure
    vec_mode = np.array([vocabs['mode'].get(mode, 0)])
    vec_map = np.array([vocabs['map'].get(map_name, 0)])
    
    brawler_names = (
        (fp_bans + ['none']*3)[:3] + 
        (lp_bans + ['none']*3)[:3] + 
        final_picks_order
    )
    
    vec_brawlers = np.array([[vocabs['brawler'].get(b, 0) for b in brawler_names]])
    
    inputs = {
        'mode_input': vec_mode,
        'map_input': vec_map,
        'brawlers_input': vec_brawlers
    }
    
    win_prob_lp = model.predict(inputs, verbose=0)[0][0]
    return float(win_prob_lp)

# --- Initialization ---

res = load_resources()

if 'draft_state' not in st.session_state:
    st.session_state.draft_state = {
        'step': 0,
        'fp_bans': [],
        'lp_bans': [],
        'fp_picks': [],
        'lp_picks': [],
        'finished': False
    }

# --- Sidebar Controls ---

st.sidebar.title("Draft Settings")

# Reset Button
if st.sidebar.button("Reset Draft"):
    st.session_state.draft_state = {
        'step': 0,
        'fp_bans': [],
        'lp_bans': [],
        'fp_picks': [],
        'lp_picks': [],
        'finished': False
    }
    st.rerun()

# Initialize variables to defaults to prevent NameError
selected_mode = None
selected_map = None
draft_type = "Solo (Me vs All)"
my_side = "First Pick Team"
my_specific_pick = None

if res:
    # Dropdowns
    mode_options = [m for m in res['vocabs']['mode'].keys() if m != 'none']
    map_options = [m for m in res['vocabs']['map'].keys() if m != 'none']
    
    selected_mode = st.sidebar.selectbox("Select Mode", mode_options)
    selected_map = st.sidebar.selectbox("Select Map", map_options)
    
    st.sidebar.markdown("---")
    draft_type = st.sidebar.radio("Draft Type", ["Solo (Me vs All)", "Team (Me vs Opponent)"])
    
    my_side = st.sidebar.radio("My Side", ["First Pick Team", "Last Pick Team"])
    
    my_specific_pick = None
    if draft_type == "Solo (Me vs All)":
        # Identify which pick slot the user is
        if my_side == "First Pick Team":
            my_specific_pick = st.sidebar.selectbox("I am...", ["Pick 1", "Pick 4", "Pick 5"])
        else:
            my_specific_pick = st.sidebar.selectbox("I am...", ["Pick 2", "Pick 3", "Pick 6"])

# --- Main Interface ---

st.title("⚔️ Brawl Stars AI Drafter")

if not res:
    st.error("Failed to load models or vocabulary. Please check your files.")
    st.stop()

# 1. Display Draft State
col1, col2 = st.columns(2)

with col1:
    st.subheader("First Pick Team (Blue)")
    st.write(f"🚫 Bans: {', '.join(st.session_state.draft_state['fp_bans'])}")
    for i, p in enumerate(st.session_state.draft_state['fp_picks']):
        st.info(f"Pick {i+1}: {p.title()}")

with col2:
    st.subheader("Last Pick Team (Red)")
    st.write(f"🚫 Bans: {', '.join(st.session_state.draft_state['lp_bans'])}")
    for i, p in enumerate(st.session_state.draft_state['lp_picks']):
        st.error(f"Pick {i+1}: {p.title()}")

st.markdown("---")

# 2. Logic to determine who acts next
ds = st.session_state.draft_state

# We add 'if res' here to be absolutely sure we don't run logic without resources
if res and not ds['finished']:
    current_step_idx = ds['step']
    actor_side, action_type, sub_idx = DRAFT_ORDER[current_step_idx]
    
    actor_name = "First Pick Team" if actor_side == 0 else "Last Pick Team"
    action_desc = f"{action_type.upper()} {sub_idx}"
    
    st.header(f"Current Turn: {actor_name} - {action_desc}")
    
    # 3. Determine if USER acts or MODEL acts
    user_is_acting = False
    
    if draft_type == "Team (Me vs Opponent)":
        # If I am FP Team (Side 0), I act when Actor == 0
        if my_side == "First Pick Team" and actor_side == 1:
            user_is_acting = True # I input opponent moves
        elif my_side == "Last Pick Team" and actor_side == 0:
            user_is_acting = True # I input opponent moves
        else:
            user_is_acting = False # Model suggests MY moves

    elif draft_type == "Solo (Me vs All)":
        # I only act if the current specific slot matches "my_specific_pick"
        # Map step index to "Pick X" strings
        # Indices: 6=P1, 7=P2, 8=P3, 9=P4, 10=P5, 11=P6
        pick_map = {
            6: "Pick 1", 7: "Pick 1", # FP P1, LP P1
            8: "Pick 2", 9: "Pick 2", # LP P2, FP P2
            10: "Pick 3", 11: "Pick 3" # FP P3, LP P3
        }
        
        # Map current index to the human-readable string used in the sidebar
        current_pick_name = ""
        if current_step_idx == 6: current_pick_name = "Pick 1"
        elif current_step_idx == 7: current_pick_name = "Pick 2"
        elif current_step_idx == 8: current_pick_name = "Pick 3"
        elif current_step_idx == 9: current_pick_name = "Pick 4"
        elif current_step_idx == 10: current_pick_name = "Pick 5"
        elif current_step_idx == 11: current_pick_name = "Pick 6"
        
        # In Solo mode: 
        if action_type == 'ban':
            user_is_acting = True
        elif action_type == 'pick':
            if current_pick_name == my_specific_pick:
                user_is_acting = False # Model drafts FOR me
            else:
                user_is_acting = True # I input everyone else
        
    # 4. Input Area
    
    # Calculate Available Brawlers
    taken = set(ds['fp_bans'] + ds['lp_bans'] + ds['fp_picks'] + ds['lp_picks'])
    all_brawlers = list(res['vocabs']['brawler'].keys())
    available = sorted([b for b in all_brawlers if b != 'none' and b not in taken])
    
    if user_is_acting:
        st.subheader("Manual Input")
        with st.form("user_action"):
            selected = st.selectbox("Select Brawler:", available)
            submitted = st.form_submit_button("Confirm Selection")
            
            if submitted:
                # Update State
                if action_type == 'ban':
                    if actor_side == 0: ds['fp_bans'].append(selected)
                    else: ds['lp_bans'].append(selected)
                else:
                    if actor_side == 0: ds['fp_picks'].append(selected)
                    else: ds['lp_picks'].append(selected)
                
                ds['step'] += 1
                if ds['step'] >= len(DRAFT_ORDER):
                    ds['finished'] = True
                st.rerun()
                
    else:
        st.subheader("🤖 AI Suggestion")
        
        # Prepare data for AI
        state_vec = get_state_vector(
            res, selected_mode, selected_map,
            ds['fp_bans'], ds['lp_bans'], ds['fp_picks'], ds['lp_picks']
        )
        
        # Create mask (1=Available, 0=Taken)
        mask = np.zeros(len(all_brawlers))
        for i, b in enumerate(all_brawlers):
            if b == 'none': mask[i] = 0
            elif b in taken: mask[i] = 0
            else: mask[i] = 1
            
        # Select Agent
        agent = res['agent_fp'] if actor_side == 0 else res['agent_lp']
        
        # Predict
        if st.button("Generate AI Pick"):
            action_id = predict_model_move(agent, state_vec, mask)
            chosen_brawler = res['id_to_brawler'][action_id]
            
            st.success(f"AI Selected: **{chosen_brawler.upper()}**")
            
            # Update State
            if action_type == 'ban':
                if actor_side == 0: ds['fp_bans'].append(chosen_brawler)
                else: ds['lp_bans'].append(chosen_brawler)
            else:
                if actor_side == 0: ds['fp_picks'].append(chosen_brawler)
                else: ds['lp_picks'].append(chosen_brawler)
            
            ds['step'] += 1
            if ds['step'] >= len(DRAFT_ORDER):
                ds['finished'] = True
            st.rerun()

# 5. Final Prediction (The Judge)
if res and (ds['finished'] or st.button("Predict Confidence Now")):
    st.markdown("### 🔮 Confidence Prediction")
    
    prob_lp = get_judge_prediction(
        res['judge'], res, selected_mode, selected_map,
        ds['fp_bans'], ds['lp_bans'], ds['fp_picks'], ds['lp_picks']
    )
    
    prob_fp = 1.0 - prob_lp
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("First Pick Team Win %", f"{prob_fp*100:.1f}%")
        st.progress(prob_fp)
    with col_b:
        st.metric("Last Pick Team Win %", f"{prob_lp*100:.1f}%")
        st.progress(prob_lp)
        
    if prob_fp > 0.55:
        st.success("First Pick Team is favored!")
    elif prob_lp > 0.55:
        st.error("Last Pick Team is favored!")
    else:
        st.warning("It's an even match!")