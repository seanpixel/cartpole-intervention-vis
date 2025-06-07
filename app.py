import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import streamlit as st
import io
from PIL import Image

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"
POLICY_CHECKPOINT = os.path.join(MODEL_DIR, "policy_cartpole.pth")

# Only 4-dim transcoder
LATENT_DIM = 4

# Sidebar controls
LAYER = st.sidebar.selectbox("Intervene on layer", ["layer1", "layer2"])
NEURON = st.sidebar.number_input("Neuron index", 0, LATENT_DIM - 1, 0)
ALPHA = st.sidebar.slider("Intervention strength (α)", -3.0, 3.0, 0.0, 0.1)

# Feature descriptions for 4-dim model
feature_descriptions = {
    ('layer1', 0): "Cart far right and moving right with slight backward pole rotation.",
    ('layer1', 1): "Cart at right edge, nearly still, with pole upright and small forward tilt.",
    ('layer1', 2): "Cart near center with pole slightly tilting left and negative angular velocity.",
    ('layer1', 3): "Cart at right side with pole swinging forward (positive angular velocity).",
    ('layer2', 0): "High confidence state: cart moving forward, pole rotating backward strongly.",
    ('layer2', 1): "Dormant feature; remains inactive during typical play.",
    ('layer2', 2): "Forward-leaning pole with positive angular velocity while cart is right of center.",
    ('layer2', 3): "Moderate forward pole swing with slight backward cart velocity."
}

st.sidebar.markdown(
    f"**Feature Description**: {feature_descriptions.get((LAYER, NEURON), 'No description available.')}"
)

# ==== Policy network matching training ====
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x, intervene_on=None):
        z1 = self.fc1(x)
        h1 = F.relu(z1)
        if intervene_on and intervene_on['layer'] == 'layer1':
            h1 = h1 + intervene_on['decoder_col'] * intervene_on['scale']
        z2 = self.fc2(h1)
        h2 = F.relu(z2)
        if intervene_on and intervene_on['layer'] == 'layer2':
            h2 = h2 + intervene_on['decoder_col'] * intervene_on['scale']
        return self.fc3(h2)

# ==== Single-layer transcoder loader ====
class SingleLayerTranscoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, LATENT_DIM), nn.ReLU())
        # decoder maps latent back to hidden dim
        self.decoder = nn.Linear(LATENT_DIM, 128 if in_dim==4 else 64)

    def load_ckpt(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        self.eval()

# ==== Load models & extract decoder column ====
def load_models():
    # Load policy
    policy = PolicyNetwork().to(DEVICE)
    policy.load_state_dict(torch.load(POLICY_CHECKPOINT, map_location=DEVICE))
    policy.eval()
    # Choose transcoder checkpoint based on layer
    if LAYER == 'layer1':
        in_dim = 4
    else:
        in_dim = 128
    transcoder = SingleLayerTranscoder(in_dim).to(DEVICE)
    ckpt = os.path.join(MODEL_DIR, f"4dim_transcoder_{LAYER}.pth")
    transcoder.load_ckpt(ckpt)
    # Select decoder column for user-specified neuron
    decoder_col = transcoder.decoder.weight.data[:, NEURON].view(1, -1).to(DEVICE)
    return policy, decoder_col

policy, decoder_col = load_models()

# ==== Run single CartPole episode and capture frames ====
def run_episode(intervene_cfg=None):
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    reset_out = env.reset()
    state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = policy(state_tensor, intervene_cfg)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        step_out = env.step(action)
        if len(step_out) == 5:
            state, _, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            state, _, done, _ = step_out
    env.close()
    return frames

# ==== Build animated GIF ==== 
def build_gif(frames, duration=30):
    buf = io.BytesIO()
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        buf,
        format='GIF',
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )
    buf.seek(0)
    return buf

# ==== Streamlit interface ==== 
st.title("CartPole Causal Intervention (4‑Dim) Explorer")
st.text(f"You are boosting the feature that detects: '{feature_descriptions.get((LAYER, NEURON), 'No description available.')}' by {ALPHA}.\nThink of it as tricking the model that the feature is activating stronger / weaker based on alpha\n\nWait 10-20 seconds for the episode to generate!")
if st.button("Show Effects"):
    baseline_gif = build_gif(run_episode(None))
    intervention_cfg = {'layer': LAYER, 'decoder_col': decoder_col, 'scale': ALPHA} if ALPHA != 0.0 else None
    intervention_gif = build_gif(run_episode(intervention_cfg))
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Baseline")
        st.image(baseline_gif, use_container_width=True)
    with col2:
        st.subheader("Intervention")
        st.image(intervention_gif, use_container_width=True)
    st.text("\n\nWe see that the baseline model tends to keep moving to the right. When we boost features that think the pole is leaning right, the model actually improves at balancing by counteracting the baseline model's tendency to keep shifting right.\n\nOn the other hand, when we boost features like {layer 2 neuron 0} which activates when the pole rotates backward, the model makes the cart zip to the right since we are pushing the baseline to go even further to the right to counteract the pole leaning left.")