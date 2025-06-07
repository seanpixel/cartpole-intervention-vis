import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import streamlit as st
import io
from PIL import Image

# ─── CONFIG ──────────────────────────────────────────────────────────────────────
ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"
POLICY_CHECKPOINT = os.path.join(MODEL_DIR, "policy_cartpole.pth")
TRANSCODER_DIM = st.sidebar.selectbox("Transcoder dim", [2, 4, 8], index=1)
LAYER = st.sidebar.selectbox("Intervene on layer", ["layer1", "layer2"], index=0)
NEURON = st.sidebar.number_input("Neuron index", 0, TRANSCODER_DIM - 1, 0)
ALPHA = st.sidebar.slider("Alpha scale", -3.0, 3.0, 0.0, 0.1)
# Only one episode for visualization
# ─────────────────────────────────────────────────────────────────────────────────

# ==== Policy Definition ====
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

# ==== Transcoder Definition ====
class SingleLayerTranscoder(nn.Module):
    def __init__(self, in_dim, latent_dim, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, latent_dim), nn.ReLU())
        self.decoder = nn.Linear(latent_dim, out_dim)

    def load_ckpt(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
        self.eval()

# ==== Load Models ====
@st.cache(allow_output_mutation=True)
def load_models():
    policy = PolicyNetwork()
    policy.load_state_dict(torch.load(POLICY_CHECKPOINT, map_location='cpu'))
    policy.eval()
    in_dim = 4 if LAYER == 'layer1' else 128
    out_dim = 128 if LAYER == 'layer1' else 64
    transcoder = SingleLayerTranscoder(in_dim, TRANSCODER_DIM, out_dim)
    ckpt = os.path.join(MODEL_DIR, f"{TRANSCODER_DIM}dim_transcoder_{LAYER}.pth")
    transcoder.load_ckpt(ckpt)
    decoder_col = transcoder.decoder.weight.data[:, NEURON].view(1, -1)
    return policy, decoder_col

policy, decoder_col = load_models()

# ==== Run Single Episode ====
def run_episode(intervene_on=None):
    env = gym.make(ENV_NAME)
    reset_out = env.reset()
    state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    done = False
    frames = []
    while not done:
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        state_tensor = torch.from_numpy(np.asarray(state)).float().unsqueeze(0)
        with torch.no_grad():
            logits = policy(state_tensor, intervene_on)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        step_out = env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            state, _, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            state, _, done, _ = step_out
    env.close()
    return frames

# ==== Utility: Build GIF ====
def build_gif(frames, duration=30):
    valid_frames = [f for f in frames if f is not None]
    pil_frames = [Image.fromarray(f) for f in valid_frames]
    buf = io.BytesIO()
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

# ==== Streamlit UI ====
st.title("CartPole Causal Intervention Visualization")
if st.button("Show Effects"):
    # Baseline and Intervention GIFs
    baseline_frames = run_episode(None)
    intervention_cfg = {'layer': LAYER, 'decoder_col': decoder_col, 'scale': ALPHA} if ALPHA != 0.0 else None
    intervention_frames = run_episode(intervention_cfg)

    gif_base = build_gif(baseline_frames)
    gif_int = build_gif(intervention_frames)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Baseline")
        st.image(gif_base, use_column_width=True)
    with col2:
        st.subheader("Intervention")
        st.image(gif_int, use_column_width=True)
