import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
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
NUM_EPISODES = st.sidebar.number_input("Episodes", 1, 100, 10)
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
        if intervene_on and intervene_on['layer']=='layer1':
            h1 = h1 + intervene_on['decoder_col'] * intervene_on['scale']
        z2 = self.fc2(h1)
        h2 = F.relu(z2)
        if intervene_on and intervene_on['layer']=='layer2':
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
    in_dim = 4 if LAYER=='layer1' else 128
    out_dim = 128 if LAYER=='layer1' else 64
    transcoder = SingleLayerTranscoder(in_dim, TRANSCODER_DIM, out_dim)
    ckpt = os.path.join(MODEL_DIR, f"{TRANSCODER_DIM}dim_transcoder_{LAYER}.pth")
    transcoder.load_ckpt(ckpt)
    decoder_col = transcoder.decoder.weight.data[:, NEURON].view(1, -1)
    return policy, decoder_col

policy, decoder_col = load_models()

# ==== Run Episodes ====
def run_episodes(intervene_on=None):
    # Initialize headless renderable env
    env = gym.make(ENV_NAME, render_mode='rgb_array')
    rewards, fail_dirs, frames = [], [], []
    for ep in range(NUM_EPISODES):
        reset_out = env.reset()
        state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        done = False
        total_r = 0.0
        ep_frames = []
        while not done:
            # capture frame for first episode
            frame = env.render()
            if ep == 0:
                ep_frames.append(frame)
            state_tensor = torch.from_numpy(np.asarray(state)).float().unsqueeze(0)
            with torch.no_grad():
                logits = policy(state_tensor, intervene_on)
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
            step_out = env.step(action)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                state, r, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                state, r, done, _ = step_out
            total_r += r
        # classify failure
        angle, pos = state[2], state[0]
        if abs(angle) >= 0.2094:
            fd = 'pole'
        elif abs(pos) >= 2.4:
            fd = 'cart'
        else:
            fd = 'none'
        rewards.append(total_r)
        fail_dirs.append(fd)
        if ep == 0:
            frames = ep_frames
    env.close()
    return rewards, fail_dirs, frames

# ==== Utility: Build GIF ====
def build_gif(frames, duration=50):
    # Filter out any None frames just in case
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
st.title("CartPole Intervention Explorer")
intervene_cfg = {'layer': LAYER, 'decoder_col': decoder_col, 'scale': ALPHA} if ALPHA != 0.0 else None
if st.button("Run Comparison"):
    base_rew, base_fail, base_frames = run_episodes(None)
    int_rew, int_fail, int_frames = run_episodes(intervene_cfg)

    st.subheader("Average Rewards")
    st.write(f"Baseline: {np.mean(base_rew):.2f} | Intervention: {np.mean(int_rew):.2f}")

    st.subheader("Failure Counts")
    df = pd.DataFrame({
        'Baseline': pd.Series(base_fail).value_counts(),
        'Intervention': pd.Series(int_fail).value_counts()
    }).fillna(0).astype(int)
    st.table(df)

    st.subheader("Reward Distributions")
    fig, ax = plt.subplots()
    ax.hist(base_rew, alpha=0.5, label='Baseline')
    ax.hist(int_rew, alpha=0.5, label='Intervention')
    ax.legend()
    st.pyplot(fig)

    st.subheader("Sample Episode Playback (Animation)")
    # Safely build GIFs
    gif1 = build_gif(base_frames)
    gif2 = build_gif(int_frames)
    col1, col2 = st.columns(2)
    with col1:
        st.image(gif1, caption="Baseline", use_column_width=True)
    with col2:
        st.image(gif2, caption="Intervention", use_column_width=True)
