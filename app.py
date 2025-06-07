import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

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
        logits = self.fc3(h2)
        return logits

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
    # Policy
    policy = PolicyNetwork()
    policy.load_state_dict(torch.load(POLICY_CHECKPOINT, map_location='cpu'))
    policy.eval()
    # Transcoder
    in_dim = 4 if LAYER=='layer1' else 128
    out_dim = 128 if LAYER=='layer1' else 64
    transcoder = SingleLayerTranscoder(in_dim, TRANSCODER_DIM, out_dim)
    ckpt = os.path.join(MODEL_DIR, f"{TRANSCODER_DIM}dim_transcoder_{LAYER}.pth")
    transcoder.load_ckpt(ckpt)
    decoder_col = transcoder.decoder.weight.data[:, NEURON].view(1, -1)
    return policy, decoder_col

policy, decoder_col = load_models()

# ==== Run Episodes (with optional frame capture) ====
def run_episodes(intervene_on=None):
    env = gym.make(ENV_NAME)
    rewards, fail_dirs = [], []
    frames = []

    for ep in range(NUM_EPISODES):
        state = env.reset()
        done = False
        total_r = 0.0
        ep_frames = []
        while not done:
            frame = env.render(mode='rgb_array')
            if ep == 0:
                ep_frames.append(frame)
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                logits = policy(state_tensor, intervene_on)
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
            state, r, done, _ = env.step(action)
            total_r += r
        # classify failure for metrics
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

# ==== Streamlit UI ====
st.title("CartPole Intervention Explorer")

# Run comparison and capture frames
intervene_cfg = {'layer': LAYER, 'decoder_col': decoder_col, 'scale': ALPHA} if ALPHA != 0.0 else None
if st.button("Run Comparison"):
    base_rew, base_fail, base_frames = run_episodes(None)
    int_rew, int_fail, int_frames = run_episodes(intervene_cfg)

    # Summary metrics
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

    # Episode playback
    st.subheader("Sample Episode Playback")
    col1, col2 = st.beta_columns(2)
    with col1:
        st.markdown("**Baseline**")
        idx = st.slider("Frame index (baseline)", 0, len(base_frames)-1, 0)
        st.image(base_frames[idx], caption=f"Frame {idx}")
    with col2:
        st.markdown("**Intervention**")
        idx2 = st.slider("Frame index (intervention)", 0, len(int_frames)-1, 0, key='int_idx')
        st.image(int_frames[idx2], caption=f"Frame {idx2}")
