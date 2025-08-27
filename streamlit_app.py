import streamlit as st
from pathlib import Path
import tempfile

from beatviz.renderer import render_tracked_effect

st.set_page_config(page_title="BeatViz", layout="wide")

st.title("🎶 BeatViz — Beat-Synced Video Overlay")

# Upload video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
if uploaded_file:
    # Save uploaded file to temp path
    tmp_in = Path(tempfile.mktemp(suffix=".mp4"))
    with open(tmp_in, "wb") as f:
        f.write(uploaded_file.read())

    st.subheader("⚙️ Parameters")

    fps = st.number_input("FPS (leave 0 for auto)", min_value=0, value=0)
    life_frames = st.slider("Life Frames", 1, 60, 10)
    pts_per_beat = st.slider("Points per Beat (max)", 1, 50, 20)
    ambient_rate = st.slider("Ambient spawn rate (per second)", 0.0, 20.0, 5.0)
    jitter_px = st.slider("Jitter (px)", 0.0, 3.0, 0.5)
    min_size = st.slider("Min square size", 5, 50, 15)
    max_size = st.slider("Max square size", 10, 100, 40)
    neighbor_links = st.slider("Neighbor links per point", 0, 10, 3)
    orb_fast_threshold = st.slider("ORB FAST threshold", 1, 50, 20)
    bell_width = st.slider("Bell width divisor", 1.0, 10.0, 4.0)
    seed = st.number_input("Random seed (0 = none)", min_value=0, value=0)
    st.markdown("### 🎛️ Audio-reactive Settings")
    beat_pulse_strength = st.slider("Beat pulse strength", 0.0, 1.0, 0.35, 0.05)
    beat_tolerance_s = st.slider("Beat tolerance (sec)", 0.00, 0.20, 0.05, 0.01)
    bass_size_boost = st.slider("Bass size boost", 0.0, 1.0, 0.25, 0.05)
    colorize_from_bands = st.toggle("Colorize from (low/mid/high) bands", True)

    if st.button("▶️ Process Video"):
        with st.spinner("Processing... This may take a while."):
            tmp_out = Path(tempfile.mktemp(suffix=".mp4"))

            render_tracked_effect(
                video_in=tmp_in,
                video_out=tmp_out,
                fps=None if fps == 0 else fps,
                pts_per_beat=pts_per_beat,
                ambient_rate=ambient_rate,
                jitter_px=jitter_px,
                life_frames=life_frames,
                min_size=min_size,
                max_size=max_size,
                neighbor_links=neighbor_links,
                orb_fast_threshold=orb_fast_threshold,
                bell_width=bell_width,
                seed=None if seed == 0 else seed,

                beat_pulse_strength=beat_pulse_strength,
                beat_tolerance_s=beat_tolerance_s,
                bass_size_boost=bass_size_boost,
                colorize_from_bands=colorize_from_bands,
            )
            st.success("Done! ✅")

            # Side-by-side display with smaller videos
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📥 Input Video**")
                st.video(str(tmp_in), width=400)  # Set width to 400px
            with col2:
                st.markdown("**🎨 Processed Video**")
                st.video(str(tmp_out), width=400)  # Set width to 400px

            # Download option
            with open(tmp_out, "rb") as f:
                st.download_button("⬇️ Download Output Video", f, file_name="output_with_boxes.mp4")