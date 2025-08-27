# 🎶 BeatWiz – Audio-Reactive Video Visualizer  

BeatWiz is a Python + Streamlit powered tool that lets you add **AI-powered, audio-synced visual effects** to videos.  
It detects **beats, rhythm, and frequency bands** from audio and syncs them with **particle effects, flashes, and transformations** in the video.  

---

## ✨ Features  
- 🎥 Upload video and generate effects automatically  
- 🔊 Audio beat + frequency analysis with **Librosa**  
- 🌌 Audio-driven effects (particles, flashes, distortions)  
- 💾 Save processed video with synced effects  
- 📊 Adjustable video preview size in Streamlit  
- ⚡ Future-ready for **real-time visualizer mode** (mic input)  

---

## 🛠️ Installation  

1. Clone the repo:  
```bash
git clone https://github.com/yourusername/beatwiz.git
cd beatwiz
```

2. Create a virtual environment & install dependencies:  
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Install FFmpeg (required for audio/video processing):  
- [FFmpeg Download](https://ffmpeg.org/download.html)  
- Make sure `ffmpeg` is in your PATH.  

---

## 📦 Dependencies  
Main libraries used:  
- [Streamlit](https://streamlit.io/) – UI  
- [MoviePy](https://zulko.github.io/moviepy/) – Video & audio extraction  
- [Librosa](https://librosa.org/) – Audio beat/frequency analysis  
- [OpenCV](https://opencv.org/) – Video frame rendering  
- [NumPy](https://numpy.org/) – Math & transformations  

Install via:  
```bash
pip install streamlit moviepy librosa opencv-python numpy
```

---

## 🚀 Usage  

Run the Streamlit app:  
```bash
streamlit run streamlit_app.py
```

### Inside the UI:  
1. 📤 Upload a video file  
2. 🎶 Audio is extracted & analyzed  
3. ✨ Beat-synced visuals are generated  
4. 🎥 Watch processed video directly in the app  
5. 💾 Option to save processed video  

---

## 🔮 Future Ideas  
- 🎛️ Add multiple visual effect styles (choose effect type)  
- 🕺 Real-time mode: mic input + live visualizer  
- 🤖 ML-powered style transfer (make it look like anime, glitch art, etc.)  
- 🎨 Customizable color palettes & particle systems  

---

## 📌 Example Output  
- Original video with **audio-driven pulse effects**  
- Flash bursts on beats  
- Particles synced to high-frequency sounds  


---

## 👨‍💻 Author  
Built with ❤️ by Kshitiz Kumar
