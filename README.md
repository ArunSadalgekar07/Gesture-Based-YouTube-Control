# 🎵 Gesture-Based YouTube Control 🎥  

Control your YouTube Video **hands-free** using simple hand gestures! This project uses **OpenCV, Mediapipe, and PyAutoGUI** to recognize gestures in real-time and simulate keyboard shortcuts for YouTube control.

## 📌 Features  
✅ **Play/Pause** → Open hand gesture 🖐️  
✅ **Volume Up** → Thumbs-up gesture 👍  
✅ **Volume Down** → Pinky-up gesture 🤙  
✅ **Real-time hand tracking** with **MediaPipe**  
✅ **Optimized performance** using time delay handling  

## 🛠️ Tech Stack  
- **Python** 🐍  
- **OpenCV** (for video processing)  
- **Mediapipe** (for hand tracking)  
- **PyAutoGUI** (for simulating keyboard events)  
- **Pynput** (for keyboard control)  

## 🎮 How It Works  
- The script **captures video** from your webcam.  
- **MediaPipe** detects your hand and identifies **finger positions**.  
- Based on the **detected gesture**, the script **simulates keyboard inputs** to control media.  

## 🔧 Customization  
You can modify the key bindings in the `media_play_pause.py` file to match shortcuts.  

## 🖥️ Demo  
 


