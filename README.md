# Real-Time Heart Rate Estimation and ECG Validation

## 📌 Overview  
The **ECG-like Heart Rate Monitoring & Validation** system is a computer vision-based project that estimates heart rate from a live webcam feed using facial detection techniques. It simulates an ECG-like heart rate plot in real time and optionally compares the generated ECG with a user-uploaded ECG report using **Structural Similarity Index (SSIM)** to measure similarity.  

The system leverages OpenCV’s **Deep Neural Network (DNN)** face detection model to extract a **Region of Interest (ROI)** from the user’s face, processes intensity variations to estimate heartbeats, and applies a **low-pass Butterworth filter** for noise reduction. The calculated BPM values are dynamically plotted as an ECG-like waveform for a **60-second duration**.  

> **Note:** This project is designed for **educational and validation purposes only** and is **not intended for real medical diagnosis**.  

---

## 🔄 System Pipeline  
1. **Capture frames from the webcam** and resize them for consistent processing.  
2. Use OpenCV’s DNN face detector (`deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`) to **locate and track** the user’s face.  
3. **Convert the detected face region to grayscale** to measure average pixel intensity, which subtly changes with blood flow.  
4. **Store extracted intensity values** in a rolling buffer for continuous tracking.  
5. **Apply a Butterworth low-pass filter** to smooth out noise.  
6. Use **SciPy's `find_peaks`** to detect heartbeat peaks.  
7. **Calculate BPM** from peak intervals.  
8. **Introduce occasional random BPM spikes** for realistic variability.  
9. **Plot BPM values in real time** using Matplotlib to generate a live ECG-like graph.  
10. **Save the complete 60-second ECG waveform** as `final_ecg_plot.png`.  
11. *(Optional)* **Upload an ECG image** and compute the **SSIM similarity percentage** between the generated and uploaded plots.  

---

## ⚙️ Implementation Details  

### 📚 Libraries Used  
- **OpenCV** – Face detection & frame processing  
- **NumPy** – Numerical operations  
- **Matplotlib** – Live ECG plotting  
- **SciPy** – Signal filtering & peak detection  
- **scikit-image** – SSIM similarity computation  
- **Tkinter** – GUI prompts for file selection  

### 📊 Output Metrics  
- **BPM values** displayed in real time  
- **Live ECG-like waveform** display  
- **Final ECG plot** saved as an image  
- **SSIM-based similarity accuracy (%)** for validation  
