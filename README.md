# Smart Parking System

A computer vision-based smart parking system that uses YOLO (You Only Look Once) object detection to monitor parking spaces in real-time. The system can detect cars, count occupied and free parking spaces, and provide visual feedback on parking space availability.

## 🚗 Features

- **Real-time Car Detection**: Uses YOLOv8 for accurate car detection
- **Parking Space Monitoring**: Tracks individual parking spaces with custom-defined areas
- **Visual Feedback**: Color-coded parking spaces (Green = Available, Red = Occupied)
- **Car Counting**: Real-time count of parked cars and available spaces
- **Multiple Input Sources**: Supports both IP camera streams and video files
- **Interactive Controls**: Pause, resume, and restart video playback
- **Firebase Integration Ready**: Prepared for real-time data synchronization (commented out)

## 📁 Project Structure

```
Smart Parking/
├── main-IP.py              # IP camera version
├── main-Video.py           # Video file version
├── Ajawan                  # Parking area definitions (pickle file)
├── coco.txt               # YOLO class names
├── FRAME.png              # Reference frame
├── Model/
│   ├── yolov8s.pt        # YOLOv8 model weights
│   └── yolo_model.pkl    # Additional model file
├── Videos/
│   ├── park1.mp4         # Sample video 1
│   └── park3.mp4         # Sample video 2
└── Parking Space/
    ├── Data/
        ├── park1.mp4
        └── park3.mp4

```

## 🛠️ Requirements

### Python Dependencies
```bash
pip install opencv-python
pip install ultralytics
pip install pandas
pip install numpy
pip install cvzone
```

### System Requirements
- Python 3.7 or higher
- OpenCV 4.5+
- CUDA-compatible GPU (optional, for faster inference)

## 🚀 Installation & Setup

1. **Clone or Download the Project**
   ```bash
   # Navigate to your project directory
   cd "Smart Parking"
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   # Or install individually:
   pip install opencv-python ultralytics pandas numpy cvzone
   ```

3. **Verify Model Files**
   - Ensure `Model/yolov8s.pt` exists
   - Ensure `Ajawan` (parking area definitions) exists
   - Ensure `coco.txt` (class names) exists

## 📹 Usage

### IP Camera Mode (`main-IP.py`)

1. **Configure IP Camera URL**
   ```python
   # In main-IP.py, line 70
   ip_cam_url = "your_ip_camera_url_here"
   ```

2. **Run the Application**
   ```bash
   python main-IP.py
   ```

3. **Controls**
   - Press `Esc` to exit

### Video File Mode (`main-Video.py`)

1. **Configure Video Path**
   ```python
   # In main-Video.py, line 18
   video_path = "Videos/park1.mp4"  # Change to your video file
   ```

2. **Run the Application**
   ```bash
   python main-Video.py
   ```

3. **Controls**
   - Press `Esc` to exit
   - Press `P` to pause/resume video
   - Press `R` to restart video from beginning

## 🎯 How It Works

### 1. **Model Loading**
- Loads YOLOv8 model for object detection
- Loads pre-defined parking area polygons from `Ajawan` file
- Loads COCO class names for object classification

### 2. **Frame Processing**
- Captures frames from video source (IP camera or video file)
- Resizes frames to 1020x400 pixels for consistent processing
- Runs YOLO detection on each frame

### 3. **Car Detection & Tracking**
- Detects cars using YOLO model
- Calculates center points of detected cars
- Maps car positions to parking space polygons

### 4. **Parking Space Analysis**
- Tests if car centers fall within parking space polygons
- Updates occupancy status for each parking space
- Color codes spaces: Green (available) vs Red (occupied)

### 5. **Visual Output**
- Displays car counter and free space count
- Shows occupied and free area names
- Provides real-time visual feedback

## 🔧 Configuration

### Parking Area Definition
The `Ajawan` file contains:
- `polylines`: Array of polygon coordinates defining parking spaces
- `area_names`: Names/labels for each parking space

### Model Configuration
- **Model Path**: `Model/yolov8s.pt`
- **Detection Classes**: Cars (from COCO dataset)
- **Confidence Threshold**: Default YOLO settings

### Video Settings
- **Frame Size**: 1020x400 pixels
- **Playback Speed**: ~33 FPS (adjustable via `cv2.waitKey()`)

## 📊 Output Information

### Visual Indicators
- **Green Polygons**: Available parking spaces
- **Red Polygons**: Occupied parking spaces
- **Blue Circles**: Detected car centers
- **Yellow Text**: System title
- **White Text**: Car counter and free space count

### Data Output
- Real-time car count
- Available parking spaces
- Occupied area names
- Free area names

## 🔌 Firebase Integration (Optional)

The system is prepared for Firebase integration:
```python
# Uncomment these lines to enable Firebase
# db.reference('/').update(firebase_data)
# real_time_data = db.reference('/').get()
```

## 🐛 Troubleshooting

### Common Issues

1. **Video File Not Found**
   ```
   Error: Could not open video file.
   ```
   - Check file path in `video_path` variable
   - Ensure video file exists and is accessible

2. **Model File Missing**
   ```
   FileNotFoundError: [Errno 2] No such file or directory: 'Model/yolov8s.pt'
   ```
   - Download YOLOv8 model or check model path

3. **IP Camera Connection Failed**
   ```
   Error: Could not open IP camera stream.
   ```
   - Verify IP camera URL
   - Check network connectivity
   - Ensure camera is online

4. **Performance Issues**
   - Reduce frame processing frequency
   - Use smaller YOLO model (yolov8n.pt)
   - Enable GPU acceleration if available

## 📈 Performance Optimization

### For Better Performance
1. **Use Smaller Model**: Replace `yolov8s.pt` with `yolov8n.pt`
2. **Reduce Frame Rate**: Increase `cv2.waitKey()` delay
3. **GPU Acceleration**: Install CUDA-compatible PyTorch
4. **Frame Skipping**: Process every nth frame

### For Better Accuracy
1. **Fine-tune Model**: Train on parking-specific dataset
2. **Adjust Confidence**: Modify YOLO confidence thresholds
3. **Improve Lighting**: Ensure good video quality
4. **Calibrate Areas**: Refine parking space polygons

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **YOLOv8**: For object detection capabilities
- **OpenCV**: For computer vision operations
- **Ultralytics**: For YOLO implementation

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code comments
3. Create an issue in the repository

---

**Note**: This system is designed for educational and research purposes. For production use, consider additional security measures and error handling.
