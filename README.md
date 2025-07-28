# Brain Tumor Detection Using Deep Learning and Computer Vision

A Flask web application for detecting brain tumors in MRI images using deep learning models. This project provides an intuitive web interface for uploading MRI scans and getting automated tumor classification results.

## Features

- **Web-based Interface**: Easy-to-use Flask web application
- **MRI Image Upload**: Support for uploading and processing MRI scan images
- **Tumor Classification**: Automated detection and classification of brain tumors
- **Multiple Tumor Types**: Supports detection of:
  - Glioma
  - Meningioma
  - Pituitary tumors
  - No tumor (healthy brain)
- **Real-time Results**: Instant prediction results with confidence scores

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: TensorFlow/Keras
- **Image Processing**: PIL (Python Imaging Library)
- **Frontend**: HTML/CSS with responsive design
- **Deep Learning**: Convolutional Neural Networks (CNN)

## Project Structure

```
Brain Tumor Detection/
├── main.py                 # Flask application main file
├── model.h5               # Pre-trained deep learning model
├── templates/
│   └── index.html         # Web interface template
├── uploads/               # Directory for uploaded images (auto-created)
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── sample images/        # Test MRI images
    ├── Te-gl_0015.jpg    # Glioma sample
    ├── Te-meTr_0001.jpg  # Meningioma sample
    ├── Te-noTr_0004.jpg  # No tumor sample
    └── Te-piTr_0003.jpg  # Pituitary tumor sample
```

## Installation & Setup

### Prerequisites
- Python 3.13 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Shriii19/Brain-Tumor-Detection.git
   cd Brain-Tumor-Detection
   ```

2. **Install required packages**
   ```bash
   pip install flask tensorflow pillow numpy
   ```
   
   Or install from requirements file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Model File Setup**
   
   ⚠️ **Important**: The `model.h5` file is not included in this repository due to GitHub's file size limits (122MB > 100MB limit).
   
   **Options:**
   - **Option A**: If you have the model file, place it in the root directory as `model.h5`
   - **Option B**: The app will run in demo mode without the model (showing placeholder predictions)
   - **Option C**: Train your own model using the included Jupyter notebook

4. **Run the application**
   ```bash
   python main.py
   ```

5. **Access the web interface**
   Open your browser and navigate to: `http://127.0.0.1:5000`

## Usage

1. **Start the Application**: Run `python main.py` to start the Flask server
2. **Upload MRI Image**: Use the web interface to upload an MRI scan image
3. **Get Results**: View the prediction results showing:
   - Tumor type classification
   - Confidence percentage
   - Uploaded image preview

## Current Status

✅ **Working Components:**
- Flask web server
- File upload functionality
- Web interface
- Image processing pipeline
- Sample MRI images
- Complete documentation

⚠️ **Known Issues:**
- Model file (`model.h5`) not included due to GitHub file size limits
- Currently running in demo mode (shows "Model not loaded - demo mode")
- Predictions show placeholder results until model is added

## Model Information

- **File**: `model.h5` (122MB - excluded from repository)
- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 128x128 pixels
- **Image Format**: Supports common formats (JPG, PNG, etc.)
- **Classes**: 4 categories (Glioma, Meningioma, Pituitary, No Tumor)
- **Status**: Compatible with older TensorFlow versions (may need updating for current TF versions)

## Troubleshooting

### Common Issues

1. **Model Loading Error**: If you encounter model loading issues, ensure you have the correct TensorFlow version or retrain the model with current libraries.

2. **Dependencies**: Make sure all required packages are installed:
   ```bash
   pip install flask tensorflow pillow numpy
   ```

3. **Port Already in Use**: If port 5000 is busy, modify the port in `main.py`:
   ```python
   app.run(debug=True, port=5001)
   ```

## Development Notes

- The application runs in debug mode for development
- Uploaded files are stored in the `uploads/` directory
- The model expects 128x128 pixel images
- Images are automatically normalized for processing

## Future Enhancements

- [ ] Fix model compatibility issues
- [ ] Add batch processing capability
- [ ] Implement result history
- [ ] Add data visualization features
- [ ] Enhance UI/UX design
- [ ] Add API endpoints for programmatic access

## License

This project is for educational and research purposes.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the project.

---

**Note**: This is a medical imaging application intended for research and educational purposes only. It should not be used for actual medical diagnosis without proper validation and medical professional oversight.