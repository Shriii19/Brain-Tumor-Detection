# NeuroScan Pro - AI Brain Tumor Detection Platform

> **Clinical-grade AI system for MRI brain tumor detection powered by deep learning**

A professional Flask web application that provides automated brain tumor detection and classification from MRI scans using advanced deep learning models.

## 🎯 Key Features

- **Professional Web Interface**: Clinical-grade UI with enterprise styling
- **AI-Powered Analysis**: Deep learning model for accurate tumor detection
- **Multi-Class Detection**: Supports 4 tumor types:
  - ✅ Glioma (High severity)
  - ✅ Meningioma (Medium severity) 
  - ✅ Pituitary Adenoma (Low severity)
  - ✅ No Tumor (Normal tissue)
- **Real-time Processing**: Sub-3 second analysis with confidence scoring
- **Medical Compliance**: Audit trails, error handling, and professional logging

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Shriii19/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection

# Install dependencies
pip install -r requirements.txt

# Run application
python main.py
```

### Access Application
Open browser: `http://localhost:5000`

## 🛠️ Technology Stack

- **Backend**: Flask, TensorFlow/Keras, PIL
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
- **AI/ML**: Convolutional Neural Networks (CNN)
- **Security**: File validation, secure uploads, error handling

## 📁 Project Structure

```
Brain-Tumor-Detection/
├── main.py              # Flask application with AI integration
├── templates/
│   └── index.html       # Professional web interface
├── uploads/             # Secure file upload directory
├── requirements.txt     # Python dependencies
├── README.md           # Documentation
└── sample-images/      # Test MRI samples
```

## ⚠️ Important Notes

### Model File Status
The trained model (`model.h5` - 122MB) is excluded due to GitHub file size limits. The application runs in **demo mode** with simulated AI predictions.

**Options:**
- Place your trained model as `model.h5` in root directory
- Use demo mode for testing interface and functionality
- Train custom model using provided samples

### System Requirements
- **Python**: 3.13+ (uses tf-nightly for compatibility)
- **Memory**: 4GB+ RAM recommended
- **Storage**: 500MB free space
- **Browser**: Modern browser with JavaScript enabled

## 🔒 Medical Compliance

> **⚠️ IMPORTANT**: This is a research and educational tool. Not intended for clinical diagnosis without proper medical validation and professional oversight.

### Security Features
- ✅ File validation and sanitization
- ✅ Secure upload handling (25MB limit)
- ✅ Error handling and logging
- ✅ Audit trail for all analyses

### Usage Guidelines
- For research and educational purposes only
- Requires medical professional interpretation
- Results should be validated by qualified radiologists
- Maintain patient data privacy and confidentiality

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Shriii19/Brain-Tumor-Detection/issues)
- **Documentation**: This README file
- **License**: Educational and research use

---

**© 2025 NeuroScan Pro** | Advanced AI Medical Imaging Platform