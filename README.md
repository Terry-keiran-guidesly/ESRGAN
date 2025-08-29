# ESRGAN Image Upscaler API

A FastAPI-based web service that uses ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks) to upscale images with accuracy metrics.

## Features

- 🚀 **FastAPI Web Service**: RESTful API for image upscaling
- 🎯 **ESRGAN Model**: State-of-the-art image super-resolution
- 📊 **Accuracy Metrics**: PSNR, SSIM, and MSE calculations
- 🔄 **Real-time Processing**: Upload and get upscaled images instantly
- 📈 **Performance Monitoring**: Built-in health checks and metrics

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch
- FastAPI
- OpenCV
- scikit-image

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ESRGAN.git
cd ESRGAN
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download ESRGAN models**
Place your ESRGAN model files in the `models/` directory:
- `RRDB_PSNR_x4.pth` (for PSNR optimization)
- `RRDB_ESRGAN_x4.pth` (for perceptual quality)

### Running the API

```bash
python app.py
```

The server will start at `http://localhost:8081`

## API Endpoints

### Health Check
```bash
GET /health
```
Returns server status and model loading information.

### Image Upscaling
```bash
POST /upscale
```
Upload an image and receive the upscaled version with metrics in HTTP headers.

### Image Upscaling with Metrics
```bash
POST /upscale-with-metrics
```
Upload an image and receive detailed JSON response with accuracy metrics.

## Usage Examples

### Using curl
```bash
# Basic upscaling
curl -X POST "http://localhost:8081/upscale" \
  -F "file=@your_image.jpg" \
  -o upscaled_image.png

# With metrics
curl -X POST "http://localhost:8081/upscale-with-metrics" \
  -F "file=@your_image.jpg"
```

### Using Python
```python
import requests

# Upload and upscale image
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8081/upscale-with-metrics',
        files={'file': f}
    )
    
result = response.json()
print(f"PSNR: {result['metrics']['psnr']}")
print(f"SSIM: {result['metrics']['ssim']}")
```

## Accuracy Metrics

The API calculates three key metrics to evaluate upscaling quality:

- **MSE (Mean Squared Error)**: Lower values indicate better quality
- **PSNR (Peak Signal-to-Noise Ratio)**: Higher values indicate better quality (typically 20-40 dB)
- **SSIM (Structural Similarity Index)**: Values between 0-1, where 1 is perfect similarity

## Model Configuration

The API uses the RRDB (Residual in Residual Dense Block) architecture:
- **Input channels**: 3 (RGB)
- **Output channels**: 3 (RGB)
- **Feature channels**: 64
- **Block count**: 23
- **Growth channels**: 32

## File Structure

```
ESRGAN/
├── app.py                 # FastAPI application
├── requirements.txt       # Python dependencies
├── test.py               # Batch processing script
├── RRDBNet_arch.py       # ESRGAN model architecture
├── models/               # Pre-trained model files
├── uploads/              # Temporary upload directory
├── results/              # Output directory
└── LR/                   # Sample low-resolution images
```

## Performance

- **CPU Processing**: Compatible with all systems
- **GPU Processing**: 5-20x faster (requires CUDA-enabled PyTorch)
- **Memory Usage**: ~2-4GB RAM for processing
- **Supported Formats**: JPG, PNG, BMP, TIFF

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original ESRGAN paper: [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219)
- PyTorch implementation based on the official ESRGAN repository

## Support

For issues and questions, please open an issue on GitHub or contact the maintainers.
