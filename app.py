from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import os
import uuid
import shutil
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

app = FastAPI(title="ESRGAN Image Upscaler", description="Upload an image to get an upscaled version using ESRGAN")

# Create directories for uploads and results
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def load_model():
    """Load the ESRGAN model once at startup"""
    model_path = 'models/RRDB_PSNR_x4.pth'  # or RRDB_ESRGAN_x4.pth
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Use most CPU cores but leave 1 for OS
    try:
        import multiprocessing
        torch.set_num_threads(max(1, (multiprocessing.cpu_count() or 2) - 1))
    except Exception:
        pass
    
    print(f'Model loaded from {model_path}')
    return model, device

def calculate_metrics(original_img, upscaled_img):
    """Calculate accuracy metrics between original and upscaled images"""
    # Convert to grayscale for metrics calculation
    if len(original_img.shape) == 3:
        original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original_img
        
    if len(upscaled_img.shape) == 3:
        upscaled_gray = cv2.cvtColor(upscaled_img, cv2.COLOR_BGR2GRAY)
    else:
        upscaled_gray = upscaled_img
    
    # Resize upscaled image to match original size for fair comparison
    h, w = original_gray.shape
    upscaled_resized = cv2.resize(upscaled_gray, (w, h))
    
    # Calculate metrics
    mse = mean_squared_error(original_gray, upscaled_resized)
    psnr = peak_signal_noise_ratio(original_gray, upscaled_resized)
    ssim = structural_similarity(original_gray, upscaled_resized)
    
    return {
        "mse": float(mse),
        "psnr": float(psnr),
        "ssim": float(ssim),
        "original_size": f"{w}x{h}",
        "upscaled_size": f"{upscaled_gray.shape[1]}x{upscaled_gray.shape[0]}"
    }

# Load model at startup
model, device = load_model()

@app.get("/")
def read_root():
    return {"message": "ESRGAN Image Upscaler API", "endpoints": ["/health", "/upscale", "/metrics"]}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/upscale")
async def upscale_image(file: UploadFile = File(...)):
    """
    Upload an image and get the upscaled version
    
    Args:
        file: Image file to upscale
        
    Returns:
        Upscaled image file
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        return JSONResponse(
            status_code=400, 
            content={"error": "File must be an image"}
        )
    
    # Generate unique filename
    filename = f"{uuid.uuid4()}_{file.filename}"
    
    # Process image using ESRGAN model
    try:
        # Read image (in-memory, avoid disk I/O)
        data = await file.read()
        arr = np.frombuffer(data, np.uint8)
        original_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if original_img is None:
            return JSONResponse(
                status_code=400, 
                content={"error": "Invalid image file"}
            )
        
        # Preprocess image (same as test.py)
        img = original_img.astype(np.float32) / 255.0
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0).to(device)
        
        # Run inference (faster contexts)
        if device.type == 'cuda':
            with torch.inference_mode():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        else:
            with torch.inference_mode():
                output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        
        # Postprocess output
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        upscaled_img = (output * 255.0).round().astype(np.uint8)
        
        # Calculate accuracy metrics
        metrics = calculate_metrics(original_img, upscaled_img)
        
        # Save result
        output_filename = f"{os.path.splitext(filename)[0]}_upscaled.png"
        output_path = os.path.join(RESULT_DIR, output_filename)
        # Faster PNG encode (lower compression) or switch to JPEG if desired
        cv2.imwrite(output_path, upscaled_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        # Save metrics to a JSON file
        metrics_filename = f"{os.path.splitext(filename)[0]}_metrics.json"
        metrics_path = os.path.join(RESULT_DIR, metrics_filename)
        
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Return the upscaled image with metrics in headers
        response = FileResponse(
            output_path, 
            media_type="image/png", 
            filename=output_filename
        )
        response.headers["X-MSE"] = str(metrics["mse"])
        response.headers["X-PSNR"] = str(metrics["psnr"])
        response.headers["X-SSIM"] = str(metrics["ssim"])
        response.headers["X-Original-Size"] = metrics["original_size"]
        response.headers["X-Upscaled-Size"] = metrics["upscaled_size"]
        
        return response
        
    except Exception as e:
        # Clean up on error
        return JSONResponse(
            status_code=500, 
            content={"error": f"Processing failed: {str(e)}"}
        )

@app.post("/upscale-with-metrics")
async def upscale_with_metrics(file: UploadFile = File(...)):
    """
    Upload an image and get both the upscaled version and accuracy metrics
    
    Args:
        file: Image file to upscale
        
    Returns:
        JSON with metrics and image data
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        return JSONResponse(
            status_code=400, 
            content={"error": "File must be an image"}
        )
    
    # Generate unique filename
    filename = f"{uuid.uuid4()}_{file.filename}"
    
    # Process image using ESRGAN model
    try:
        # Read image (in-memory, avoid disk I/O)
        data = await file.read()
        arr = np.frombuffer(data, np.uint8)
        original_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if original_img is None:
            return JSONResponse(
                status_code=400, 
                content={"error": "Invalid image file"}
            )
        
        # Preprocess image (same as test.py)
        img = original_img.astype(np.float32) / 255.0
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0).to(device)
        
        # Run inference
        if device.type == 'cuda':
            with torch.inference_mode():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        else:
            with torch.inference_mode():
                output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        
        # Postprocess output
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        upscaled_img = (output * 255.0).round().astype(np.uint8)
        
        # Calculate accuracy metrics
        metrics = calculate_metrics(original_img, upscaled_img)
        
        # Save result
        output_filename = f"{os.path.splitext(filename)[0]}_upscaled.png"
        output_path = os.path.join(RESULT_DIR, output_filename)
        cv2.imwrite(output_path, upscaled_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        
        return {
            "metrics": metrics,
            "output_filename": output_filename,
            "output_path": output_path,
            "message": "Image upscaled successfully. Download the image from the output_path."
        }
        
    except Exception as e:
        # Clean up on error
        return JSONResponse(
            status_code=500, 
            content={"error": f"Processing failed: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8081)
