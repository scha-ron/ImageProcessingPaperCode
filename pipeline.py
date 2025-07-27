import cv2
import numpy as np
from PIL import Image
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", ".*iCCP.*")

def calculate_psnr(original, compressed):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_mse(original, compressed):
    """Calculate Mean Squared Error"""
    return np.mean((original - compressed) ** 2)

def get_file_size(filepath):
    """Get file size in bytes"""
    return os.path.getsize(filepath)

def compress_with_opencv(image_path, output_path, quality):
    """Compress image using OpenCV"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Record compression time
    start_time = time.time()
    
    # Compress with specified quality (0-100)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success = cv2.imwrite(output_path, img, encode_param)
    
    compression_time = time.time() - start_time
    
    if not success:
        return None
    
    # Read compressed image back for comparison
    compressed_img = cv2.imread(output_path)
    
    return {
        'original': img,
        'compressed': compressed_img,
        'compression_time': compression_time,
        'file_size': get_file_size(output_path)
    }

def compress_with_pillow(image_path, output_path, quality):
    """Compress image using Pillow"""
    try:
        # Open image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Record compression time
        start_time = time.time()
        
        # Save with specified quality
        img.save(output_path, 'JPEG', quality=quality, optimize=True)
        
        compression_time = time.time() - start_time
        
        # Read back for comparison
        compressed_img = Image.open(output_path)
        
        # Convert to numpy arrays for comparison
        original_array = np.array(img)
        compressed_array = np.array(compressed_img)
        
        return {
            'original': original_array,
            'compressed': compressed_array,
            'compression_time': compression_time,
            'file_size': get_file_size(output_path)
        }
    except Exception as e:
        print(f"Error processing {image_path} with Pillow: {e}")
        return None

def process_images(folder_path, output_folder):
    """Process all images in a folder and compare compression methods"""
    
    # Quality settings
    quality_settings = {
        'very_low': 10,
        'low': 30,
        'medium': 50,
        'high': 70,
        'very_high': 90
    }
    
    # Create output directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Results storage
    results = []
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(folder_path).glob(f'*{ext}'))
        image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return pd.DataFrame()
    
    print(f"Processing {len(image_files)} images from {folder_path}")
    
    for image_file in image_files:
        print(f"Processing: {image_file.name}")
        
        # Get original file size
        original_size = get_file_size(image_file)
        
        for quality_name, quality_value in quality_settings.items():
            # OpenCV compression
            opencv_output = Path(output_folder) / f"opencv_{image_file.stem}_{quality_name}.jpg"
            opencv_result = compress_with_opencv(str(image_file), str(opencv_output), quality_value)
            
            if opencv_result:
                # Calculate metrics for OpenCV
                psnr_cv = calculate_psnr(opencv_result['original'], opencv_result['compressed'])
                mse_cv = calculate_mse(opencv_result['original'], opencv_result['compressed'])
                compression_ratio_cv = original_size / opencv_result['file_size']
                
                results.append({
                    'image': image_file.name,
                    'library': 'OpenCV',
                    'quality': quality_name,
                    'quality_value': quality_value,
                    'psnr': psnr_cv,
                    'mse': mse_cv,
                    'compression_ratio': compression_ratio_cv,
                    'compression_time': opencv_result['compression_time'],
                    'original_size': original_size,
                    'compressed_size': opencv_result['file_size']
                })
            
            # Pillow compression
            pillow_output = Path(output_folder) / f"pillow_{image_file.stem}_{quality_name}.jpg"
            pillow_result = compress_with_pillow(str(image_file), str(pillow_output), quality_value)
            
            if pillow_result:
                # Calculate metrics for Pillow
                psnr_pil = calculate_psnr(pillow_result['original'], pillow_result['compressed'])
                mse_pil = calculate_mse(pillow_result['original'], pillow_result['compressed'])
                compression_ratio_pil = original_size / pillow_result['file_size']
                
                results.append({
                    'image': image_file.name,
                    'library': 'Pillow',
                    'quality': quality_name,
                    'quality_value': quality_value,
                    'psnr': psnr_pil,
                    'mse': mse_pil,
                    'compression_ratio': compression_ratio_pil,
                    'compression_time': pillow_result['compression_time'],
                    'original_size': original_size,
                    'compressed_size': pillow_result['file_size']
                })
    
    return pd.DataFrame(results)

def analyze_results(df, dataset_name):
    """Analyze and visualize results"""
    if df.empty:
        print(f"No results to analyze for {dataset_name}")
        return
    
    print(f"\n{'='*50}")
    print(f"ANALYSIS RESULTS FOR {dataset_name.upper()} DATASET")
    print(f"{'='*50}")
    
    # Group by library and quality for summary statistics
    summary = df.groupby(['library', 'quality']).agg({
        'psnr': ['mean', 'std'],
        'mse': ['mean', 'std'],
        'compression_ratio': ['mean', 'std'],
        'compression_time': ['mean', 'std']
    }).round(4)
    
    print("\nSUMMARY STATISTICS:")
    print(summary)
    
    # Overall comparison
    overall_comparison = df.groupby('library').agg({
        'psnr': 'mean',
        'mse': 'mean',
        'compression_ratio': 'mean',
        'compression_time': 'mean'
    }).round(4)
    
    print(f"\nOVERALL COMPARISON:")
    print(overall_comparison)
    
    # Find best performing library for each metric
    best_psnr = overall_comparison['psnr'].idxmax()
    best_mse = overall_comparison['mse'].idxmin()  # Lower MSE is better
    best_compression_ratio = overall_comparison['compression_ratio'].idxmax()
    best_compression_time = overall_comparison['compression_time'].idxmin()  # Lower time is better
    
    print(f"\nBEST PERFORMING LIBRARY BY METRIC:")
    print(f"Best PSNR (image quality): {best_psnr}")
    print(f"Best MSE (lower is better): {best_mse}")
    print(f"Best Compression Ratio: {best_compression_ratio}")
    print(f"Best Compression Time: {best_compression_time}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Compression Comparison - {dataset_name.title()} Dataset', fontsize=16)
    
    # PSNR comparison
    psnr_data = df.groupby(['library', 'quality'])['psnr'].mean().unstack()
    psnr_data.plot(kind='bar', ax=axes[0,0], title='PSNR by Quality Level')
    axes[0,0].set_ylabel('PSNR (dB)')
    axes[0,0].legend(title='Quality')
    
    # MSE comparison
    mse_data = df.groupby(['library', 'quality'])['mse'].mean().unstack()
    mse_data.plot(kind='bar', ax=axes[0,1], title='MSE by Quality Level')
    axes[0,1].set_ylabel('MSE')
    axes[0,1].legend(title='Quality')
    
    # Compression ratio comparison
    ratio_data = df.groupby(['library', 'quality'])['compression_ratio'].mean().unstack()
    ratio_data.plot(kind='bar', ax=axes[1,0], title='Compression Ratio by Quality Level')
    axes[1,0].set_ylabel('Compression Ratio')
    axes[1,0].legend(title='Quality')
    
    # Compression time comparison
    time_data = df.groupby(['library', 'quality'])['compression_time'].mean().unstack()
    time_data.plot(kind='bar', ax=axes[1,1], title='Compression Time by Quality Level')
    axes[1,1].set_ylabel('Time (seconds)')
    axes[1,1].legend(title='Quality')
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_compression_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return overall_comparison

def main():
    """Main function to run the compression comparison"""
    
    # Check if folders exist
    test_folder = "test"
    validation_folder = "validation"
    
    if not os.path.exists(test_folder):
        print(f"Warning: {test_folder} folder not found. Creating empty folder.")
        os.makedirs(test_folder, exist_ok=True)
    
    if not os.path.exists(validation_folder):
        print(f"Warning: {validation_folder} folder not found. Creating empty folder.")
        os.makedirs(validation_folder, exist_ok=True)
    
    # Process test dataset
    print("Processing TEST dataset...")
    test_results = process_images(test_folder, "output_test")
    test_comparison = analyze_results(test_results, "test")
    
    # Save test results
    if not test_results.empty:
        test_results.to_csv('test_compression_results.csv', index=False)
        print("Test results saved to 'test_compression_results.csv'")
    
    # Process validation dataset
    print("\nProcessing VALIDATION dataset...")
    validation_results = process_images(validation_folder, "output_validation")
    validation_comparison = analyze_results(validation_results, "validation")
    
    # Save validation results
    if not validation_results.empty:
        validation_results.to_csv('validation_compression_results.csv', index=False)
        print("Validation results saved to 'validation_compression_results.csv'")
    
    # Final recommendation
    print(f"\n{'='*60}")
    print("FINAL RECOMMENDATION")
    print(f"{'='*60}")
    
    if not test_results.empty and not validation_results.empty:
        # Combine results for overall analysis
        combined_results = pd.concat([test_results, validation_results], ignore_index=True)
        
        overall_performance = combined_results.groupby('library').agg({
            'psnr': 'mean',
            'mse': 'mean',
            'compression_ratio': 'mean',
            'compression_time': 'mean'
        }).round(4)
        
        print("Combined Performance Across Both Datasets:")
        print(overall_performance)
        
        

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import cv2
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages with:")
        print("pip install opencv-python pillow pandas matplotlib")
        exit(1)
    main()