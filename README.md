# JPEGCompressionStudy

This repository contains the code and artifacts for the term paper:
**Exploring the Effectiveness of Compressing High-Quality Images for Web Design Purposes (SS2025).**

The project investigates whether high-resolution images can be effectively compressed for web design, focusing on **lossy JPEG compression**. Two open-source libraries, **OpenCV** and **Pillow**, were compared in terms of:

* **Image quality** (PSNR, MSE)
* **Compression ratio**
* **Compression time**

## Contents

* `pipeline.py` – Python script implementing the compression pipeline.
* `test_compression_results.csv` – Collected results from experiments.
* `test_compression_comparison.png` – Visualization of test dataset results.
* `validation_compression_comparison.png` – Visualization of validation dataset results.


## Data
This [Google Drive Folder](https://drive.google.com/drive/folders/18SkgbHKvsbzoPhcyMYz4xC9M-ex0nN3Y?usp=sharing) contains images from the training set of the LIU4K-v2 dataset, used for the experiment carried out in the work required for the
IPCV course in SS 2025.

Dataset authors: Jing Liu, Dong Liu, Weiyao Yang, Shiqi Xia, Xiaolin Zhang, Yuchao Dai  
Dataset source: https://structpku.github.io/LIU4K_Dataset/LIU4K_v2.html  
Training dataset drive link: https://drive.google.com/drive/folders/1FtVQtY2t_ecuy_gzJqZ-CatqrJBAdq_d  

License: CC0 (public domain)

For more information, please refer to the dataset paper:

**Jing Liu, Dong Liu, Weiyao Yang, Shiqi Xia, Xiaolin Zhang, and Yuchao Dai**,  
"A Comprehensive Benchmark for Single Image Compression Artifact Reduction,"  
*IEEE Transactions on Image Processing*, vol. 29, pp. 7845–7860, 2020.

These images are shared and used in accordance with the dataset's public domain license. Full citation is also provided in the work that has used them.




