# DVSM Overview

## Description:  
DVSM is a decoder-only transformer for novel-view synthesis. The model takes user input images and reconstructs the scene as KV cache (the key-value pairs from attention layers). The user can then perform arbitrary novel-view synthesis within that scene. This model is intended for researchers and developers working on computer vision and graphics who need to generate novel views of 3D scenes from sparse input images.
_This model is for research and development only._  

### License/Terms of Use:
NSCLv1
### Deployment Geography:  
Global


### Use Case:  
Researchers and developers working on computer vision and graphics who need to generate novel views of 3D scenes from sparse input images. The model supports view synthesis tasks on datasets like DL3DV and Re10k, enabling applications in virtual reality, augmented reality, and 3D content creation.  
  

### Release Date:
**HuggingFace:** 07/31/2026 via https://huggingface.co/nvidia/DVSM.  

  

## Reference(s):
[Technical report](https://arxiv.org/abs/2605.29891)  

## Model Architecture:   
**Architecture Type:** Transformer   
**Network Architecture:** Each block consists of a internal-view attention and a cross-view attention layer, each followed by a MLP layer. We stack 12 blocks with 768 hidden dimensions.  
**Number of model parameters:** 163M  

  
 
## Input:  
**Input Type(s):** Image  
**Input Format(s):** RGB (Red, Green, Blue)  
**Input Parameters:** Two-Dimensional (2D)  
**Other Properties Related to Input**: Image resolutions ranging from 340p to 960p are supported. Cameras are normalized and embedded as plucker ray maps. Linear patchifiers are applied with 8x8 patch size.
  

## Output:  
**Output Type(s):** Image  
**Output Format**: RGB (Red, Green, Blue)
**Output Parameters:** Two-Dimensional (2D)  
**Other Properties Related to Output:** The output is in the same resolution as the input images.  
   

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration:
**Runtime Engine(s):** Transformers    
**Supported Hardware Microarchitecture Compatibility:**
* NVIDIA Ampere
* NVIDIA Blackwell
* NVIDIA Hopper
* NVIDIA Lovelace

**Supported Operating System(s):** Linux  

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.  

To integrate DVSM, start from the NVIDIA PyTorch container (nvcr.io/nvidia/pytorch:25.10-py3) and install the required Python packages including PyTorch, Transformers, Diffusers, Accelerate, and additional libraries such as OpenCV, Einops, XFormers, and others listed in the requirements. Then mount the dataset and checkpoint directories, and launch training or inference using the provided SLURM scripts or direct python calls.  

## Model Version(s): 
DVSM  
 
## Training, Testing, and Evaluation Datasets:  


## Training Dataset:

**Data Modality:** Image  

**Image Training Data Size:** 1 Million to 1 Billion Images  
**Data Collection Method by dataset:** Automatic/Sensors  
**Labeling Method by dataset:** Hybrid: Automated, Manually-Labeled  
**Properties (Quantity, Dataset Descriptions, Sensor(s)):**
DL3DV: 10K image sequences capturing wide variety of indoor and outdoor scenes.
Re10K: 65K image sequences capturing indoor scenes.

### Testing Dataset:

**Data Collection Method by dataset:** Automatic/Sensors  
**Labeling Method by dataset:** Hybrid: Automated, Manually-Labeled  
**Properties (Quantity, Dataset Descriptions, Sensor(s)):**
DL3DV: 140 image sequences capturing wide variety of indoor and outdoor scenes.  
Re10K: 5K image sequences capturing indoor scenes.  

### Evaluation Dataset:

| Benchmark | Metric | Score |
| --- | --- | --- |
| DL3DV 64 views 960p | PSNR | 29.71 |
| DL3DV 32 views 960p | PSNR | 27.96 |
| DL3DV 16 views 960p | PSNR | 25.88 |
| Re10K 2 views 512x512 | PSNR | 30.17 |
| MipNerf360 64 views | PSNR | 24.76 |
| Free 64 views | PSNR | 25.57 |
| Hike 64 views | PSNR | 23.10 |   

**Data Collection Method by dataset:** Automatic/Sensors  
**Labeling Method by dataset:** Hybrid: Automated, Manually-Labeled  
**Properties (Quantity, Dataset Descriptions, Sensor(s)):**
DL3DV: 140 image sequences capturing wide variety of indoor and outdoor scenes.  
Re10K: 5k image sequences capturing indoor scenes.  
MipNerf360: 9 image sequences capturing indoor and outdoor scenes where camera exposure are fixed and motion blur is minimized.  
Free: 7 image sequences capturing indoor and outdoor scenes with default camera setting. Some frames exhibit motion blur.  
Hike: 6 image sequences capturing hiking trail.  



## Inference:
**Acceleration Engine:** PyTorch
**Test Hardware:**  NVIDIA Ampere (A100 80 GB)

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail).

**Generated by NVIDIA Model Card Generator Toolkit.**

