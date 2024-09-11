# TumorInsight
The TumorTrace project seeks to create an automated system for the identification and segmentation of tumor regions in brain MRI images. Precise segmentation is essential for effective diagnosis, treatment planning, and monitoring of brain tumors. This initiative employs the U-Net architecture, a widely recognized deep learning model specifically tailored for image segmentation tasks in the medical field. Our objective is to deliver a reliable solution for automated tumor detection, significantly enhancing clinical decision-making processes.
![image](https://github.com/user-attachments/assets/a71aa2b6-1b49-487c-a697-b1075feda6c6)

# Overview
The TumorInsigght describes the classification of brain tumors using deep learning and machine learning techniques, emphasizing the availability of datasets for effective implementation. It highlights two specific datasets, Br35H and MRI images, which are essential for training models in tumor detection taken from Kaggle.
# Steps includes-
1. Data Preparation: Collect MRI images and ground truth masks; preprocess (resize, normalize).
2. Data Augmentation: Enhance dataset with rotation, flipping, scaling, and noise.
3. Dataset Splitting: Divide into training, validation, and testing sets.
4. Model Building: Implement U-Net with encoder-decoder architecture.
5. Loss Function: Select suitable loss function (e.g., Dice loss).
6. Training: Train model on training set, monitor validation performance.
7. Evaluation: Assess performance on testing set using metrics like Dice coefficient.
8. Post-processing: Refine masks with morphological operations.
9. Visualization: Overlay results on original images.
10. Deployment: Deploy for clinical use or integrate into software.

![image](https://github.com/user-attachments/assets/25353456-7c31-4c52-864d-bfb36f96f666)

# Dataset links-
Dataset Br35H 👉🏻 https://www.kaggle.com/ahmedhamada0/brain-tumor-detection

Dataset MRI Images👉🏻 https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection

# Working

https://github.com/user-attachments/assets/7bc2d64d-091c-45cf-a3ed-e636b83a6ba6





