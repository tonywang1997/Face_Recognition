# Project Title: Enhancing Face Recognition Robustness with Random Erasing Data Augmentation

## Team Name: Coda Area

### Members:
- Yang Zhao
- Zihao Wang
- Yihui Chen

## Summary
This project aims to improve the robustness and generalization of face recognition models under occlusions and varied conditions by incorporating the Random Erasing data augmentation technique. The goal is to systematically evaluate the impact of this augmentation on the model's performance, focusing on improving accuracy and reducing overfitting in face recognition tasks.

## Key Related Work
1. Zhong et al.'s "Random Erasing Data Augmentation," which introduces a novel method for training CNNs that improves robustness to occlusion.
2. Taigman et al.'s "DeepFace: Closing the Gap to Human-Level Performance in Face Verification," showcasing advances in deep learning for face recognition tasks.
3. Sun et al.'s "DeepID3: Face Recognition with Very Deep Neural Networks," which explores deep learning architectures for improved face recognition accuracy.
4. Schroff et al.'s "FaceNet: A Unified Embedding for Face Recognition and Clustering," presenting a method that directly learns a mapping from face images to a compact Euclidean space.
5. Deng et al.'s "Imagenet: A large-scale hierarchical image database," providing foundational datasets for training deep learning models in various recognition tasks, including face recognition.

## Final Goals & Evaluation
- **Goal:** To integrate Random Erasing into the training pipeline of face recognition models, aiming to enhance their robustness to partial occlusions and improve generalization across different datasets.
- **Evaluation:** The effectiveness of the proposed method will be quantitatively evaluated by comparing the face recognition accuracy, precision, recall, and F1-score before and after applying Random Erasing. We will use widely recognized benchmarks such as the Labeled Faces in the Wild (LFW) and YouTube Faces (YTF) datasets for this purpose.
- **Success Metric:** Achieving a statistically significant improvement in accuracy and F1-score on the LFW and YTF datasets, demonstrating the model's enhanced robustness to occlusions and variations in facial appearance.

## References
- Labelled Faces in the Wild (LFW) Dataset: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset
- Yale Face Database: https://www.kaggle.com/datasets/olgabelitskaya/yale-face-database

## Milestone by April 12th
- Complete the integration of Random Erasing with a baseline face recognition model and conduct preliminary experiments on a subset of the LFW dataset to assess its impact.

## Data & Technical Requirements
- **Datasets:** Labeled Faces in the Wild (LFW), YouTube Faces (YTF).
- **Software Libraries:** PyTorch for model implementation and training, OpenCV for image preprocessing.
- **Technical Needs:** Access to GPUs for model training and evaluation, and software for statistical analysis of results.

## Goals for Each Teammate
### Phase 1: Setup and Preliminary Implementation (Weeks 1-2)
- **Teammate 1 & 2 (Collaborative Task):**
  - **Task:** Implement the Random Erasing algorithm within a small, standalone Python script to ensure its functionality and effectiveness in augmenting images. This task requires understanding the algorithm from the original paper and translating it into code that can be integrated into existing data pipelines.
  - **Outcome:** A Python function or class that can take an image as input and return a version with random erasing applied.
- **Teammate 3:**
  - **Task:** Conduct a literature review to identify suitable face recognition models for the project, considering factors such as accuracy, computational efficiency, and ease of integration with the Random Erasing augmentation.
  - **Outcome:** A report summarizing the pros and cons of at least three face recognition models, with a recommendation on which model to proceed with.

### Phase 2: Data Preparation and Model Integration (Weeks 3-4)
- **Teammate 1:**
  - **Task:** Integrate the Random Erasing code into the data preprocessing pipeline for the chosen face recognition model, ensuring that images are correctly augmented before being fed into the model for training.
  - **Outcome:** A data loader that applies Random Erasing augmentation dynamically to each batch of images during model training.
- **Teammate 2 & 3 (Collaborative Task):**
  - **Task:** Download and preprocess the Labeled Faces in the Wild (LFW) and YouTube Faces (YTF) datasets, setting them up for training and evaluation. This includes splitting the datasets into training, validation, and test sets.
  - **Outcome:** Preprocessed and split datasets ready for use, with a simple script to load data for training and evaluation.

### Phase 3: Model Training and Initial Evaluation (Weeks 5-6)
- **Teammate 2 & 3 (Collaborative Task):**
  - **Task:** Train the chosen face recognition model on the LFW dataset with and without Random Erasing augmentation, adjusting hyperparameters as necessary to optimize performance.
  - **Outcome:** A trained face recognition model, with preliminary performance metrics (accuracy, precision, recall, F1-score) on the validation set.
- **Teammate 1:**
  - **Task:** Begin drafting the project report, focusing on the methodology section that details the Random Erasing implementation, model selection rationale, and data preprocessing steps.
  - **Outcome:** An initial draft of the methodology section of the project report.

### Phase 4: Comprehensive Evaluation and Reporting (Weeks 7-8)
- **Teammate 1 & 2 (Collaborative Task):**
  - **Task:** Extend the evaluation to the YouTube Faces (YTF) dataset to assess the model's generalization ability. Fine-tune the model as needed based on the results.
  - **Outcome:** Finalized model performance metrics on both LFW and YTF datasets, along with insights into the model's generalization capabilities.
- **Teammate 3:**
  - **Task:** Finalize the project report by incorporating results, discussions, and conclusions. Create the presentation slides summarizing the project's objectives, methodology, results, and implications.
  - **Outcome:** A comprehensive project report and presentation ready for submission/delivery.
