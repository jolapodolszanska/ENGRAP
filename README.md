# ENGRAP

This repo is part of publication _ENGRAP: An Explainable AI Application for MRI-based Staging of Alzheimer’s Disease_. Manuscript will be  sent to _Neural Computing and Applications_ Springer Nature. I will keep you informed about the process on an ongoing basis.

## About App
This application works on the basis of two tasks that it is supposed to perform:

This app 
**Task 1:** Classification - Takes MRI input, preprocesses it (resize, normalize, augment), runs ONNX inference to get class probabilities, applies softmax ordering from Non to Moderate severity, and calculates a weighted severity score.
**Task 2:** Attribution/Heatmap - Uses Signed RISE attribution with random masks, computes signed scores, and applies normalization with Gaussian smoothing to generate explanatory heatmaps.
The system provides immediate results (top prediction, severity score), an interactive UI with asynchronous heatmap generation, and export capabilities for PNG overlays and CSV data in batch mode.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/07b7b8b9-de23-4488-8a3e-6c93877c04f0" />

This image shows a grid of 10 brain MRI scans (axial slices) with their classification results, representing an extension of the work from the publication "Leveraging Deep Q-Network Agents with Dynamic Routing Mechanisms in Convolutional Neural Networks for Enhanced and Reliable Classification of Alzheimer's Disease from MRI Scans" [1].
Each scan displays:

Predicted class (0-3, representing severity levels from normal to severe)
True class (ground truth labels)

The results demonstrate the enhanced classification capabilities achieved by integrating deep Q-network agents with dynamic routing mechanisms. Most predictions correctly match the true labels, indicating improved model reliability in classifying Alzheimer's disease progression and related brain abnormalities across different severity stages. This extension builds upon the original DQN-CNN framework to provide more robust and interpretable diagnostic capabilities for neurodegenerative conditions.

<img width="2841" height="1146" alt="image" src="https://github.com/user-attachments/assets/f6f8eab0-f216-4474-b919-ac4e9b2b9e4a" />

Below image shows a comparative analysis of three different neural network architectures for brain MRI classification, extending the work from Podolszanska, J. (2025) [1]:
CapsNet (top rows) - Shows moderate performance with some misclassifications, particularly struggling with severity level distinctions.
ResNet50 (middle rows) - Demonstrates good classification accuracy across different severity levels (0-3), with mostly correct predictions matching true labels.
ENGRAP (bottom rows) - Displays enhanced performance with distinctive heatmap visualizations showing red-yellow activation patterns that highlight disease-relevant brain regions. The attribution maps provide clear visual explanations for the classification decisions.
Each method shows predicted vs. true severity classes (0=normal, 3=severe), but ENGRAP's interpretability through color-coded attribution maps offers superior clinical utility by indicating which brain regions contribute most to the diagnostic decision, representing a significant advancement in explainable AI for neuroimaging.


<img width="1414" height="2000" alt="image" src="https://github.com/user-attachments/assets/c3e83a7b-4303-425c-88c5-295da64fa5f0" />

Screenshot shows the ENGRAP web interface for brain MRI analysis, demonstrating the practical implementation of the methodology from Podolszanska, J. (2025) [1].
Key Interface Features:

Upload area for MRI slices with support for JPG, PNG, WEBP formats (recommended 299×299, max 10MB)
Real-time classification results showing probability distributions across severity classes
Interactive controls including transparency and smoothing sliders for heatmap visualization
Asynchronous processing with the note "Heatmaps jeszcze ostateczna, aby poczekaj" (heatmaps still processing, please wait)

Classification Results:

Very Mild Demented: 99.4% (primary prediction)
Non Demented: 0.5%
Mild Demented: 0.1%
Moderate Demented: 0.0%

The interface successfully demonstrates the ENGRAP system's ability to provide immediate classification with high confidence (99.4%) while preparing interpretable attribution heatmaps, making the advanced DQN-CNN methodology accessible for clinical use.

<img width="546" height="480" alt="image" src="https://github.com/user-attachments/assets/9d42cd2a-2cc3-46ec-b421-107a14cf139c" />

Overall high accuracy across all severity levels with minimal confusion between adjacent classes
Strongest performance in moderate (Class 2) and severe (Class 3) categories
Minor confusion primarily occurs between adjacent severity levels, which is clinically reasonable
Total validation samples: 1,280 cases with excellent generalization.

# References

[1] Podolszanska, J. (2025). "Leveraging Deep Q-Network Agents with Dynamic Routing Mechanisms in Convolutional Neural Networks for Enhanced and Reliable Classification of Alzheimer's Disease from MRI Scans."

