# ENGRAP

This repo is part of publication _ENGRAP: An Explainable AI Application for MRI-based Staging of Alzheimer’s, Neural Computing and Applications, Springer Nature, 2026.
## About App
This application works on the basis of two tasks that it is supposed to perform:

This app 
**Task 1:** Classification - Takes MRI input, preprocesses it (resize, normalize, augment), runs ONNX inference to get class probabilities, applies softmax ordering from Non to Moderate severity, and calculates a weighted severity score.
**Task 2:** Attribution/Heatmap - Uses Signed RISE attribution with random masks, computes signed scores, and applies normalization with Gaussian smoothing to generate explanatory heatmaps.
The system provides immediate results (top prediction, severity score), an interactive UI with asynchronous heatmap generation, and export capabilities for PNG overlays and CSV data in batch mode.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/07b7b8b9-de23-4488-8a3e-6c93877c04f0" />

Overview of the proposed HybridCNN architecture for 4-class Alzheimer’s disease classification using 2D MRI slices from the ADNI dataset.  
The model combines ResNet50-based feature extraction with capsule representations and Transformer encoder layers, followed by late-fusion classification. The diagram also summarizes the training configuration, optimization settings, evaluation metrics, and approximate parameter counts for individual model components.


<img width="680" height="740" alt="image" src="https://github.com/user-attachments/assets/f9e55470-9070-459e-a62f-71ea15dc2006" />

Overview of the proposed Signed RISE attribution pipeline applied to the HybridCNN model for explainable Alzheimer’s disease classification.  
The framework consists of three stages: randomized mask generation, forward inference through the HybridCNN architecture, and accumulation and normalization of signed attribution maps. The diagram summarizes the masking strategy, Transformer-based inference process, attribution normalization, rendering procedure, and computational configuration used for Grad-CAM-style interpretability analysis.

<img width="680" height="740" alt="image" src="https://github.com/user-attachments/assets/9bf47738-42aa-45c2-9139-4b5d731c9890" />

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

Example Grad-CAM visualization for a correctly classified NonDemented case.  
The MRI slice is overlaid with AAL3v1-based anatomical regions, and the accompanying bar chart shows relative Grad-CAM attention within selected AD-relevant regions. In this example, the strongest attention is observed in temporal regions, while posterior regions such as the precuneus and posterior cingulate show lower activation.

<img width="1545" height="637" alt="image" src="https://github.com/user-attachments/assets/e1e55ec1-aef1-4335-9379-18062a577be5" />


# References

[1] World Health Organization. Dementia. WHO Fact Sheet. Accessed March 2025
(2025). https://www.who.int/news-room/fact-sheets/detail/dementia

[2] Wong, W., Jiang, Y., Chen, H., Jiang, Y., Shi, M., Wu, Z., Zhang, W. The
global macroeconomic burden of Alzheimer’s disease and other dementias. The
Lancet Global Health 12(9), 1476–1487 (2024). https://doi.org/10.1016/S2214-
109X(24)00264-X

[3] Rahman, M.M., Ahmed, S.T., Hossain, M.A., Rahman, M.M. Early detection of
Alzheimer’s disease: A review of machine learning techniques. Diagnostics 13(22),
3428 (2023). https://doi.org/10.3390/diagnostics13223428

[4] Lee, G., Nho, K., Kang, B., Sohn, K.-A., Kim, D. Identification of Alzheimer’s
disease using a CNN model based on T1-weighted MRI. Scientific Reports 10(1),
22252 (2020). https://doi.org/10.1038/s41598-020-79243-9

[5] Gharibi, H., Rezaei, M., Ferdowsi, S. An efficient method for early Alzheimer’s
disease detection using deep CNNs. Frontiers in Artificial Intelligence 8, 1563016
(2025). https://doi.org/10.3389/frai.2025.1563016

[6] Salahuddin, Z., Woodruff, H.C., Chatterjee, A., Lambin, P. Explainable AI in
medical imaging: Saliency-based XAI approaches. European Journal of Radiology
162, 110787 (2023). https://doi.org/10.1016/j.ejrad.2023.110787

[7] Velden, B.H.M., Kuijf, H.J., Gilhuijs, K.G.A., Viergever, M.A. Explaining
explainability: The role of XAI in medical imaging. European Journal of
Radiology 172, 111339 (2024). https://doi.org/10.1016/j.ejrad.2024.111339

[8] Nohara, Y., Matsumoto, K., Soejima, H., Nakashima, N. Evaluating XAI
techniques in chest radiology imaging. PLOS ONE 19(10), e0308758 (2024).
https://doi.org/10.1371/journal.pone.0308758

[9] Alzheimer’s Association. 2024 Alzheimer’s disease facts and figures. Alzheimer’s
& Dementia 20(5), 3708–3821 (2024). https://doi.org/10.1002/alz.13809

[10] Aksoy, S., Daou, A. An explainable web-based diagnostic system for Alzheimer’s
disease using XRAI. bioRxiv (2025). https://doi.org/10.1101/2025.08.16.670652

[11] Aksoy, S., Demircioglu, P., Bogrekci, I. A web-deployed explainable AI system for brain tumor diagnosis. Neurology International 17(8), 121 (2025).
https://doi.org/10.3390/neurolint17080121

[12] Li, Z., Dib, O. Empowering brain tumor diagnosis through explainable deep
learning. Machine Learning and Knowledge Extraction 6(4), 2248–2281 (2024).
https://doi.org/10.3390/make6040111

[13] Buga, R., Buzea, C.G., Agop, M., Ochiuz, L., Vasincu, D., Popa, O.,
Rusu, D.I., Stirban, I., Eva, L. Streamlit application and deep learning model for brain metastasis monitoring. Biomedicines 13(2), 423 (2025).
https://doi.org/10.3390/biomedicines13020423

[14] Santhosh, T.R.S., Mohanty, S.N., Pradhan, N.R., Khan, T., Derbali, M. Neurovision: A deep learning driven web application for brain tumour detection. Digital
Health (2025). https://doi.org/10.1177/20552076251333195

[15] Verdú-Díaz, J., Bolano-Díaz, C., et al. Myo-guide: A machine learning-based web
application for neuromuscular disease diagnosis. Journal of Cachexia, Sarcopenia
and Muscle (2025). https://doi.org/10.1002/jcsm.13815

[16] Aksoy, S. SeruNet: A unified multi-modal AI system for neurological disorder
detection. IJFMR 7(4) (2025). Article ID: IJFMR250452891

[17] Alp, S., Akan, T., Bhuiyan, M.S., Disbrow, E.A., Conrad, S.A., Vanchiere, J.A.,
Kevil, C.G., Bhuiyan, M.A.N. Joint transformer architecture in brain 3D MRI
classification. Scientific Reports 14, 8996 (2024). https://doi.org/10.1038/s41598-
024-59578-3

[18] Dessain, Q., Delinte, N., Hanseeuw, B., Dricot, L., Macq, B. Leveraging Swin
Transformer for enhanced diagnosis of Alzheimer’s disease. arXiv preprint (2025).
arXiv:2507.09996

[19] Petersen, R.C., Aisen, P.S., Beckett, L.A., Donohue, M.C., Gamst, A.C., Harvey,
D.J., Jack Jr., C.R., Jagust, W.J., Shaw, L.M., Toga, A.W., Trojanowski, J.Q.,
Weiner, M.W. ADNI: Clinical characterization. Neurology 74(3), 201–209 (2010).
https://doi.org/10.1212/WNL.0b013e3181cb3e25

[20] Podolszanska, J. (2025, February). CapTrAD: A Hybrid Model for Alzheimer’s Disease Classification. 
In International Conference on Agents and Artificial Intelligence (pp. 208-229). Cham: Springer Nature Switzerland.

[21] Folego, G., Weiler, M., Casseb, R.F., Pires, R., Rocha, A. Alzheimer’s disease
detection through whole-brain 3D-CNN MRI. Frontiers in Bioengineering and
Biotechnology 8, 534592 (2020). https://doi.org/10.3389/fbioe.2020.534592

[22] Sabour, S., Frosst, N., Hinton, G.E. Dynamic routing between capsules. NeurIPS
30, 3856–3866 (2017).

[23] Rolls, E.T., Huang, C., Lin, C., Feng, J., Joliot, M. Automated anatomical
labelling atlas 3. NeuroImage 206, 116189 (2020).

[24] Fan, L., Li, H., Zhuo, J., Zhang, Y., Wang, J., Chen, L., Yang, Z., Chu, C., Xie, S.,
Laird, A.R., Fox, P.T., Eickhoff, S.B., Yu, C., Jiang, T. The human brainnetome
atlas. Cerebral Cortex 26(8), 3508–3526 (2016).

[25] F. H. Saif, M. N. Al-Andoli, and W. M. Y. W. Bejuri, "Explainable AI for
Alzheimer Detection: A Review of Current Methods and Applications," Applied
Sciences, vol. 14, no. 22, p. 10121, 2024. doi: 10.3390/app142210121

[26] D. Muhammad and M. Bendechache, "Unveiling the Black Box: A Systematic
Review of Explainable Artificial Intelligence in Medical Image Analysis," Computational and Structural Biotechnology Journal, vol. 24, pp. 542–560, 2024. doi:
10.1016/j.csbj.2024.08.005

[27] S. Alp et al., "Joint Transformer Architecture in Brain 3D MRI Classification:
Its Application in Alzheimer’s Disease Classification," Scientific Reports, vol. 14,
p. 8996, 2024. doi: 10.1038/s41598-024-59578-3

