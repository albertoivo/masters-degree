# 3. Results

## 3.1. Study Selection and Temporal Distribution
A total of 11 studies were analyzed in this systematic review. The temporal distribution of the included literature demonstrates a recent and rapidly accelerating interest in the application of artificial intelligence to ionospheric monitoring, with all articles published between 2022 and 2025. Specifically, the distribution of the publications is as follows: 2022 (2 studies), 2023 (3 studies), 2024 (1 study), and 2025 (5 studies). While the primary focus of the corpus was the detection and analysis of Equatorial Plasma Bubbles (EPBs) (9 studies), two studies investigating related upper atmospheric phenomena—Medium-Scale Traveling Ionospheric Disturbances (MSTIDs) and OH* airglow turbulence—were included due to their analogous application of machine learning to ionospheric imagery. 

## 3.2. Summary of Included Studies
An overview of the 11 studies included in this systematic review is presented in Tables 1 and 2. Table 1 summarizes key aspects such as the primary physical objective, the specific computational problem addressed, and the origin and type of datasets utilized. In complement, Table 2 details the specific machine learning techniques applied, the evaluation metrics used to assess performance, the main quantitative results achieved, and the primary limitations reported by the authors of each study.

### Table 1: Objectives, Problem Types and Datasets

| ID | Author | Year | Objective | Problem Type | Dataset |
| :--- | :--- | :--- | :--- | :--- | :--- |
| P1 | Githio et al. | 2024 | Predict EPB zonal drift velocities. | Prediction / Regression | ASI & GNSS (Brazil, '13–'17), HWM-14, OMNI db. |
| P2 | Sedlak et al. | 2023 | Identify "dynamic" episodes (e.g., turbulence) in OH* airglow data for energy dissipation estimation. | Classification | FAIM 3 SWIR camera nocturnal images (Germany). |
| P3 | Zhong et al. | 2025 | DL program for EPB detection, segmentation, and feature extraction (latitude, zonal drift). | Detection & Segmentation | OI 630 nm ASI (Qujing, China, '12-'22). |
| P4 | Yacoub et al. | 2025 | Low-cost XAI method for real-time EPB detection & classification. | Detection / Classification | Raw & Radon-transformed ASI (Brazil). |
| P5 | Thanakulketsarat et al. | 2023 | Real-time plasma bubble detection & classification via VHF radar. | Classification | VHF radar QL plots (Thailand). |
| P6 | Srisamoodkham et al. | 2022 | CNN for EPB detection in ASI, integrated into real-time space weather website. | Detection | ASI (OMTI) from Thailand, tested on AUS/BRA/HI data. |
| P7 | Siddiqui | 2025 | Identify EPBs in airglow imagery using 2DPCA & XAI. | Classification / Detection | 630 nm airglow ASI. |
| P8 | Thanakulketsarat et al. | 2023 | EPB image detection using hybrid learning to study characteristics. | Detection / Classification | VHF radar QL plots (Thailand). |
| P9 | Liu et al. | 2022 | Real-time DL to detect MSTIDs in dTEC maps & build database. | Segmentation / Detection | ~1.2M dTEC maps (GEONET, Japan, '97-'19). |
| P10 | Okoh et al. | 2025 | Optimize automated EPB detection on ASI via bootstrapping CNN. | Classification / Detection | Raw ASI (Nigeria, OMTI, '15-'20). |
| P11 | Zhong et al. | 2025 | ML-based feature extraction for statistical profile of EPB evolution over a solar cycle. | Stat. Analysis / Feature Extraction (ML) | OI 630 nm airglow (Qujing, China, '12-'22). |

### Table 2: Results, Techniques and Limitations

| ID | Result | ML Technique | Metric | Limitation |
| :--- | :--- | :--- | :--- | :--- |
| P1 | r = 0.98 (train), 0.96 (val); RMSE = 10.61 & 10.06 m/s. | Random Forest (RF) | r, RMSE | Sensitive to foF2 inaccuracy; trained on high solar activity, tested on moderate; restricted to 23-05 UT. |
| P2 | mAP = 0.82; AP = 0.63 ("dynamic"). 13 turbulence episodes extracted. | Temporal Convolutional Network (TCN) | Prec., Rec., AP, mAP, Acc., CE Loss, Conf. Mat. | Confusion in calm vs dynamic transitions; single image training is sub-optimal. |
| P3 | ResNet18+CBAM: 99% Prec., 91% Rec., 95% F1; DeepLab: 88.2% Acc. & 78.2% IoU. | Deep CNN (ResNet18+CBAM, DeepLabV3Plus) | Prec., Rec., F1, IoU, Acc., mIoU, mAcc, aAcc. | Accuracy drops in severe weather; generalizability favors quiet geomagnetic periods. |
| P4 | 98.17% Acc. (Raw), 97.35% Acc. (Radon). Model size reduced 70%. | 2DPCA + RFE + Random Forest + XAI | Acc., Prec., Rec., F1, Train. Time | Sensitive to cloudy skies causing misclassification. |
| P5 | CNN-SVM (RBF kernel) achieved 93.67% Acc. | SVM + CNN + SVD | Acc., TPR, FPR, AUC-ROC, Conf. Mat. | >7 CNN layers yield marginal accuracy gain but high latency. |
| P6 | YOLOv3 detected EPBs. Best anomaly threshold: 0.40. | YOLOv3-tiny (CNN) | Anomaly threshold % | YOLOv3-tiny has lower accuracy than standard YOLOv3, but faster processing. |
| P7 | SHAP/LIME showed vertical extent & width of dark plume impact classification. | 2DPCA + SVM/RF/NN + SHAP/LIME (XAI) | Acc., Prec., Rec., F1 | Varies by instrument/location; struggles with severe contamination; reactive detection. |
| P8 | CNN-SVM (RBF kernel) achieved 93.08% Acc. | CNN + SVM (RBF/Polynomial kernel) | Accuracy | N/A |
| P9 | Mask R-CNN achieved ~80% Acc. at 8 fps (real-time). | Mask R-CNN + FPN | IoU, F1, mAP, ROI Loss, FPS | Fails for small structures; manual annotation subjectivity limits near 100% accuracy. |
| P10 | Ensemble improved Acc. to 99.17% (mean) and 99.33% (mode). | Bootstrapping CNN ensemble | Acc., Conf. Mat. | Mean aggregation sensitive to overconfidence; mode voting needs fallback for ambiguous cases. |
| P11 | High solar activity promotes EPB area, PMLE & velocity. ROF drops/rises near midnight. | Deep CNN (DCNN) | r, p-val, std. dev. | Aggregation over 19-07 LT hides interval differences; airglow doesn't give full causal mechanism. |

As outlined in the tables, optical imagery—specifically All-Sky Imagers (ASI)—is the most frequently utilized data modality across the analyzed corpus. Methodologically, the application of Deep Convolutional Neural Network (DCNN) architectures constitutes the most prevalent approach for detection and segmentation tasks, while a smaller subset of studies employs traditional machine learning algorithms, such as Support Vector Machines (SVM) or Random Forests, often in conjunction with dimensionality reduction techniques.

## 3.3. Types of Problems Addressed
The reviewed studies addressed a variety of computational and analytical problems, often combining multiple approaches within a single framework. Detection was the most prevalent task (8 studies), frequently paired with Classification (6 studies) to automatically categorize the presence, absence, or specific type of ionospheric irregularities. Other addressed problems included Instance Segmentation (2 studies) for delineating the morphological boundaries of plasma structures, Prediction/Regression (1 study) for forecasting the zonal drift velocities of EPBs, and Statistical Analysis/Feature Extraction (1 study) utilizing ML-derived outputs to establish long-term characteristic databases.

## 3.4. Data Types and Sources
The predominant data type utilized across the analyzed studies consisted of optical imagery, specifically All-Sky Imager (ASI) data capturing nighttime airglow emissions (e.g., OI 630.0 nm and OH*). Other significant data sources included Very High Frequency (VHF) radar quick-look (QL) plot images and Global Navigation Satellite System (GNSS) observations, such as Detrended Total Electron Content (dTEC) maps and Rate of TEC Index (ROTI) data.

## 3.5. Machine Learning Techniques
Deep learning architectures and traditional machine learning algorithms were both extensively utilized. Convolutional Neural Networks (CNNs) and their variants (e.g., ResNet18, ResNet101, Mask R-CNN, YOLO v3 tiny, DeepLabV3Plus, and Temporal Convolutional Networks) were the most widely implemented architectures, particularly for image-based detection and segmentation tasks. Traditional machine learning algorithms were also frequently applied, with Support Vector Machines (SVM) and Random Forest (RF) being the most prominent. Several studies emphasized dimensionality reduction and feature extraction using Two-Dimensional Principal Component Analysis (2DPCA) and Singular Value Decomposition (SVD) prior to classification. Furthermore, recent literature incorporated Explainable Artificial Intelligence (XAI) frameworks (e.g., SHAP, LIME) and ensemble learning techniques (e.g., bootstrapping, probability averaging, and mode voting).

## 3.6. Evaluation Metrics
To evaluate model performance, the most commonly used metrics for classification and detection tasks were Accuracy, Precision, Recall (Sensitivity), and the F1-score. For morphological segmentation and bounding-box detection, Intersection over Union (IoU) and Mean/Average Precision (mAP / AP) were standard. Regression and prediction models primarily utilized the Correlation Coefficient (r) and Root Mean Square Error (RMSE). Computational efficiency—measured in processing time, frames per second (fps), and elapsed training time—was also a reported metric in studies prioritizing real-time operational deployment.

## 3.7. Main Results Reported
The main results reported in the analyzed literature demonstrate the high efficacy of machine learning in automating the monitoring of ionospheric irregularities. Classification and detection models consistently reported high performance, with baseline accuracies generally exceeding 86%. CNN-based models, including hybrid CNN-SVM architectures and ensemble bootstrapping CNNs, achieved prediction accuracies ranging from 93% to over 99%. Segmentation networks, such as ResNet18 combined with DeepLabV3Plus, reached precision rates of 99% and IoU scores of 78.23% for the morphological extraction of EPBs. 

Studies focusing on model optimization found that integrating dimensionality reduction techniques, like 2DPCA and Radon transforms, significantly decreased trained model size (up to a 70% reduction) and accelerated training times without compromising detection accuracy (maintaining levels around 97% to 98%). The application of XAI successfully identified that the vertical extent and plume width of the central dark band were the primary physical features influencing automated EPB classification. Finally, the automated extraction of EPB features enabled the generation of comprehensive, solar-cycle-scale statistical profiles. These ML-derived profiles revealed that high solar activity promotes EPB area, poleward latitudinal extension, and zonal drift velocity, while geomagnetic disturbances tend to suppress these specific characteristics.