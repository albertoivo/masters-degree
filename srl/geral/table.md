# Table 1: Objectives, Problem Types and Datasets

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

# Table 2: Results, Techniques and Limitations

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