# Zero-Shot Anomaly Detection using Vision Foundation Models
This project explores how recent vision foundation models such as DINOv3, the mirroring DINO model and/or SAM can be applied for zero-shot and few-shot anomaly detection on industrial data. 

Students will use pre-trained models to detect surface defects and irregular textures in the MVTec Anomaly Detection (AD) dataset without additional training — by comparing patch-level embeddings between normal and anomalous samples.

Optional extensions include testing fine-tuning or prompt-based feature adaptation for improved results.


**Key task:**
* Use DINOv3 to extract embeddings or mirroring DINO and SAM for extra help.
* Compare normal vs. defective samples using similarity metrics.
* Visualize and interpret anomaly maps.
* Evaluate performance on the MVTec AD dataset.


**Students will learn to:**  
* Apply large pre-trained vision models for industrial inspection.
* Use zero-shot and few-shot learning strategies.
* Perform embedding-based similarity and visualization in PyTorch.
* Evaluate unsupervised anomaly detection results.

Data: 
MVTec AD dataset (publicly available; already preprocessed for easy use).

This project will be supervised by Emil Hovad (emilh@dtu.dk) and Murat Kulachi.