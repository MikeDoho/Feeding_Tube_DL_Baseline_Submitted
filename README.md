# Feeding_Tube_DL_Baseline_Submitted

**General Comments**
Included is the baseline code used to train and evaluate our deep learning feeding tube model. The clinical model was implemented outside our server and therefore only an old implement is found.


**Abstract**

DOI: 10.1016/j.radonc.2022.04.016

Purpose/objectives: Radiation therapy (RT) for the treatment of patients with head and neck cancer (HNC) leads to side effects that can limit a person's oral intake. Early identification of patients who need aggressive nutrition supplementation via a feeding tube (FT) could improve outcomes. We hypothesize that traditional machine learning techniques used in combination with deep learning techniques could identify patients early during RT who will later need a FT.

Materials/methods: We evaluated 271 patients with HNC treated with RT. Sixteen clinical features, planning computed tomography (CT) scans, 3-dimensional dose, and treatment cone-beam CT scans were gathered for each patient. The outcome predicted was the need for a FT or ≥10% weight loss during RT. Three conventional classifiers, including logistic regression (LR), support vector machine, and multilayer perceptron, used the 16 clinical features for clinical modeling. A convolutional neural network (CNN) analyzed the imaging data. Five-fold cross validation was performed. The area under the curve (AUC) values were used to compare models' performances. ROC analyses were performed using a paired DeLong Test in R-4.1.2. The clinical and imaging model outcomes were combined to make a final prediction via evidential reasoning rule-based fusion.

Results: The LR model performed the best on the clinical dataset (AUC 0.69). The MedicalNet CNN trained via transfer learning performed the best on the imaging dataset (AUC 0.73). The combined clinical and image-based model obtained an AUC of 0.75. The combined model was statistically better than the clinical model alone (p = 0.001).

Conclusions: An artificial intelligence model incorporating clinical parameters, dose distributions and on-treatment CBCT achieved the highest performance to identify the need to place a reactive feeding tube.
