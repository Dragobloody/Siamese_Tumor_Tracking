#  Deep Siamese Real-Time Tumor Tracking for Lung Cancer Patients

Here we provide a brief explanation on how to run our pipeline.

1. Run the generate_patient_data.ipynb file in order to save your dicom planning CT scans as numpy vectors.
2. Run the data_deformation.ipynb file in order to generate new pCT scans using the Diffeomorphic registration.
3. Run the generate_drrs.ipynb file in order to generate the DRRs training and testing datasets.
4. (Optional) If you want to apply the GAN-bases translation strategy then run the translate_drrs.ipynb file.
5. You can train the proposed siamese model and the chosen baselines by running train_siamese.ipynb, train_mrcnn.ipynb and train_semseg.ipynb, respectively.
6. Same idea from 5 but for testing the trained models: run test_siamese.ipynb, test_mrcnn.ipynb and test_semseg.ipynb, respectively.
