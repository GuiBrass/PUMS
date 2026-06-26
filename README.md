# Soft-robotic project

This repository contains a series of ML experiments on PUMS finger (surface estimation, surface classification, and tool estimation) and on PUMS gripper (object recognition and width prediction) for the paper: Proprioception with high spatiotemporal accuracy from unstructured multichannel sensing in soft robots.

# Experiments performed

PUMS finger

1.  Surface Estimation
    Estimating the surface geometry pressed on a soft finger.
    A) training set reduction
    B) sensor reduction

2.  Surface Classification
    Classifying the surface (35 classes in 4 positions) used in the surface estimation experiment

3.  Tool Estimation (or Tool position and force estimation in the paper)
    Estimating the position and force of a tool mounted on the tip of the PUMS finger.
    A) training set reduction
    B) sensor reduction

PUMS gripper

1.  Object Recognition (or Object identification in the paper)
    Identify which object the gripper is holding. The dataset is made of 9 classes (8 objects + empty grasp).
    A) training set reduction
    B) sensor reduction
    C) flip fingers

2.  Width Prediction (or Object length estimation in the paper)
    Predicting the width of the grasped object (same shape) while varying the angle of grasp.
    A) training set reduction
    B) sensor reduction
    C) flip fingers

Principal component analysis (PCA)
A) PCA of the input matrices on each dataset.
B) A remocal of the N last components was also investigated on the surface estimation dataset.

# Installation steps

1. Clone the repo
   git clone <repository-url>
   cd PUMS_code
2. Create and activate a virtual environment
   python -m venv sr_env
   source sr_env/bin/activate # Linux/macOS
   sr_env\Scripts\activate # Windows
3. Install dependencies
   pip install -r requirements.txt
4. Prepare the datasets and trained models
   Download the datasets and trained models
   Make sure they are correctly placed (see next section)

# Folders

PUMS_code/
│
├── CADs/ # .stl of all 3D printed objects and molds in the paper.
│
├── experiments/ # Code for all 4 ML experiments (plus the surface estimation converted into a surface classification) and PCA
│
├── figures/ # Saved plots, visualizations, and outputs
│
├── PCBs_pdf/ # PDFs (schematics and PCB layers) of all the boards used for PUMS.
│
├── nogit_dataset/ # Add the datasets (not tracked on Git, get the data here: https://zenodo.org/uploads/20838108)
│
├── nogit_NN/ # Trained Neural Network models (not tracked on Git, create a folder here with this name)
│
├── results/ # Result data used to preduce figures.
│
└── README.md # Project documentation
