# Shuffled Video-based Smoke Classification Using Virtual Sample Augmentation and Transfer Learning 

This repository corresponds to our study entitled
"*Visual process monitoring of biomass conversion reactors using transfer learning and generative AI.*".


<div align="center">

![Language](https://img.shields.io/badge/language-Python-blue?&logo=python)
![Dependencies Status](https://img.shields.io/badge/dependencies-PyTorch-brightgreen.svg)
[![License](https://img.shields.io/badge/license-GPL--3.0-blue)](LICENSE)
</div>



## Requirements
- PyTorch 2.5.0
- Python 3.11.5
    - torchvision
    - torchmetrics
    - PIL
    - numpy
    - pandas
    - prettytable


## Data Information
The data comprises both real images captured from field biomass-to-biochar conversion experiments and virtual images. Additional details about the dataset can be found in our paper.

The images are stored in the `data` folder. The "Normal" (Non-smoke) and "Smoke" subfolders are the two classes of images. 

Below is the `data` folder tree structure:

<details>
  <summary>Click to show folder tree structure</summary>


```
.
├── Gen1_200
│   ├── vid1
│   │   ├── Normal
│   │   └── Smoke
│   ├── vid10
│   │   ├── Normal
│   │   └── Smoke
│   ├── vid2
│   │   ├── Normal
│   │   └── Smoke
│   ├── vid3
│   │   ├── Normal
│   │   └── Smoke
│   ├── vid4
│   │   ├── Normal
│   │   └── Smoke
│   ├── vid5
│   │   ├── Normal
│   │   └── Smoke
│   ├── vid6
│   │   ├── Normal
│   │   └── Smoke
│   ├── vid7
│   │   ├── Normal
│   │   └── Smoke
│   ├── vid8
│   │   ├── Normal
│   │   └── Smoke
│   └── vid9
│       ├── Normal
│       └── Smoke
├── Gen2_200
│   ├── vid1
│   │   ├── Normal
│   │   └── Smoke
│   ├── vid10
│   │   ├── Normal
│   │   └── Smoke
│   ├── vid2
│   │   ├── Normal
│   │   └── Smoke
│   ├── vid3
│   │   ├── Normal
│   │   └── Smoke
│   ├── vid4
│   │   ├── Normal
│   │   └── Smoke
│   ├── vid5
│   │   ├── Normal
│   │   └── Smoke
│   ├── vid6
│   │   ├── Normal
│   │   └── Smoke
│   ├── vid7
│   │   ├── Normal
│   │   └── Smoke
│   ├── vid8
│   │   ├── Normal
│   │   └── Smoke
│   └── vid9
│       ├── Normal
│       └── Smoke
└── rawImgs_200
    ├── vid1
    │   ├── Normal
    │   └── Smoke
    ├── vid10
    │   ├── Normal
    │   └── Smoke
    ├── vid2
    │   ├── Normal
    │   └── Smoke
    ├── vid3
    │   ├── Normal
    │   └── Smoke
    ├── vid4
    │   ├── Normal
    │   └── Smoke
    ├── vid5
    │   ├── Normal
    │   └── Smoke
    ├── vid6
    │   ├── Normal
    │   └── Smoke
    ├── vid7
    │   ├── Normal
    │   └── Smoke
    ├── vid8
    │   ├── Normal
    │   └── Smoke
    └── vid9
        ├── Normal
        └── Smoke

94 directories, 0 files

```
</details>


## Script Description 
- `data` folder contains the dataset images. See the above section for more details.
- `datasetCSVPrep` folder contains information about loading training dataset and testing dataset for each rotated run (10 runs in total as described in our paper). This dataset loading information are saved in the *.csv format, including the image path address and the corresponding label. An example of ten-run dataset information is shown in the `datasetCSVPrep` folder.
- `sh_beluga` folder contains job submission script for running experiments on the "Beluga" cluster in Computer Canada. Our model training utilizes the Distributed Data Parallel (DDP) strategy using multi-GPUs (e.g. four GPUs in the example).

- `src` folder contains source scripts of our shuffled video-based smoke classification. 
  - `distDataParallel.py` includes functions for the Distributed Data Parallel (DDP) training.
  - `trainingFunc_ResNet_DDP.py` defines training functions for the ResNet-50 model in DDP mode using transfer learning. 
  - `trainingFunc_ResNet_testing.py` includes functions for testing the trained ResNet-50 model (not in DDP mode).
  - `utility.py` includes the utility functions.







- `test` folder contains scripts for running and evaluating the models.
  - `test_training_ResNet_Scenario_DDP.py` is to test model training of ResNet-50 using transfer learning in DDP mode.
  - `test_testing_ResNet_Scenario_Shuffle.py` is to test trained smoke classification model (not in DDP mode).


- `trainedModels` folder contains the trained models. The trained models are saved in the `*.pth` format. You can access the dataset via Google Drive.

## Examples 
The examples can be implemented in cluster server via:
```Shell 
# test the model training in DDP
sbatch sh_beluga/ShuffledVideoBasedTraining.sh
```


## License
This repository is published under the terms of the `GNU General Public License v3.0 `. 
