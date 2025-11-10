<h1 align="center">TD-CAG</h1>

This repository contains code to reproduce results from the paper:


TD-CAG: A Curvature-Aware and Translation-Dislocation Framework for Improving Adversarial Transferability


## Requirements

+ Python >= 3.8.1
+ numpy == 1.19.2
+ Tensorflow == 2.5.0
+ scikit-image==0.18.0
+ opencv-python >= 4.10.0
+ scipy ==1.6.3
+ pandas == 1.2.4
+ imageio >= 2.33.1
+ tf-slim == 1.1.0
+ tensorflow-addons==0.14.0

## Qucik Start

### Prepare the data and models

You should download [data](https://drive.google.com/drive/folders/1CfobY6i8BfqfWPHL31FKFDipNjqWwAhS) and [models](https://drive.google.com/drive/folders/10cFNVEhLpCatwECA6SPB-2g0q5zZyfaw).  
Place them in dev_data/ and models/, respectively.

### TD-CAG

File td-cag.py and bsr-td-cag.py generate adversarial examples on inception_v3 model.  
File td-bga-dtsm.py generate adversarial examples on ensemble model.
If you want to attack other models, replace the model in `graph` function and load such models in `main` function.

#### Runing attack

Taking TD-CAG attack for example, you can run this attack as following:

```
CUDA_VISIBLE_DEVICES=gpuid python td-cag.py
```

#### Evaluating the attack

The generated adversarial examples would be stored in directory `./outputs_td-cag`. Then run the file `simple_eval.py` to evaluate the success rate of each model used in the paper:

```
CUDA_VISIBLE_DEVICES=gpuid python simple_eval.py
```

## Acknowledgments

Codes refers to [SI-NI-FGSM](https://github.com/JHL-HUST/SI-NI-FGSM) and [BSR](https://github.com/Trustworthy-AI-Group/BSR)

