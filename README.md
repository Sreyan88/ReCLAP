# **ReCLAP: Improving Zero Shot Audio Classification by Describing Sounds**
<p align="center"><img src="https://github.com/Sreyan88/ReCLAP/blob/main/assets/reclap.png" alt="GAMA Logo." width="300"/></p>

This is the official implementation of our paper [ReCLAP: Improving Zero Shot Audio Classification by Describing Sounds](https://arxiv.org/abs/2406.11768).


Training data, and checkpoints can be downloaded from [ReCLAP's Google Drive](https://drive.google.com/drive/folders/1ZUf3HNo8wO2Ec6_cfQ0nc1fUknkHSP9e?usp=sharing).

---

### Setup 🏋️
```shell
conda create -n reclap python=3.10
conda activate reclap
cd ReCLAP
pip install -r requirements.txt
```

### Training 🏃‍♂️
Update the path to training csv (--train-data) validation csv (--val-data) and pretrained HTSAT checkpoint (--pretrained-audio) in `run.sh` file.
The csvs can be downloaded from [ReCLAP's Google Drive](https://drive.google.com/drive/folders/1ZUf3HNo8wO2Ec6_cfQ0nc1fUknkHSP9e?usp=sharing).

To run the training:

```
cd ReCLAP/train/src/laion_clap
sh run.sh
```

The checkpoint for CLAP 2.3M (tiny) and ReCLAP Base can be donwloaded from [ReCLAP's Google Drive](https://drive.google.com/drive/folders/1ZUf3HNo8wO2Ec6_cfQ0nc1fUknkHSP9e?usp=sharing).
The CLAP 2.3M (base) will be uploaded soon.
### Prompt Augmentations
More details on prompt augemntation can be found in [`prompts`](https://github.com/Sreyan88/ReCLAP/tree/main/prompts).

### Acknowledgement 🌻
We would like to thank the authors of [LAION-CLAP](https://arxiv.org/abs/2211.06687) for open-sourcing their code, which inspired our work.

