# Mitigating Hallucinations in Image Captioning via Frequency Perturbation Entropy

<p align="center">
    <img src="img/backbone.png" width="100%"> <br>
    The framework of Image captioning with Frequency Perturbation Entropy. This framework consists of three main components: a image encoder, a text generator, and a perturbation modeler.
</p>

In this paper, we study the cause of hallucinations from frequency domain and attribute them to the model’s susceptibility to the frequency perturbations, which manifesting as the generation of hallucinations with high uncertainty. Based on this finding, we introduce frequency perturbations into models to reduce sensitivity and propose a novel metric Frequency Perturbation Entropy (FPE) to reduce the language generation uncertainty when conditioned on these perturbations, thereby mitigating hallucinations. 

#### NEWS：
- 2025.11.14: The FPE paper is under review.
- 2025.11.14: The code has been released.
- 2025.11.01: The FPE's code repository was built.

## 1. Environment setup

Please refer to [meshed-memory-transformer](https://github.com/aimagelab/meshed-memory-transformer).


## 2. Prepare Data

You need to download images from each dataset and locate the images within data folder whose structure should be as follows:
```bash
data
│─ cc3m
│    │─ train_list.txt                   # cc3m training annotation
│    │─ val_list.txt                     # cc3m validation annotation
│    │─ images                           # link to root directory for images
│         │─ train                       # link to directory for training images
│         │─ val                         # link to directory for validation images
│
│─ coco
│    │─ captions_train2017.json          # training annotation
│    │─ captions_val2017.json            # validation annotation
│    │─ captions_karpathy_test.json      # Download on your own. Please check the scripts/download_mscoco.sh
│    │─ images                           # link to root directory for images
│         │─ val2017                     # link to directory for validation images
│         │─ val2014                     # link to directory for karpathy test split
│
│─ nocap
│    │─ val_4500_captions.json           # validation annotation
│    │─ images                           # link to root directory for images
│         │─ validation                  # link to directory for validation images
```
## 3. Evaluation
* Captioning evaluation score are calculate by [coco-caption](https://github.com/tylin/coco-caption) tool.
* Hallucination evaluation score are calculate by [POPE](https://github.com/AoiDragon/POPE/tree/main) and [CHAIR](https://arxiv.org/pdf/1809.02156)

## 4. Training
We train FPE (based GRIT) on 4 GPU A100 (80GB) in DDP mode by:
```python
export DATA_ROOT=/gemini/data-1/COCO2014
python train_caption.py exp.name=caption_caption_FPE_GRIT_FINETUNE  \
    model.detector.checkpoint=/gemini/pretrain/region_ckpt.pth \
    optimizer.batch_size=32 \
    optimizer.num_workers=2 \
    exp.ngpus_per_node=4 \
    exp.world_size=4 \
    model.freq_net.visual_type=ViT \
    model.freq_net.gamma=0.2 \
    model.freq_net.visualization=False \
    dataset.overfit=False 
```

## 5. Performance

<p align="center">
    <img src="img/performance.png" width="55%"> <br>
    Performance on hallucination and in-domain captioning for base methods with FPE.
</p>

## 6. Visualization

<p align="center">
    <img src="img/visualization1.png" width="50%"> <br>
    The evolution of frequency perturbation when generating specific caption words during training. In each epoch, the perturbation modeler receives the low- and high-frequency maps and generated word sequences to model the frequency perturbation.
</p>
<p align="center">
    <img src="img/visualization2.png" width="90%"> <br>
    Visualization of frequency perturbation at each step of caption generation. (a) The inferencing results of GRIT with FPE. (b) The inferencing results of LLaVA-v1.5 with FPE.
</p>


## 7. Citation
This paper is currently under review.

## 8. Contact for Issues
Please feel free to contact with [panlz_bit@hotmail.com](panlz_bit@hotmail.com)

## 9. License
This project is licensed under the terms of the [MIT License](./LICENSE).
