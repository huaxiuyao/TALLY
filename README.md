# Multi-Domain Long-Tailed Learning by Augmenting Disentangled Representation

This is the official PyTorch implementation for the following paper: Multi-Domain Long-Tailed Learning by Augmenting Disentangled Representation

## Installation

```bash
conda create -n TALLY python=3.8
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html 
pip install wilds
```

## Prepare Dataset

* **Download Dataset:**

  * **VLCS:** Download from https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8

  * **PACS:**Download from  https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd

  * **OfficeHome:** Download from https://cornell.box.com/shared/static/xwsbubtcr8flqfuds5f6okqbr3z0w82t

  * **DomainNet:** Download from http://ai.bu.edu/M3SDA/

  * **TerraIncognita:** Follow https://github.com/facebookresearch/DomainBed/blob/main/domainbed/scripts/download.py

  * **iWildCam:** Auto-download when running the code

* **Generate Multi-Domain Long-Tailed Split**: Our split for subpopulation shift and domain shift on each dataset is provided in `./data`.  It is also recommand to generate your own split using code in `./data_generator`

## Run experiments

```python
python main.py --dataset Dataset --data-dir /Your/Path/to/Data --split suffix of split file 
```
