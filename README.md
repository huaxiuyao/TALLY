# Multi-Domain Long-Tailed Learning by Augmenting Disentangled Representation

#### [paper](https://openreview.net/pdf?id=4UXJhNSbwd)

## Abstract

There is an inescapable long-tailed class-imbalance issue in many real-world classification problems. Current methods for addressing this problem only consider scenarios where all examples come from the same distribution. However, in many cases, there are multiple domains with distinct class imbalance. We study this multi-domain long-tailed learning problem and aim to produce a model that generalizes well across all classes and domains. Towards that goal, we introduce TALLY, a method that addresses this multi-domain long-tailed learning problem. Built upon a proposed selective balanced sampling strategy, TALLY achieves this by mixing the semantic representation of one example with the domain-associated nuisances of another, producing a new representation for use as data augmentation. To improve the disentanglement of semantic representations, TALLY further utilizes a domain-invariant class prototype that averages out domain-specific effects. We evaluate TALLY on several benchmarks and real-world datasets and find that it consistently outperforms other state-of-the-art methods in both subpopulation and domain shift. 

## Usage

### Environment Setup

```bash
conda create -n TALLY python=3.8
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html 
pip install wilds
```

### Dataset Preparation

* **Download Dataset:**

  * **VLCS:** Download from https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8

  * **PACS:** Download from  https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd

  * **OfficeHome:** Download from https://cornell.box.com/shared/static/xwsbubtcr8flqfuds5f6okqbr3z0w82t

  * **DomainNet:** Download from http://ai.bu.edu/M3SDA/

  * **TerraIncognita:** Follow https://github.com/facebookresearch/DomainBed/blob/main/domainbed/scripts/download.py

  * **iWildCam:** Auto-download when running the code

* **Generate Multi-Domain Long-Tailed Split**: We've provided splits for both subpopulation shift and domain shift for each dataset in the `./data` directory. We recommend downloading all datasets to this location. If you prefer, you can also create your own splits using the code in `./data_generator`.

## Run experiments

```python
python main.py --dataset {dataset_name} --data-dir {data_dir} --split {suffix_of_split_file}
```

Examples on **VLCS:**

```
python main.py --dataset VLCS --data-dir ./data --split sub
python main.py --dataset VLCS --data-dir ./data --split SUN09
python main.py --dataset VLCS --data-dir ./data --split VOC2007
python main.py --dataset VLCS --data-dir ./data --split Caltech101
python main.py --dataset VLCS --data-dir ./data --split LabelMe
```

## Citation

If TALLY is useful or relevant to your research, please kindly recognize our contributions by citing our paper:

```
@article{
yang2023multidomain,
title={Multi-Domain Long-Tailed Learning by Augmenting Disentangled Representations},
author={Xinyu Yang and Huaxiu Yao and Allan Zhou and Chelsea Finn},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=4UXJhNSbwd},
note={}
}
```

## Acknowledgments

We thank Pang Wei Koh, Yoonho Lee, Sara Beery, and members of the IRIS lab for the many insightful discussions and helpful feedback.
