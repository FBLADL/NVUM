# Dataset Preparation

## NIH Chest X-ray14
- Download [NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC) and [Google NIH](https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest)
- ``` cp Google_NIH/four_findings_expert_labels/test_labels.csv NIH/ ```

## CheXpert
- Download [CXP](https://stanfordmlgroup.github.io/competitions/chexpert/)
- ``` python CXP_preprocess.py ``` and change dir accordingly 

## Indiana Open-I
- Download [OPI](https://openi.nlm.nih.gov/faq#collection), PNG and Report
- ``` python OPI_preprocess.py ``` and change dir accordingly

## PadChest
- Download [PDC](https://bimcv.cipf.es/bimcv-projects/padchest/)
- ``` python PDC_preprocess.py ``` and change dir accordingly

## ISIC2019
- Download [ISIC2019](https://challenge.isic-archive.com/data/)
- `data/isic_dataloader.py` has preprocess code, according to [Paper](https://arxiv.org/abs/2110.08866)