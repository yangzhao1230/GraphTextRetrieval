
# GraphTextRetrieval

## 1. Checkpoints Preprocessing
You should download the folder from <https://huggingface.co/allenai/scibert_scivocab_uncased> (Pre-trained Sci-Bert), put it into the project and rename it as "bert_pretrained". Then you should put "MoMu-S" and "MoMu-K" into the folder "all_checkpoints".

## 2. Zeroshot Testing
#### zeroshot testing on our datasets with paragraph-level:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --data_type 0 --if_test 2 --if_zeroshot 1 --pth-test data/phy_data/test
```
#### zeroshot testing on our datasets with sentence-level:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --data_type 1 --if_test 2 --if_zeroshot 1 --pth-test data/phy_data/test
```
#### zeroshot testing on PCdes with paragraph-level:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --data_type 0 --if_test 2 --if_zeroshot 1 --pth-test data/kv_data/test
```
#### zeroshot testing on our datasets with sentence-level:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --data_type 1 --if_test 2 --if_zeroshot 1 --pth-test data/kv_data/test
```
## 3. Model Finetuning
#### finetuning on our datasets with paragraph-level and testing:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune.pt --data_type 0 --if_test 0 --if_zeroshot 0 --pth-test data/phy_data/test
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune.pt --data_type 0 --if_test 2 --if_zeroshot 0 --pth-test data/phy_data/test
```
#### finetuning on our datasets with sentence-level and testing:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune.pt --data_type 1 --if_test 0 --if_zeroshot 0 --pth-test data/phy_data/test
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune.pt --data_type 1 --if_test 2 --if_zeroshot 0 --pth-test data/phy_data/test
```
#### finetuning on PCdes with paragraph-level and testing:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune.pt --data_type 0 --if_test 0 --if_zeroshot 0 --pth-test data/kv_data/test
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune.pt --data_type 0 --if_test 2 --if_zeroshot 0 --pth-test data/kv_data/test
```
#### finetuning on PCdes with sentence-level and testing:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune.pt --data_type 1 --if_test 0 --if_zeroshot 0 --pth-test data/kv_data/test
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune.pt --data_type 1 --if_test 2 --if_zeroshot 0 --pth-test data/kv_data/test
```

## 4. Citation
Please refer to our paper:

Su B, Du D, Yang Z, et al. A Molecular Multimodal Foundation Model Associating Molecule Graphs with Natural Language[J]. arXiv preprint arXiv:2209.05481, 2022.

```
https://arxiv.org/abs/2209.05481
```
```
@article{su2022molecular,
  title={A Molecular Multimodal Foundation **Model** Associating Molecule Graphs with Natural Language},
  author={Su, Bing and Du, Dazhao and Yang, Zhao and Zhou, Yujie and Li, Jiangmeng and Rao, Anyi and Sun, Hao and Lu, Zhiwu and Wen, Ji-Rong},
  journal={arXiv preprint arXiv:2209.05481},
  year={2022}
}
```
