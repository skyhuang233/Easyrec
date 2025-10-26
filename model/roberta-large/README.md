---
license: apache-2.0
language:
- en
tags:
- recommendation
- collaborative filtering
---

# EasyRec-Large

## Overview

- **Description**: EasyRec is a series of language models designed for recommendations, trained to match the textual profiles of users and items with collaborative signals.
- **Usage**: You can use EasyRec to encode user and item text embeddings based on the textual profiles that reflect their preferences for various recommendation scenarios.
- **Evaluation**: We evaluate the performance of EasyRec in: (i) Text-based zero-shot recommendation and (ii) Text-enhanced collaborative filtering.
- **Finetuned from model:** EasyRec is finetuned from [RoBERTa](https://huggingface.co/FacebookAI/roberta-large) within English.

For details please refer to our [💻[GitHub Code](https://github.com/HKUDS/EasyRec)] and [📖[Paper](https://arxiv.org/abs/2408.08821)].

## Get Started

### Environment

Please run the following  commands to create a conda environment:

```bash
conda create -y -n easyrec python=3.11
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
pip install -U "transformers==4.40.0" --upgrade
pip install accelerate==0.28.0
pip install tqdm
pip install sentencepiece==0.2.0
pip install scipy==1.9.3
pip install setproctitle
pip install sentence_transformers
```

### Example Codes
Please first download the codes.
```ssh
git clone https://github.com/HKUDS/EasyRec.git
cd EasyRec
```

Here is an example code snippet to utilize EasyRec for encoding **text embeddings** based on user and item profiles for recommendations.

```Python
import torch
from model import Easyrec
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

config = AutoConfig.from_pretrained("hkuds/easyrec-roberta-large")
model = Easyrec.from_pretrained("hkuds/easyrec-roberta-large", config=config,)
tokenizer = AutoTokenizer.from_pretrained("hkuds/easyrec-roberta-large", use_fast=False,)

profiles = [
    'This user is a basketball fan and likes to play basketball and watch NBA games.', # user
    'This basketball draws in NBA enthusiasts.', # item 1
    'This item is nice for swimming lovers.'     # item 2
]

inputs = tokenizer(profiles, padding=True, truncation=True, max_length=512, return_tensors="pt")
with torch.inference_mode():
    embeddings = model.encode(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
embeddings = F.normalize(embeddings.pooler_output.detach().float(), dim=-1)

print(embeddings[0] @ embeddings[1])    # 0.8576
print(embeddings[0] @ embeddings[2])    # 0.2171
```

### Model List
We release a series of EasyRec checkpoints with varying sizes. You can easily load these models from Hugging Face by replacing the model name.
|              Model              | Model Size | Recall@20 on Amazon-Sports |
|:-------------------------------|:--------:| :--------:|
| [hkuds/easyrec-roberta-small](https://huggingface.co/hkuds/easyrec-roberta-small) |  82M  | 0.0286 |
| [hkuds/easyrec-roberta-base](https://huggingface.co/hkuds/easyrec-roberta-base)   |  125M  | 0.0518  |
| [hkuds/easyrec-roberta-large](https://huggingface.co/hkuds/easyrec-roberta-large) |  355M  | 0.0557  |

## 🌟 Citation
If you find this work is helpful to your research, please consider citing our paper:
```bibtex
@article{ren2024easyrec,
  title={EasyRec: Simple yet Effective Language Models for Recommendation},
  author={Ren, Xubin and Huang, Chao},
  journal={arXiv preprint arXiv:2408.08821},
  year={2024}
}
```
**Thanks for your interest in our work!**