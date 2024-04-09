<div align="center">
<h1>LOCR: Location-Guided Transformer for Optical Character Recognition</h1>

[![Paper](https://img.shields.io/badge/Paper-arxiv.2403.02127-white)](https://arxiv.org/abs/2403.02127)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Community%20Space-blue)](https://huggingface.co/Merry-bee/LOCR/tree/main)

</div>

This is the official repository for LOCR, the academic document PDF parser that guided by text location.


## Install

From repository:
```
git clone https://github.com/Merry-bee/LOCR.git
```

### Download the model

You need to download the `pytorch_model.bin` from Hugging Face: https://huggingface.co/Merry-bee/LOCR/tree/main.

The checkpoints directory structure can look as follows:

```
checkpoints/
├── config.json
├── pytorch_model.bin
├── special_tokens_map.json
├── tokenizer_config.json
└── tokenizer.json
```

## Get prediction for a PDF

To get predictions for a PDF run 

```source predict_prompt.sh```

```
bash file: python predict_prompt.py pdf

environ:
  decay                 decay_weight, 1 as no decay.

arguments:
  pdf                   PDF(s) to process. Either a pdf file or a txt file containing paths of pdf files.

options:
  --batchsize           Batch size to use.
  --checkpoint          Path to checkpoint directory.
  --out                 Output directory.
  --recompute           Recompute already computed PDF, discarding previous predictions.
  --return_attention    True when prediction.
  --interaction         Whether to turn on human-interactive mode.
```

## Training with your own dataset

To train or fine tune a LOCR model, run 

```
python train_prompt.py --config config/train_LOCR.yaml
```

If you do not want to use wandb, run

```
python train_prompt.py --config config/train_LOCR.yaml --debug
```

### Prepare a dataset

To generate a dataset you need

1. A training dataset and a validation dataset with `.jsonl` format.
2. An image directory containing the images.
3. Each jsonl file contains json lines, with each line as a dict of `{'image':$image_path,'prompt':$list_of_bbox,'pretext':$list_of_tokens}`, where `$image_path` is the relative path to your image directory.
4. Run `locr.dataset.gen_seek.py` to generate `.seek.map` files.


The data directory structure can look as follows:

```
dataset/
├── images
├── train.jsonl
├── train.seek.map
├── validation.jsonl
└── validation.seek.map
```

## Evaluation

Run 

```
source test.sh
```

## Citation

```
@ARTICLE{2024arXiv240302127S,
       author = {{Sun}, Yu and {Zhou}, Dongzhan and {Lin}, Chen and {He}, Conghui and {Ouyang}, Wanli and {Zhong}, Han-Sen},
        title = "{LOCR: Location-Guided Transformer for Optical Character Recognition}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Artificial Intelligence, Computer Science - Computation and Language},
         year = 2024,
        month = mar,
          eid = {arXiv:2403.02127},
        pages = {arXiv:2403.02127},
          doi = {10.48550/arXiv.2403.02127},
archivePrefix = {arXiv},
       eprint = {2403.02127},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240302127S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```

## Acknowledgments

This repository builds on top of the [Donut](https://github.com/clovaai/donut/) and [Nougat](https://github.com/facebookresearch/nougat) repository.

## License

LOCR codebase is licensed under apache-2.0.
