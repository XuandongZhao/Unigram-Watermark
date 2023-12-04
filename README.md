# Unigram-Watermark

This repository contains the code for the paper [Provable Robust Watermarking for AI-Generated Text](https://arxiv.org/abs/2306.17439).

**🔍 [HuggingFace Demo](https://huggingface.co/spaces/Xuandong/Unigram-Watermark)**

If you find this repository useful, please cite our paper:

```
@article{zhao2023provable,
  title={Provable Robust Watermarking for AI-Generated Text},
  author={Zhao, Xuandong and Ananth, Prabhanjan and Li, Lei and Wang, Yu-Xiang},
  journal={arXiv preprint arXiv:2306.17439},
  year={2023}
}
```

## Example

First, download the data by running the following command:

```
cd data
bash download_data.sh
```

You can generate watermarked text by running the following command:

```
python run_generate.py
```

You can detect the watermark by running the following command:

```
python run_detect.py --input_file {your_input_file}
```



## Acknowledgement

We thank the authors of the following research works and open-source projects:

[A Watermark for Large Language Models](https://github.com/jwkirchenbauer/lm-watermarking)

[Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense](https://github.com/martiansideofthemoon/ai-detection-paraphrases)
