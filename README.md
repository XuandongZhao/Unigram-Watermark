# GPTWatermark

This repository contains the code for the paper [Provable Robust Watermarking for AI-Generated Text](https://arxiv.org/abs/2306.17439).

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

This work is based on the following amazing research works and open-source projects, thanks a lot to all the authors for sharing!


[A Watermark for Large Language Models](https://github.com/jwkirchenbauer/lm-watermarking)

```
@article{kirchenbauer2023watermark,
title={A watermark for large language models},
author={Kirchenbauer, John and Geiping, Jonas and Wen, Yuxin and Katz, Jonathan and Miers, Ian and Goldstein, Tom},
journal={arXiv preprint arXiv:2301.10226},
year={2023}
}
```

[Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense](https://github.com/martiansideofthemoon/ai-detection-paraphrases)

```
@article{krishna2023paraphrasing,
  title={Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense},
  author={Krishna, Kalpesh and Song, Yixiao and Karpinska, Marzena and Wieting, John and Iyyer, Mohit},
  journal={arXiv preprint arXiv:2303.13408},
  year={2023}
}
```
