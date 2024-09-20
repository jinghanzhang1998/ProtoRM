# Prototypical Reward Network for Data-Efficient RLHF

## About The Project
This repository contains the implementation of the paper "Prototypical Reward Network for Data-Efficient RLHF", which proposes the Proto-RM framework. Our framework enhances reward models for Reinforcement Learning from Human Feedback (RLHF) using prototypical networks. By utilizing fewer samples of human feedback effectively, Proto-RM significantly improves the adaptability and accuracy of Large Language Models (LLMs) in understanding and interpreting human preferences. The project is based on the trlX framework developed by CarperAI.

**Paper**: [Prototypical Reward Network for Data-Efficient RLHF](https://aclanthology.org/2024.acl-long.748/)

**Framework**: [trlX by CarperAI](https://github.com/CarperAI/trlx)

## Getting Started

### Prerequisites
Install necessary dependencies following the [installation guide of trlX](https://github.com/CarperAI/trlx).

### Usage
To train the Proto-RM model, navigate to the examples directory and run:
```bash
python trlx/examples/summarize_rlhf/reward_model/train_reward_model_ProtoNet_gptj.py
```

After training, to deploy the model for RLHF tasks, execute:
```bash
python trlx_gptj_text_summarization.py
```

## Citation
If you find our work useful in your research, please consider citing:
```bibtex
@inproceedings{zhang-etal-2024-prototypical,
    title = {Prototypical Reward Network for Data-Efficient RLHF},
    author = {Zhang, Jinghan and Wang, Xiting and Jin, Yiqiao and Chen, Changyu and Zhang, Xinhao and Liu, Kunpeng},
    editor = {Ku, Lun-Wei and Martins, Andre and Srikumar, Vivek},
    booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
    month = aug,
    year = {2024},
    address = {Bangkok, Thailand},
    publisher = {Association for Computational Linguistics},
    url = {https://aclanthology.org/2024.acl-long.748},
    pages = {13871--13884}
}
```