# CausalIntersectionalityDualGradient
## Summary
This repository contains the codes for the paper titled as `Causal Intersectionality and Dual Form of Gradient Descent for Multimodal Analysis: A Case Study on Hateful Memes`.
- [arXiv](https://arxiv.org/abs/2308.11585)
- [LREC-COLING 2024 (oral) version](https://aclanthology.org/2024.lrec-main.259/)

# How to Reproduce the Result
## Summary
| Result | How to Reproduce |
| --- | --- |
| Figure 3-5 | miATE / MIDAS |
| Table 5 | miATE vs MIDAS |
| Table 6,7 | meta-optimization |

## Preparation
- Use Google Drive to place data/token.txt on MyDrive/vilio
- For LLM experiment, clone this repo and `pip install -r llm_requirements.txt`
## miATE / MIDAS
- Run `notebook/vilio_gradient.ipynb` 
## miATE vs MIDAS
- Run `notebook/intersectioanlity_tables.ipynb`

## LLM
### BLIP-2 image captioning
- Run `script/image2text.py`
```
python blip2/image2text.py \
    --input_dir <path-to-Hateful-Memes> \
    --output_dir <path-to-BLIP2-Captions>
```
### BLIP-2 and LLaMA
- Run `notebook/hf_llama2.ipynb`
### meta-optimization
- Run `notebook/intersectioanlity_tables.ipynb`
# Appendix
## BLIP-2 and BERT
- Run `script/bert_midas.py`
```
python blip2/bert_midas.py \
    --memes_dir <path-to-Hateful-Memes> \
    --caption_dir <path-to-BLIP2-Captions> \
    --output_dir <path-to-output> \
    --exp_name test --eval_set test_seen --train_set train,dev_seen \
    --random_seed <seeds from 1987 to 1991>
```

# Citation
```
@misc{2308.11585,
    Author = {Yosuke Miyanishi and Minh Le Nguyen},
    Title = {Causal Intersectionality and Dual Form of Gradient Descent for Multimodal Analysis: a Case Study on Hateful Memes},
    Year = {2023},
    Eprint = {arXiv:2308.11585},
}
@inproceedings{miyanishi-nguyen-2024-causal,
    title = "Causal Intersectionality and Dual Form of Gradient Descent for Multimodal Analysis: A Case Study on Hateful Memes",
    author = "Miyanishi, Yosuke  and
      Nguyen, Minh Le",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.259",
    pages = "2901--2916",
    abstract = "Amidst the rapid expansion of Machine Learning (ML) and Large Language Models (LLMs), understanding the semantics within their mechanisms is vital. Causal analyses define semantics, while gradient-based methods are essential to eXplainable AI (XAI), interpreting the model{'}s {`}black box{'}. Integrating these, we investigate how a model{'}s mechanisms reveal its causal effect on evidence-based decision-making. Research indicates intersectionality - the combined impact of an individual{'}s demographics - can be framed as an Average Treatment Effect (ATE). This paper demonstrates that hateful meme detection can be viewed as an ATE estimation using intersectionality principles, and summarized gradient-based attention scores highlight distinct behaviors of three Transformer models. We further reveal that LLM Llama-2 can discern the intersectional aspects of the detection through in-context learning and that the learning process could be explained via meta-gradient, a secondary form of gradient. In conclusion, this work furthers the dialogue on Causality and XAI. Our code is available online (see External Resources section).",
}
```
