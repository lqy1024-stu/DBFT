# DBFT
Mitigating Fine-tuning Bias: A Parameter-Efficient Debiasing Framework for Large Language Models

data —— train set

test —— test set

biased data —— biased train set

eval —— evaluation function

Fine-tuning based on Lora：https://github.com/hiyouga/LLaMA-Factory

# The result under different θ, DBFT is insensitive to θ.
Note that we reprt the results of mixed bias in NLI task and the results on the lexical overlap subset of HANS.
| Bias type | MNLI | HANS | ANLI | BBQ  | Unqover |
|-------|--------|--------|--------|--------|--------|
| θ=0.5   | 91.71  | 92.73  | 78.34  | 97.90  | 90.00 |
| θ=0.6   | 92.00  | 93.61  | 79.22  | 98.00  | 93.23 |
| θ=0.7   | 92.04  | 93.00  | 78.24  | 97.88  | 92.89 |
| θ=0.8   | 92.18  | 93.22  | 78.33  | 97.80  | 91.01 |
| θ=0.9   | 92.30  | 93.50  | 78.56  | 97.85  | 90.75 |

# The results of DBFT on both large (Qwen2.5-14B) and small (Qwen2-1.5B) models, confirming its robustness.
Note that we reprt the results of mixed bias in NLI task and the results on the lexical overlap subset of HANS. 
| Qwen2-1.5B | MNLI | HANS | ANLI | BBQ  | Unqover |
|-------|--------|--------|--------|--------|--------|
| BASE   | 71.93  |  69.59 | 75.52  | 34.20  | 16.14 |
| LoRA   | 91.14  | 84.91  | 70.30  | 83.65  | 58.53 |
| $\Delta$ LoRA   | 85.45  | 83.64  | 80.96  | 38.70  | 22.18  |
| Ext-sub   | 90.72  | 82.15  | 65.91  | 79.75  | 62.92 |
| $DBFT_{prom}$   | 91.17  | 86.87  | 76.61  | 82.75  | 67.80  |
| $DBFT_{data}$   | 90.83  | 85.58  | 75.91  | 83.10  | 61.27  |

| Qwen2.5-14B | MNLI | HANS | ANLI | BBQ  | Unqover | 
|-------|--------|--------|--------|--------|--------|
| BASE   |   |   |   |   |  |
| LoRA   |   |   |   |   |  |
| $\Delta$ LoRA   |   |   |   |   |  |
| Ext-sub   |   |   |   |   |  |
| $DBFT_{prom}$   |   |   |   |   |  |
| $DBFT_{data}$   |   |   |   |   |  |

# The results of reviewer-suggested baselines.
Note that: 1) PEFTDebias requires full fine-tuning models on downstream tasks. To avoid this costly process, we combine CDA with PEFT on the upstream tasks and omit the fine-tuning step on the downstream task. 2) SOD requires full fine-tuning models and only focuses on mitigating position bias. ADEPT relies on neutral and attribute token tuples, limiting it to single social stereotypes (e.g., gender, age). AIM-Fair, designed for the image domain with full fine-tuning (feasible for ResNet-18 but impractical for Qwen2-7B). Thus they are excluded for fairness and feasibility reasons. Besides, we reprt the results of mixed bias in NLI task and the results on the lexical overlap subset of HANS.
| Qwen2-7B | MNLI | HANS | ANLI | BBQ  | Unqover | General News  | Unbiased News |
|-------|--------|--------|--------|--------|--------|--------|--------|
| BASE   | 84.10  | 87.86  | 75.39  | 76.35  | 65.70 | 51.35  | 13.49 |
| LoRA   | 92.41  | 86.49  | 73.96  | 97.95  | 90.80 | 56.14  | 17.22 |
| $\Delta$ LoRA   | 89.33  | 91.33  | 78.43  | 80.60  | 59.33 | 48.62  | 18.16 |
| Ext-sub   | 91.38  | 89.51  | 78.30  | 97.85  | 91.38 | 49.26  | 18.39 |
| PEFTDebias   | 84.30  | 85.69  | 74.82  | 92.06  | 87.60  | 50.26  | 14.12 |
| ADEPT   | /  | /  | /  | 95.23  | 90.54  | /  | / |
| $DBFT_{prom}$   | 92.00  | 93.61  | 79.22  | 98.00  | 93.23 | 55.24  | 20.15 |
| $DBFT_{data}$   | 92.34  | 95.23  | 79.09  | 99.20  | 95.65 | /  | / |
