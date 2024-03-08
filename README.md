# Large Language Model per Dataset Giuridici Italiani: Esperimenti con Prompting, Retrieval-Augmented Generation e Fine-Tuning

## How to replicate tests
Testing was performed in a docker container that can be built with the `Dockerfile` in the repo.

Run scripts in `scripts` folder to perform the fine-tuning of the models.
Run inference scripts in the same folder to make the model predict answers to the questions in the test split of the datasets.
The accuracy evaluation on multiple-choice predictions can be done with the script `evaluate_accuracy.sh`.
The ROGUE score and the hits@k score can be calculated with the notebook `eval_rk_rouge.ipynb`.
