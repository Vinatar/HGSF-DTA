## Organization of Result Files and Scripts

- The `architecture` and `predictor` folders contain model structure parameters and outputs for different experimental scenarios (such as S1 and S4) on the Davis and KIBA datasets, including cross-validation and test set results.
- The `evaluate` folder includes scripts for generating loss function trend plots (`draw_loss.py`) and MSE trend plots (`draw_mse.py`), as well as partial experimental results (such as `fold0.csv` and `loss_log_without_graph_info.csv`). The file `loss_log_without_graph_info.csv` corresponds to the ablation experiment (removal of molecular graph information).

## Model Architecture Statement

- The model structures in the `architecture` and `predictor` folders are not the latest versions, mainly to protect experimental details and avoid disclosing core model architectures.

## Reproducibility Support

- The `README` and `requirements.txt` files provide detailed environment and usage instructions.
- Users can reproduce the main experimental workflow and results by using the provided training scripts (such as `train1.py`) and evaluation scripts together with the data files.
- The result files and plotting scripts can be directly used to generate the main figures in the paper.

## Reproducibility Assurance

- Although the latest model architecture is not disclosed, all key experimental parameters, results, and analysis scripts are provided to ensure the main conclusions of the paper are reproducible.
- The model structure and parameters are either class initialization parameters or passed through configuration/command line, without directly written "hard coded" experimental parameters.
