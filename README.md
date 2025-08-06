# Adaptive Width Neural Networks

### How to create the data and data splits

`mlwiz-data --config-file DATA_CONFIGS/[select dataset]`

### How to run the experiments

`mlwiz-train  --config-file EXP_CONFIGS/[select config] [--debug]`

The debug option will run everything sequentially, otherwise experiments will be launched in parallel using Ray. Adjust the hardware requirements in the configuration file. We used the [MLWiz](https://github.com/diningphil/mlwiz) library to automatize the experiments.

#### Requirements: 

`pip install -r requirements.txt`

