# COMRADE
this is the source code of COMRADE
## Requirement
python == 3.8.1

Pytorch == 1.10.1 with cuda 11.1

torch_geometric == 2.2.0

networkx == 3.0

numpy == 1.24.2

dgl == 0.4.3

pygod == 0.3.1

## Usage
python main.py --dataset cora --gama 0.6 --beta 0.9

please make sure the diff of a dataset is generated before running. If not please use the gdc() in aug.py to generate.

*For the rest of the datasets, please refer to the .sh files, and simply run:
```Bash
chmod +x runEachDataset.sh
./runEachDataset.sh
```

