# I-MLE sampling
Specify `--sample_node_k` `--num_subgraphs`, especially `--train_embd_model` hparams. `--sample_node_k` can be positive integers as number of nodes per subgraph or negative number as node deletion. 

e.g.

`python main.py --batch_size 128 --epochs 1000 --sample_node_k -1 --num_subgraphs 3 --train_embd_model`

# ESAN
Specify `--esan_policy` as `node_deletion` or else. `--sample_mode` is to sample from the _deck_ of subgraph set with ratio or fixed number of subgraphs. `--esan_frac` is the ratio, `--esan_k` is the fixed number, depending on `--sample_mode`. `--voting` is for inference as [here](https://github.com/beabevi/ESAN/blob/98b6c346e8bca77db1597f88bac78178871e652c/main.py#L121), but can be used for other settings as well. 

e.g. 

Node delete: `python main.py --batch_size 128 --epochs 1000 --esan_policy node_deletion --sample_mode int --esan_k 3 --voting 5`

Edge delete: `python main.py --batch_size 128 --epochs 1000 --esan_policy edge_deletion --sample_mode int --esan_k 3 --voting 5`

# Normal training

No subgraph thing. Take a normal batch from the dataset and train it. 

__Remember__ to set `--num_subgraphs 0`

e.g.

`python main.py --batch_size 128 --epochs 1000 --num_subgraphs 0`

# Sample on the fly

Functionally similar to but practically different from ESAN. The latter keeps a _deck_ of subgraphs as a new dataset, but can be expensive when there are toooooo many combinations e.g. sample 10 nodes from 30.

Do __not__ specify `--train_embd_model` and everything will be randomly sampled.  

Node delete: `python main.py --batch_size 128 --epochs 1000 --sample_policy node --sample_node_k -1 --num_subgraphs 3`

Edge delete: `python main.py --batch_size 128 --epochs 1000 --sample_policy edge --sample_edge_k -1 --num_subgraphs 3`