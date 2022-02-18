# I-MLE sampling
Specify `--sample_k`, `--num_subgraphs`, `--train_embd_model` hparams. `--sample_k` can be positive integers as number of nodes per subgraph or negative number as node deletion. 

e.g.

`python main.py --batch_size 128 --epochs 1000 --sample_k -1 --num_subgraphs 3 --train_embd_model`

# ESAN
Specify `--policy` as `node_deletion` or else. `--sample_mode` is to sample from the _deck_ of subgraph set with ratio or fixed number of subgraphs. `--esan_frac` is the ratio, `--esan_k` is the fixed number, depending on `--sample_mode`. `--voting` is for inference as [here](https://github.com/beabevi/ESAN/blob/98b6c346e8bca77db1597f88bac78178871e652c/main.py#L121). 

e.g. 

`python main.py --batch_size 128 --epochs 1000 --policy node_deletion --sample_mode int --esan_k 3 --voting 5`

# Normal training

e.g.

`python main.py --batch_size 128 --epochs 1000 --policy null`