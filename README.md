To run the experiments, install the environment first:

```angular2html
echo y | conda create -n diffsub python=3.8
conda activate diffsub
pip install -e .
pip install torch-geometric
```

Note: if you want to use other versions of [torch](https://pytorch.org/) or [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html#), or other versions of cuda, see to these links [torch versions](https://download.pytorch.org/whl/torch_stable.html), [PyG version](https://data.pyg.org/whl/) and replace them in the `setup.py` file. You can also download them separately via `pip` or `conda`.

After successful installation then simply run `python main.py with /path/to/your/configs`

e.g. `python main.py with configs/zinc/normal_training.yaml`