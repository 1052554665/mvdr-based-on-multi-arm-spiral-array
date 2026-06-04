- create a new configuration file similar with `train.yaml`
- Replace the `data` (`transformer/data/`) with `collected_batch_pairwise` (`mvdr/data/collected_batch_pairwise/`) in the path to load the data and iterate through the `collected_batch_pairwise` directory.
- Finally, the structure of the enhanced data  is as follows:
    ```
    transformer/experiments/dataset_undivided/
    ├── DCBias
    │       ├── enhanced_0001.npz
    │       ├── enhanced_0002.npz
    │       └── ...
    ├── Harmonic
    │       ├── enhanced_0001.npz
    │       ├── enhanced_0002.npz
    │       └── ...
    ├── Loosen
    │       ├── enhanced_0001.npz
    │       ├── enhanced_0002.npz
    │       └── ...
    └── PartialDischarge
            ├── enhanced_0001.npz
            ├── enhanced_0002.npz
            └── ...
    ```