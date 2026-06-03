- create a script to recollect the `mvdr/output/batch_pairwise` folder, the collected directory tree is as follows:

```
mvdr/data/collected_batch_pairwise
├── DCBias
│   └── clean (all figures named as `interference_with_target.png` from subfolders)
│       └── interference_with_target.png
│       └── interference_with_target.png
│       └── ....
│   └── noise (all figures named as `target_with_interference.png` from subfolders)
│       └── target_with_interference.png
│       └── target_with_interference.png
│       └── ....
├── Harmonic
│   └── clean 
│       └── interference_with_target.png
│       └── interference_with_target.png
│       └── ....
│   └── noise
│       └── target_with_interference.png
│       └── target_with_interference.png
│       └── ....
├── Loosen
│   └── clean 
│       └── interference_with_target.png
│       └── interference_with_target.png
│       └── ....
│   └── noise
│       └── target_with_interference.png
│       └── target_with_interference.png
│       └── ....
└── PartialDischarge
        └── clean 
                └── interference_with_target.png
                └── interference_with_target.png
                └── ....
        └── noise
                └── target_with_interference.png
                └── target_with_interference.png
                └── ....
```

```bash
bash scripts/recollect_batch_pairwise.sh
```