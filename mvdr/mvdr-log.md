
- Both saved as high-resolution PNG images (300 DPI) with white backgrounds, suitable for transformer training pipelines.

- Expected Output Structure:

```
output/batch_pairwise/
├── DCBias/
│       ├──target__interference/
│           ├── target_with_interference.png
│           └── interference_with_target.png
│       ├──target__interference/
│           ├── target_with_interference.png
│           └── interference_with_target.png
│       ...
├── Harmonic/
│           └── ...
├── Loosen/
│       └── ...
└── PartialDischarge/
        └── ...
```

- Current Output Structure:

```
output/batch_pairwise/
├── DCBias/
│   ├── target_with_interference.png
│   └── interference_with_target.png
├── Harmonic/
│   ├── target_with_interference.png
│   └── interference_with_target.png
├── Loosen/
│   ├── target_with_interference.png
│   └── interference_with_target.png
├── PartialDischarge/
│   ├── target_with_interference.png
│   └── interference_with_target.png
└── batch_manifest.csv
```
