# ConLearn

This is the source code for SDM 2022: ConLearn: Contextual-knowledge-aware Concept Prerequisite Relation Learning with Graph Neural Network

## Dependencies

- numpy\==1.19.2
- transformers\==4.1.1
- torch\==1.6.0
- pandas\==1.1.3

## Dataset

For simplification, we only supply the DSA and ML dataset used in the paper.

```
dataset
└── MOOC
    ├── DSA
    │   ├── Captions_algorithms_Princeton.json
    │   ├── Captions_algorithms_Stanford.json
    │   ├── Captions_data-structure-and-algorithm_UC-San-Diego.json
    │   ├── CoreConcepts_DSA
    │   ├── DSA_LabeledFile
    │   └── W-DSA_LabeledFile
    └── ML
        ├── Captions_machine-learning_Stanford.json
        ├── Captions_machine-learning_Washington.json
        ├── CoreConcepts_ML
        ├── ML_LabeledFile
        └── W-ML_LabeledFile
```

## Running

Type the following command in the terminal:

```
python main.py
```

The default dataset is DSA dataset, use --dataset configuration if you want to train on another dataset.

When the training procedure is completed, the terminal will print the results stated in our paper.
