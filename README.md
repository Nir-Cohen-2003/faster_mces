Installation:
Installing from prebuilt packages:
if using conda/mamba/micromamba: (replace codna with respective tool)
conda install -c https://prefix.dev/nir-cohen mces_splitting
if using pixi (recommended in general):
pixi add mces_splitting
build from source:
install pixi and cloen this directory.
to create package:
pixi build
to install into the default environemnt of the project:
pixi install
to run tests:
pixi run splitting_test 

Usage:
from mces_splitting import split_dataset_lower_bound_only
smiles_list = [CCO,CCN]
train_set, validation_set, test_set, threshold = split_dataset_lower_bound_only(
        smiles_list.copy(),
        validation_fraction=0.1,
        test_fraction=0.1,
        initial_distinction_threshold=10,
        min_distinction_threshold=0,
        threshold_step=-1,
        mces_matrix_save_path=test_matrix_path
    )
