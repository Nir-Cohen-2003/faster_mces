Installation:
Installign from prebuilt packages:
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
