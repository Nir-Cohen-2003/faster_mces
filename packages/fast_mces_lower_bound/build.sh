#!/bin/bash
set -ex

python -m build --wheel --outdir "$SRC_DIR/dist/wheels" --no-isolation --skip-dependency-check
python -m pip install "$SRC_DIR/dist/wheels/"*.whl --no-deps --no-build-isolation --prefix "$PREFIX" -vv
