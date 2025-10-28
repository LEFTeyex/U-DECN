#!/usr/bin/env bash

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test_dcnv3.py
