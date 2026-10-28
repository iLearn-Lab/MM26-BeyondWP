#!/usr/bin/env bash
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/launch.sh" TWP/predict.py "$@"
