#!/usr/bin/env bash
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/launch.sh" VLN-CE/run.py --run-type inference "$@"
