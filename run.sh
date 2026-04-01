#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=test.out
#SBATCH --error=test.err
#SBATCH --account=cosmo
#SBATCH --nodes=1
#SBATCH --partition=mig
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2


tox -e tests -- utils/test_additive.py::test_composition_uncoupled_atomicbasis
