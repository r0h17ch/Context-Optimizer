# source this before running anything:  source ~/Context-Optimizer/SAC/env.sh
export SAC_ROOT=$HOME/sac_assets
export SAC_HOME=$HOME/Context-Optimizer/SAC
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate SAC
# single consumer GPU: avoid NCCL p2p issues
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0
# the evaluators call plt.show(); on a desktop session that opens a Tk window and blocks forever
export MPLBACKEND=Agg
