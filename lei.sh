export PYTHONPATH=.
export OMP_NUM_THREADS=1
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  scripts/offline_edge_prediction_pipethread.py \
  --model DyGFormer \
  --data LASTFM \
  --random-node-dim 128 \
  --random-edge-dim 128 \
  --ingestion-batch-size 1000 \
  --num-workers 0 \
  --num-chunks 1 \
  --batch-size 4000 \
  --print-freq 10 \
  --epoch 10 \
  --pipe-threads 1