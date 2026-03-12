export PYTHONPATH=.
export OMP_NUM_THREADS=1
time torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  scripts/offline_edge_prediction_pipethread.py \
  --model TGN \
  --data LASTFM \
  --random-edge-dim 128 \
  --cache LRUCache \
  --edge-cache-ratio 0 \
  --node-cache-ratio 0 \
  --snapshot-time-window 0 \
  --ingestion-batch-size 1000 \
  --num-workers 0 \
  --num-chunks 1 \
  --batch-size 4000 \
  --print-freq 10 \
  --epoch 10