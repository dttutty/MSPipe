torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  scripts/offline_edge_prediction_pipethread.py \
  --model TGN \
  --data LASTFM \
  --cache LRUCache \
  --edge-cache-ratio 0 \
  --node-cache-ratio 0 \
  --snapshot-time-window 0 \
  --ingestion-batch-size 1000 \
  --num-workers 0 \
  --num-chunks 1 \
  --print-freq 10 \
  --epoch 10



torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/offline_edge_prediction_pipethread.py \
  --model TGN \
  --data LASTFM \
  --cache LRUCache \
  --edge-cache-ratio 0 \
  --node-cache-ratio 0 \
  --snapshot-time-window 0 \
  --ingestion-batch-size 1000 \
  --num-workers 0 \
  --num-chunks 1 \
  --print-freq 10 \
  --epoch 10 \
  --strict-memory
