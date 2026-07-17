#!/bin/bash

BASE_DIR="/path/to/data/Pseudo_data/benchmark_data/step3_trajectories_split12"
SCRIPT="robust_benchmark_generator.py"

for i in $(seq -w 1 12); do
  INPUT="${BASE_DIR}/step3_trajectories_part${i}.json"
  OUTPUT="${BASE_DIR}/step3_trajectories_part${i}_gptoss.json"

  echo "Starting part ${i}..."

  python ${SCRIPT} \
    --input "${INPUT}" \
    --output "${OUTPUT}" \
    --trajectory-mode 32k \
    --num-trajectories 4193 \
    --questions-per-trajectory 3 \
    --easy-ratio 0.6 \
    --medium-ratio 0.4 \
    --hard-ratio 0.0 \
    --total-questions 10000 \
    --balanced-depth &

done

wait
echo "All 12 jobs finished."
