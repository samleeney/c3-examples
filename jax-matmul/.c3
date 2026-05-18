project: jax-matmul-example
script: run.sh
gpu: l40
time: "00:10:00"
job_name: jax-matmul

python:
  project: ./

output:
  - ./results
