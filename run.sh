docker run -v "$(pwd)":/code -w /code/pyronn --gpus all tensorflow/tensorflow:latest-gpu bash bench.sh
docker run -v "$(pwd)":/code -w "/code/torch-radon" --gpus all matteoronchetti/torch-radon bash bench.sh
docker run -v "$(pwd)":/code -w "/code/astra" --gpus all matteoronchetti/astra bash bench.sh