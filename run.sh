docker run --rm -it -v "$(pwd)":/code -w /code/pyronn --gpus all tensorflow/tensorflow:latest-gpu bash bench.sh
docker run --rm -it -v "$(pwd)":/code -w "/code/torch-radon" --gpus all matteoronchetti/torch-radon:beta bash bench.sh
docker run --rm -it -v "$(pwd)":/code -w "/code/astra" --gpus all matteoronchetti/astra bash bench.sh
