# PULSE
This is the repository for Bayesian-based uncertainty estimation for multi-agent systems.

## Requirements
Please install the environment by using the environment.yml

## Dataset
You can find the HumanEval dataset in the data directory.

## Run

python main_run.py --model Qwen/Qwen2.5-Coder-32B-Instruct --dataset HumanEval --agent_num 4 --device 1 --random_split 100 --epochs 200 --lr 0.001 --mixer_hidden_dim 256 --noise 0.001


## License

This project is licensed under the Apache-2.0 License.
