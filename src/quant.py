import subprocess
import argparse
import os
import json

def arguments(raw_args):
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='script for RangeNet model quantization')
    parser.add_argument('--config', help='model configuration to use', type=str, required=True)
    args = parser.parse_args(raw_args)
    print(vars(args))
    return args


def main(raw_args=None):
    """ Run evaluations """
    args = arguments(raw_args)
    if os.path.exists(args.config):
        with open(args.config) as f_in:
            config = json.load(f_in)

    print("FP32 Evaluation:")
    command = [
        "python",
        "src/infer.py",
        "-d", config['dataset'],
        "-l", "src/FP32_prediction",
        "-m", config['pretrained_model'],
        "-cfg", args.config,
    ]
    subprocess.run(command)
    command = [
        "python",
        "src/evaluate_iou.py",
        "-d", config['dataset'],
        "-p", "src/FP32_prediction",
        "--split", "valid"
    ]
    subprocess.run(command)
    print("\nQuantize Evaluation:")
    command = [
        "python",
        "src/infer.py",
        "-d", config['dataset'],
        "-l", "src/Int8_prediction",
        "-m", config['pretrained_model'],
        "-cfg", args.config,
        "-q", "True"
    ]
    subprocess.run(command)
    command = [
        "python",
        "src/evaluate_iou.py",
        "-d", config['dataset'],
        "-p", "src/Int8_prediction",
        "--split", "valid"
    ]
    subprocess.run(command) 

if __name__ == '__main__':
    main()