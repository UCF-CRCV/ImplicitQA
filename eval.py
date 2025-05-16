# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use vLLM for running offline inference on videos
with the correct prompt format on vision language models for text generation.
We separate model creation and prompt formatting so the model can be reused
with different questions.
"""

import random
import time
import os
import json
from turtledemo.penrose import start

from transformers import AutoTokenizer
import torch
from torchcodec.decoders import VideoDecoder
from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser
import cv2
import numpy as np
from tqdm import tqdm
import csv

def save_json(save_path, content):
    with open(save_path, 'w') as f:
        f.write(json.dumps(content))

def save_list_to_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

def load_list_from_json(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    return data

def uniform_sampling(video_tensor, num_frames):
    total_frames = video_tensor.shape[0]
    if num_frames == -1:
        return video_tensor
    indices = torch.linspace(0, total_frames - 1, num_frames).long()
    return video_tensor[indices]


def sample_frames_from_video(frames, num_frames):
    total_frames = frames.data.shape[0]
    if num_frames == -1:
        return frames
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    sampled_frames = frames[frame_indices, ...]
    return sampled_frames


def load_full_video(video_path):
    device = "cpu" #"cuda"
    decoder = VideoDecoder(video_path, seek_mode='exact', device=device)
    return decoder

def load_video_slice(decoder, start_time, end_time, num_frames):
    # device = "cuda"
    # decoder = VideoDecoder(video_path, seek_mode='exact', device=device)
    start_time = max(start_time, 0.01)
    end_time = min(end_time, decoder.metadata.duration_seconds - 0.01)

    batch = decoder.get_frames_played_in_range(start_seconds=start_time, stop_seconds=end_time)
    sampled_frames = uniform_sampling(batch.data, num_frames)
    return sampled_frames


def video_to_ndarrays(path, num_frames=-1):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    frames = np.stack(frames)
    frames = sample_frames_from_video(frames, num_frames)
    if num_frames != -1 and len(frames) < num_frames:
        raise ValueError(f"Could not read enough frames from video file {path}"
                         f" (expected {num_frames} frames, got {len(frames)})")
    return frames


#############################
# LLaVA-NeXT-Video Functions
#############################
def create_llava_next_video_model():
    llm = LLM(model="llava-hf/LLaVA-NeXT-Video-7B-hf",
              max_model_len=8192,
              max_num_seqs=5,
              disable_mm_preprocessor_cache=False)
    stop_token_ids = None
    return llm, stop_token_ids


def format_llava_next_video_prompt(question: str, modality: str):
    if modality != "video":
        raise ValueError("This function supports video inference only.")
    prompt = f"USER: <video>\n{question} ASSISTANT:"
    return prompt


#############################
# LLaVA-OneVision Functions
#############################
def create_llava_onevision_model():
    llm = LLM(model="llava-hf/llava-onevision-qwen2-7b-ov-hf",
              max_model_len=16384,
              max_num_seqs=5,
              disable_mm_preprocessor_cache=False)
    stop_token_ids = None
    return llm, stop_token_ids


def format_llava_onevision_prompt(question: str, modality: str):
    if modality != "video":
        raise ValueError("This script supports video inference only.")
    prompt = f"<|im_start|>user <video>\n{question}<|im_end|> <|im_start|>assistant\n"
    return prompt


#############################
# LLaVA-Video Functions
#############################
def create_llava_video_model():
    llm = LLM(model="weights/LLaVA-Video-7B-Qwen2-hf",
              max_model_len=16384,
              max_num_seqs=5,
              disable_mm_preprocessor_cache=False)
    stop_token_ids = None
    return llm, stop_token_ids


def format_llava_video_prompt(question: str, modality: str):
    if modality != "video":
        raise ValueError("This script supports video inference only.")
    prompt = f"<|im_start|>user <video>\n{question}<|im_end|> <|im_start|>assistant\n"
    return prompt




def create_qwen2_vl_model():
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    llm = LLM(
        model=model_name,
        max_model_len=16384,
        max_num_seqs=5,
        mm_processor_kwargs={
            "max_pixels": 360 * 420,
        },
        disable_mm_preprocessor_cache=False,
    )
    stop_token_ids = None
    return llm, stop_token_ids

def create_qwen2_vl_2b_model():
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    llm = LLM(
        model=model_name,
        max_model_len=16384,
        max_num_seqs=5,
        mm_processor_kwargs={
            "max_pixels": 360 * 420,
        },
        disable_mm_preprocessor_cache=False,
    )
    stop_token_ids = None
    return llm, stop_token_ids

def create_qwen2_vl_72b_model():
    model_name = "Qwen/Qwen2-VL-72B-Instruct"
    llm = LLM(
        model=model_name,
        max_model_len=16384,
        max_num_seqs=5,
        mm_processor_kwargs={
            "max_pixels": 360 * 420,
        },
        disable_mm_preprocessor_cache=False,
    )
    stop_token_ids = None
    return llm, stop_token_ids


def format_qwen2_vl_prompt(question: str, modality: str):
    if modality != "video":
        raise ValueError("This function supports video inference only.")
    placeholder = "<|video_pad|>"
    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    return prompt


def create_qwen2_5_vl_7b_model():
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    llm = LLM(
        model=model_name,
        max_model_len=32768,  # 16384,
        max_num_seqs=5,
        mm_processor_kwargs={
            "max_pixels": 360 * 420,
            "fps": 1,
        },
        disable_mm_preprocessor_cache=False,
    )
    stop_token_ids = None
    return llm, stop_token_ids


def create_qwen2_5_vl_3b_model():
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    llm = LLM(
        model=model_name,
        max_model_len=32768,  # 16384,
        max_num_seqs=5,
        mm_processor_kwargs={
            "max_pixels": 360 * 420,
            "fps": 1,
        },
        disable_mm_preprocessor_cache=False,
    )
    stop_token_ids = None
    return llm, stop_token_ids


def create_qwen2_5_vl_32b_model():
    model_name = "Qwen/Qwen2.5-VL-32B-Instruct"
    llm = LLM(
        model=model_name,
        max_model_len=16384, #32768,  # 16384,
        max_num_seqs=5,
        mm_processor_kwargs={
            "max_pixels": 360 * 420,
            "fps": 1,
        },
        disable_mm_preprocessor_cache=False,
    )
    stop_token_ids = None
    return llm, stop_token_ids


def create_qwen2_5_vl_72b_model():
    model_name = "Qwen/Qwen2.5-VL-72B-Instruct"
    llm = LLM(
        model=model_name,
        max_model_len=8192, # 32768,  # 16384,
        max_num_seqs=5,
        mm_processor_kwargs={
            "max_pixels": 360 * 420,
            "fps": 1,
        },
        disable_mm_preprocessor_cache=False,
    )
    stop_token_ids = None
    return llm, stop_token_ids


def create_qwen2_5_vl_72b_awq_model():
    model_name = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
    llm = LLM(
        model=model_name,
        max_model_len=32768,  # 16384,
        max_num_seqs=5,
        mm_processor_kwargs={
            "max_pixels": 360 * 420,
            "fps": 1,
        },
        disable_mm_preprocessor_cache=False,
    )
    stop_token_ids = None
    return llm, stop_token_ids

def format_qwen2_5_vl_prompt(question: str, modality: str):
    if modality != "video":
        raise ValueError("This function supports video inference only.")
    placeholder = "<|video_pad|>"
    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    return prompt


#############################
# Mapping Dictionaries
#############################
# For models that do not need a tokenizer for prompt formatting.
model_create_map = {
    "llava-next-video": (create_llava_next_video_model, format_llava_next_video_prompt),
    "llava-onevision": (create_llava_onevision_model, format_llava_onevision_prompt),
    "llava-video": (create_llava_video_model, format_llava_video_prompt),
    "qwen2_vl": (create_qwen2_vl_model, format_qwen2_vl_prompt),
    "qwen2_vl_2b": (create_qwen2_vl_2b_model, format_qwen2_vl_prompt),
    "qwen2_vl_72b": (create_qwen2_vl_72b_model, format_qwen2_vl_prompt),
    "qwen2_5_vl_7b": (create_qwen2_5_vl_7b_model, format_qwen2_5_vl_prompt),
    "qwen2_5_vl_3b": (create_qwen2_5_vl_3b_model, format_qwen2_5_vl_prompt),
    "qwen2_5_vl_32b": (create_qwen2_5_vl_32b_model, format_qwen2_5_vl_prompt),
    "qwen2_5_vl_72b": (create_qwen2_5_vl_72b_model, format_qwen2_5_vl_prompt),
    "qwen2_5_vl_72b_awq": (create_qwen2_5_vl_72b_awq_model, format_qwen2_5_vl_prompt),
}


def load_annotations(input_file):
    data = []
    with open(input_file, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(dict(row))
    return data

def list_all_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            file_list.append(full_path)
    return file_list

def build_question(sample_dict):
    question = sample_dict['question_text']
    for k,v in sample_dict['options'].items():
        question += f'Option {k}: {v} '

    return question

def check_file_empty(file_path):
    """
    Checks if a file exists and is not empty.

    Args:
        file_path: The path to the file.

    Returns:
        True if the file exists and is not empty, False otherwise.
    """
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return True
    else:
        return False

def main(args):
    modality = "video"
    video_format = 'mp4'

    qs_prompt = "\nAnswer with the option's letter from the given choices directly."

    if args.model_type in model_create_map:
        create_fn, format_fn = model_create_map[args.model_type]
        llm, stop_token_ids = create_fn()
        print(f'created model {args.model_type}')
    else:
        raise ValueError(f"Model type {args.model_type} is not supported.")

    base_save_dir = f'outputs_v3/{args.model_type}_n{args.num_frames}'
    base_log_dir = f"logs_v3/{args.model_type}_n{args.num_frames}"

    total_acc = 0
    total_correct = 0
    total_num = 0
    total_num = 0
    total_long_ans_count = 0
    total_long_ans_fail = 0

    for filename in tqdm(args.files):
        print(f'{filename=}')
        data = load_list_from_json(filename) 
        if len(data) > 0:
            try:
                annotator_name = filename.split('/')[-2]
                video_name = filename.split('/')[-1].replace('.json', '').replace('video_', '')
                video_file = f'{args.video_base_path}/{annotator_name}/{video_name}.{video_format}'
                input_file = filename.split('/')[-1]
                save_dir = f'{base_save_dir}/{annotator_name}'
                output_file = f"{save_dir}/{video_name}.json"

                if check_file_empty(output_file):
                    print(f"File '{output_file}' exists. Skipping operations.")
                    continue

                log_dir = f"{base_log_dir}/{annotator_name}/"

                if not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                log_file = open(f"{log_dir}/{video_name}.log", "w")

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)

                video_decoder = load_full_video(video_file)

                
                for question in data:
                    prompts = [(format_fn(build_question(question) + qs_prompt, modality),
                                load_video_slice(decoder=video_decoder, start_time=question['question_start_time'], end_time=question['question_stop_time'], num_frames=args.num_frames).cpu().numpy())]
                    inputs = [{
                        "prompt": prompt[0],
                        "multi_modal_data": {modality: prompt[1]},
                    } for prompt in prompts]
                    sampling_params = SamplingParams(temperature=0.0,
                                                     max_tokens=64,
                                                     stop_token_ids=stop_token_ids)
                    print(f'Generating predictions ...', file=log_file, flush=True)
                    outputs = llm.generate(inputs, sampling_params=sampling_params)
                    question['pred'] = outputs[0].outputs[0].text

                save_list_to_json(data=data, output_file=output_file)
            except Exception as e:
                print(f'Exception: {e}')
                continue



if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='Inference on videos with vllm models for text generation')
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        default="llava-next-video",
                        choices=list(model_create_map.keys()),
                        help='Huggingface model type for video inference.')
    parser.add_argument('--num-frames',
                        type=int,
                        default=16,
                        help='Number of frames to extract from the video.')
    parser.add_argument('--input-dir',
                        type=str,
                        default=None,
                        help='Input dir to process.')
  parser.add_argument('--video-base-path',
                        type=str,
                        default=None,
                        help='Path to Video Directory.')
    
    args = parser.parse_args()

    start_time = time.perf_counter()

    args.files = list_all_files(args.input_dir)

    main(args)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
