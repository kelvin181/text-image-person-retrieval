"""
vLLM/Qwen2-VL wrapper for ICL reranking.

Adapted from MLLMs class in ICL/2025-CVPR-ICL/vllm_infer_ICL.py (lines 89–140).
Changes from upstream:
  - tensor_parallel_size is a constructor parameter (default 1) instead of hardcoded 2
  - Renamed generate_response_qwen2vl → generate_response_text_only
  - No imports from parent project
"""

import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


class MLLMs(object):
    def __init__(self, model_dir, tensor_parallel_size=1, gpu_memory_utilization=0.8):
        self.model_dir = model_dir
        print("Loading MLLM...")
        print(model_dir)
        self.llm = LLM(
            model=model_dir,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(model_dir)
        print("MLLM loaded.")

    def generate_response_multi_images(self, questions, images, sys_prompt="You are a helpful assistant.", temperature=0.01):
        messages = [
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [
                    {"type": "image", "image": images[i], "min_pixels": 50176, "max_pixels": 50176},
                    {"type": "text", "text": p},
                ]},
            ]
            for i, p in enumerate(questions)
        ]
        prompts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_data = [process_vision_info(msg)[0] for msg in messages]
        inputs = [
            {"prompt": p, "multi_modal_data": {"image": image_data[i]}}
            for i, p in enumerate(prompts)
        ]
        sampling_params = SamplingParams(temperature=temperature, max_tokens=2048, skip_special_tokens=True)
        outputs = self.llm.generate(inputs, sampling_params=sampling_params)
        return [o.outputs[0].text for o in outputs]

    def generate_response_text_only(self, questions, sys_prompt="You are a helpful assistant.", temperature=0.01):
        messages = [
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [{"type": "text", "text": p}]},
            ]
            for i, p in enumerate(questions)
        ]
        prompts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        inputs = [{"prompt": p} for p in prompts]
        sampling_params = SamplingParams(temperature=temperature, max_tokens=2048, skip_special_tokens=True)
        outputs = self.llm.generate(inputs, sampling_params=sampling_params)
        return [o.outputs[0].text for o in outputs]
