# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import copy
import io
import json
import logging
import os
import re
from collections import defaultdict
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
import pandas as pd
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from PIL import Image

logger = logging.getLogger(__name__)


def convert_gt_to_ui_format(gt_action, gt_bbox=None, gt_input_text=None):
    """Convert ground truth data to UI agent action format."""
    # Convert numpy arrays to lists if needed
    if hasattr(gt_bbox, 'tolist'):
        gt_bbox = gt_bbox.tolist()
    
    if gt_action == "click":
        if gt_bbox is not None and len(gt_bbox) >= 2:
            if len(gt_bbox) == 2:
                x, y = gt_bbox
                return f"click(start_box='<|box_start|>({x},{y})<|box_end|>')"
            elif len(gt_bbox) == 4:
                x1, y1, x2, y2 = gt_bbox
                # Use center of bbox
                x = (x1 + x2) // 2
                y = (y1 + y2) // 2
                return f"click(start_box='<|box_start|>({x},{y})<|box_end|>')"
        return "click(start_box='<|box_start|>(0,0)<|box_end|>')"
    
    elif gt_action == "left_double":
        if gt_bbox is not None and len(gt_bbox) >= 2:
            if len(gt_bbox) == 2:
                x, y = gt_bbox
                return f"left_double(start_box='<|box_start|>({x},{y})<|box_end|>')"
            elif len(gt_bbox) == 4:
                x1, y1, x2, y2 = gt_bbox
                x = (x1 + x2) // 2
                y = (y1 + y2) // 2
                return f"left_double(start_box='<|box_start|>({x},{y})<|box_end|>')"
        return "left_double(start_box='<|box_start|>(0,0)<|box_end|>')"
    
    elif gt_action == "right_single":
        if gt_bbox is not None and len(gt_bbox) >= 2:
            if len(gt_bbox) == 2:
                x, y = gt_bbox
                return f"right_single(start_box='<|box_start|>({x},{y})<|box_end|>')"
            elif len(gt_bbox) == 4:
                x1, y1, x2, y2 = gt_bbox
                x = (x1 + x2) // 2
                y = (y1 + y2) // 2
                return f"right_single(start_box='<|box_start|>({x},{y})<|box_end|>')"
        return "right_single(start_box='<|box_start|>(0,0)<|box_end|>')"
    
    elif gt_action == "drag":
        # For drag, we need start and end boxes
        if gt_bbox is not None and len(gt_bbox) >= 4:
            if len(gt_bbox) == 4:
                # Assume it's start_x, start_y, end_x, end_y
                x1, y1, x2, y2 = gt_bbox
                return f"drag(start_box='<|box_start|>({x1},{y1})<|box_end|>', end_box='<|box_start|>({x2},{y2})<|box_end|>')"
            elif len(gt_bbox) == 8:
                # start_box and end_box coordinates
                x1, y1, x2, y2, x3, y3, x4, y4 = gt_bbox
                start_x = (x1 + x2) // 2
                start_y = (y1 + y2) // 2
                end_x = (x3 + x4) // 2
                end_y = (y3 + y4) // 2
                return f"drag(start_box='<|box_start|>({start_x},{start_y})<|box_end|>', end_box='<|box_start|>({end_x},{end_y})<|box_end|>')"
        return "drag(start_box='<|box_start|>(0,0)<|box_end|>', end_box='<|box_start|>(100,100)<|box_end|>')"
    
    elif gt_action == "type":
        if gt_input_text:
            # Escape special characters
            content = gt_input_text.replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
            return f"type(content='{content}')"
        return "type(content='')"
    
    elif gt_action == "scroll":
        direction = gt_input_text if gt_input_text else "down"
        if gt_bbox is not None and len(gt_bbox) >= 2:
            if len(gt_bbox) == 2:
                x, y = gt_bbox
                return f"scroll(start_box='<|box_start|>({x},{y})<|box_end|>', direction='{direction}')"
            elif len(gt_bbox) == 4:
                x1, y1, x2, y2 = gt_bbox
                x = (x1 + x2) // 2
                y = (y1 + y2) // 2
                return f"scroll(start_box='<|box_start|>({x},{y})<|box_end|>', direction='{direction}')"
        return f"scroll(start_box='<|box_start|>(0,0)<|box_end|>', direction='{direction}')"
    
    elif gt_action == "hotkey":
        if gt_input_text:
            return f"hotkey(key='{gt_input_text}')"
        return "hotkey(key='')"
    
    elif gt_action == "wait":
        return "wait()"
    
    elif gt_action == "finished":
        if gt_input_text:
            content = gt_input_text.replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
            return f"finished(content='{content}')"
        return "finished(content='')"
    
    elif gt_action == "call_user":
        return "call_user()"
    
    else:
        # Default to click if action type is unknown
        return "click(start_box='<|box_start|>(0,0)<|box_end|>')"


def collate_fn(data_list: list[dict]) -> dict:
    """Collate a batch of data."""
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}

def convert_parquet_to_json(parquet_file: str, json_file: str):
    df = pd.read_parquet(parquet_file)
    records = df.to_dict(orient='records')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def pil_to_data_uri(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe.filter(
                lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()


class RLHFAgentDataset(Dataset):
    def __init__(self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        processor = None, # Compatible with verl
        config = None # Compatible with verl
    ):
        # print(f'data_files: {data_files}')
        self.tokenizer = tokenizer
        self.data_files = copy.deepcopy(data_files)
        self.data = []
        self.sources = []
        self.truncation = "error"
        if isinstance(self.data_files, str):
            self.data_files = [self.data_files]
        elif isinstance(self.data_files, list):
            self.data_files = [f for f in self.data_files]
        else:
            raise ValueError(f"Unsupported data_files type: {type(self.data_files)}")
        # for i, data_file in enumerate(self.data_files):
        #     self.data.extend(json.load(open(data_file)))
        #     file_name = os.path.basename(data_file)
        #     self.sources.extend([file_name] * len(json.load(open(data_file))))
        self._read_data()

        

    def _read_data(self):
        # self._convert_parquet_to_json(self.data_files)
        # json_files = [f for f in self.data_files if f.endswith('.json')]

        # if json_files:
        #     for json_file in json_files:
        #         with open(json_file, 'r', encoding='utf-8') as f:
        #             json_data = json.load(f)
        #             self.data.extend(json_data)
        #             file_name = os.path.basename(json_file)
        #             self.sources.extend([file_name] * len(json_data))
        for data_file in self.data_files:
            if data_file.endswith('.parquet'):
                df = pd.read_parquet(data_file)
                self.data.extend(df.to_dict(orient='records'))
                self.sources.extend([os.path.basename(data_file)] * len(df))
            elif data_file.endswith('.json'):
                with open(data_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    self.data.extend(json_data)
                    self.sources.extend([os.path.basename(data_file)] * len(json_data))
            else:
                raise ValueError(f"Unsupported file type: {data_file}")
        print(f"dataset len: {len(self.data)}")


    def __len__(self):
        return len(self.data)
    
    def _build_messages(self, row_dict):
        question_keys = ['question', 'problem', 'instruction']
        for key in question_keys:
            question = None
            if key in row_dict:
                question = row_dict[key]
                break
        if question is None:
            raise ValueError(f"question not found in row_dict: {row_dict}")
        
        if "image" in row_dict:
            from verl.utils.dataset.vision_utils import process_image
            image = process_image(row_dict["image"])
            image = pil_to_data_uri(image)
            # convert PIL Image to base64
            # buffer = io.BytesIO()
            # image.save(buffer, format="PNG")
            # image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
        else:
            image = None
        
        single_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question}
                ]
            }
        ]
        if image is not None:
            # OpenAI chat completion API only supports image_url
            single_messages[0]["content"].append({"type": "image_url", "image_url": {"url": image}})
        
        # Build the messages structure with ground truth data
        messages = {
            "messages": single_messages,
            "question": question,
            # Include ground truth data if available
            "gt_action": row_dict.get("gt_action", ""),
            "gt_bbox": row_dict.get("gt_bbox", []),
            "gt_input_text": row_dict.get("gt_input_text", ""),
        }

        # Add any other info from row_dict
        other_info = {}
        for k, v in row_dict.items():
            if k not in ['question', 'messages', 'gt_action', 'gt_bbox', 'gt_input_text']:
                other_info[k] = v

        messages.update(other_info)

        return messages, question


    
    def __getitem__(self, item):
        row_dict = self.data[item]
        
        # Convert any numpy arrays to lists recursively before building messages
        def convert_numpy_to_list(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_list(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_to_list(item) for item in obj)
            else:
                return obj
        
        row_dict = convert_numpy_to_list(row_dict)
        
        messages, question = self._build_messages(row_dict)
        row_dict["messages"] = messages
        row_dict["data_source"] = self.sources[item]
        row_dict["question"] = question
        
        # Convert gt_action to UI agent format if present
        if "gt_action" in row_dict:
            gt_action = row_dict.get("gt_action", "")
            gt_bbox = row_dict.get("gt_bbox", [])
            gt_input_text = row_dict.get("gt_input_text", "")
            
            # Convert to UI agent action format
            ui_action = convert_gt_to_ui_format(gt_action, gt_bbox, gt_input_text)
            row_dict["ui_action"] = ui_action
            
            # For GUI tasks, ensure we have an image
            if "image" not in row_dict and "messages" in row_dict:
                # Add a placeholder image if none exists
                print(f"[RLAgentDataset] Warning: GUI task without image for index {row_dict.get('index', 'unknown')}")
        
        # May be for compatibility with the original dataset
        # And we don't actually need this
        # inputs = self.tokenizer(question, return_tensors='pt')
        # row_dict["input_ids"] = inputs.input_ids
        # row_dict["attention_mask"] = inputs.attention_mask
        row_dict["input_ids"] = torch.tensor([0])
        row_dict["attention_mask"] = torch.tensor([1])
        row_dict["position_ids"] = torch.tensor([0])
        
        return row_dict