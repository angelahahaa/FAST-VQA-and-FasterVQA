from typing import List
import yaml
import decord
from fastvqa.datasets import get_spatial_fragments, SampleFrames, FragmentSampleFrames
from fastvqa.models import DiViDeAddEvaluator
import torch
import numpy as np

import pandas as pd
import os
from datetime import datetime


def sigmoid_rescale(score, model="FasterVQA"):
    mean, std = mean_stds[model]
    x = (score - mean) / std
    print(f"Inferring with model [{model}]:")
    score = 1 / (1 + np.exp(-x))
    return score

mean_stds = {
    "FasterVQA": (0.14759505, 0.03613452), 
    "FasterVQA-MS": (0.15218826, 0.03230298),
    "FasterVQA-MT": (0.14699507, 0.036453716),
    "FAST-VQA":  (-0.110198185, 0.04178565),
    "FAST-VQA-M": (0.023889644, 0.030781006), 
}

opts = {
    "FasterVQA": "./options/fast/f3dvqa-b.yml", 
    "FasterVQA-MS": "./options/fast/fastervqa-ms.yml", 
    "FasterVQA-MT": "./options/fast/fastervqa-mt.yml", 
    "FAST-VQA": "./options/fast/fast-b.yml", 
    "FAST-VQA-M": "./options/fast/fast-m.yml", 
}
def evaluate(video_paths: List[str], model: str, device: str) -> float:

    opt = opts.get(model, opts["FAST-VQA"])
    with open(opt, "r") as f:
        opt = yaml.safe_load(f)
    ### Model Definition
    evaluator = DiViDeAddEvaluator(**opt["model"]["args"]).to(device)
    evaluator.load_state_dict(torch.load(opt["test_load_path"], map_location=device)["state_dict"])

    ### Data Definition
    vsamples = {}
    t_data_opt = opt["data"]["val-kv1k"]["args"]
    s_data_opt = opt["data"]["val-kv1k"]["args"]["sample_types"]

    for video_path in video_paths:
        video_reader = decord.VideoReader(video_path)
        
        for sample_type, sample_args in s_data_opt.items():
            ## Sample Temporally
            if t_data_opt.get("t_frag",1) > 1:
                sampler = FragmentSampleFrames(fsize_t=sample_args["clip_len"] // sample_args.get("t_frag",1),
                                            fragments_t=sample_args.get("t_frag",1),
                                            num_clips=sample_args.get("num_clips",1),
                                            )
            else:
                sampler = SampleFrames(clip_len = sample_args["clip_len"], num_clips = sample_args["num_clips"])
            
            num_clips = sample_args.get("num_clips",1)
            frames = sampler(len(video_reader))
            print("Sampled frames are", frames)
            frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}
            imgs = [frame_dict[idx] for idx in frames]
            video = torch.stack(imgs, 0)
            video = video.permute(3, 0, 1, 2)

            ## Sample Spatially
            sampled_video = get_spatial_fragments(video, **sample_args)
            mean, std = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375])
            sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
            
            sampled_video = sampled_video.reshape(sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]).transpose(0,1)
            vsamples[sample_type] = sampled_video.to(device)
            print(sampled_video.shape)
        result = evaluator(vsamples)
        score = sigmoid_rescale(result.mean().item(), model=model)
        yield score


if __name__ == '__main__':
    root_dir = r'D:\dev\data\synth3_mp4\test'
    outdir = r'C:\Users\pengshiya\dev\shiya\local\fast-vqa'

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f'{datetime.now().timestamp():.0f}.csv')
    video_paths = [os.path.join(root_dir, vid_fname) for vid_fname in os.listdir(root_dir)]
    with open(outpath, 'w') as f:
        f.write(f'video_path,score\n')
        i=0
        for score in evaluate(video_paths, 'FAST-VQA', 'cuda'):
            f.write(f'{video_paths[i]},{score}\n')
            f.flush()
            i+=1