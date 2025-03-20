import os
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import hydra
import torch
from fastapi import FastAPI, UploadFile, File
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tracklab.datastruct import TrackerState
from tracklab.pipeline import Pipeline
from tracklab.utils import wandb

# Initialize FastAPI app
app = FastAPI(title="TrackLab API", description="API for running tracking inference on videos")

# Global variables to store initialized components
tracking_dataset = None
evaluator = None
pipeline = None
cfg = None

def init_environment(cfg):
    # For Hydra and Slurm compatibility
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: '{device}'")
    wandb.init(cfg)
    if cfg.print_config:
        print(OmegaConf.to_yaml(cfg))
    return device

def set_sharing_strategy():
    torch.multiprocessing.set_sharing_strategy("file_system")

@app.on_event("startup")
@hydra.main(version_base=None, config_path="pkg://tracklab.configs", config_name="config")
def startup(config):
    global tracking_dataset, evaluator, pipeline, cfg
    cfg = config
    
    # Initialize environment
    device = init_environment(cfg)
    set_sharing_strategy()

    # Instantiate all modules
    tracking_dataset = instantiate(cfg.dataset)
    evaluator = instantiate(cfg.eval, tracking_dataset=tracking_dataset)

    # Initialize pipeline modules
    modules = []
    if cfg.pipeline is not None:
        for name in cfg.pipeline:
            module = cfg.modules[name]
            inst_module = instantiate(module, device=device, tracking_dataset=tracking_dataset)
            modules.append(inst_module)

    pipeline = Pipeline(models=modules)

    # Train tracking modules if needed
    for module in modules:
        if module.training_enabled:
            module.train()

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify the API is running"""
    return {"status": "ok", "message": "API is running"}

@app.post("/invoke")
async def process_video(video: UploadFile = File(...)):
    """Process a video file and return tracking results"""
    
    if not video.filename.endswith('.mp4'):
        return {"error": "Only .mp4 files are supported"}
        
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Save uploaded video
        video_path = temp_dir / "input.mp4"
        with open(video_path, "wb") as f:
            f.write(await video.read())
            
        # Extract frames
        cap = cv2.VideoCapture(str(video_path))
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir()
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(frames_dir / f"frame_{frame_count:06d}.jpg"), frame)
            frame_count += 1
        cap.release()
        
        # Initialize tracker state
        tracking_set = tracking_dataset.sets[cfg.dataset.eval_set]
        tracker_state = TrackerState(
            tracking_set, 
            pipeline=pipeline,
            save_file=temp_dir / "results.zip",
            **cfg.state
        )
        
        # Run tracking
        tracking_engine = instantiate(
            cfg.engine,
            modules=pipeline,
            tracker_state=tracker_state,
        )
        tracking_engine.track_dataset()
        
        # Save visualization video
        output_dir = Path("outputs/videos")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Return results
        return {
            "status": "success",
            "predictions": str(tracker_state.save_file),
            "visualization": str(output_dir / f"{video.filename}_output.mp4")
        } 