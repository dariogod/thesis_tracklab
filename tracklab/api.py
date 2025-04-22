import os
import tempfile
from pathlib import Path
import rich.logging
from typing import Optional
import sys

import cv2
import hydra
from hydra import compose, initialize
import torch
from fastapi import FastAPI, UploadFile, File
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tracklab.utils import progress  # needed to avoid complex hydra stacktraces when errors occur in "instantiate(...)"

from tracklab.datastruct import TrackerState
from tracklab.pipeline import Pipeline
from tracklab.utils import wandb
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

pipeline = None
cfg = None
tracking_dataset = None
evaluator = None

def load_config():
    """Loads specified hydra config using the Compose API.
    This method is used because FastAPI and Hydra's @main decorator don't work well together.
    """
    # Clear any existing Hydra configuration
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    # Initialize Hydra with the config path
    with initialize(version_base=None, config_path="pkg://tracklab.configs"):
        # Create a new config with hydra configuration
        config = compose(config_name="soccernet", overrides=[])
        
        # Create a new config with hydra configuration
        hydra_config = OmegaConf.create({
            "hydra": {
                "job": {
                    "chdir": True,
                    "name": "api"
                },
                "run": {
                    "dir": "outputs/api/${now:%Y-%m-%d}/${now:%H-%M-%S}"
                }
            }
        })
        
        # Merge the configs
        config = OmegaConf.merge(hydra_config, config)
        
        # Handle interpolations manually
        project_dir = os.getcwd()
        config.home_dir = os.path.expanduser("~")
        config.data_dir = os.path.join(project_dir, "data")
        config.model_dir = os.path.join(project_dir, "pretrained_models")
        
        return config

def init_environment(config):
    global cfg
    cfg = config
    # For Hydra and Slurm compatibility
    progress.use_rich = cfg.use_rich
    set_sharing_strategy()  # Do not touch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: '{device}'.")
    wandb.init(cfg)
    if cfg.print_config:
        log.info(OmegaConf.to_yaml(cfg))
    if cfg.use_rich:
        for handler in log.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.ERROR)
        log.root.addHandler(rich.logging.RichHandler(level=logging.INFO))
    else:
        # TODO : Fix for mmcv fix. This should be done in a nicer way
        for handler in log.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.INFO)
    return device

def setup_pipeline():
    """Initialize the pipeline with the loaded configuration"""
    device = init_environment(cfg)

    # Instantiate all modules
    global tracking_dataset
    tracking_dataset = instantiate(cfg.dataset)
    global evaluator
    evaluator = instantiate(cfg.eval, tracking_dataset=tracking_dataset)

    modules = []
    if cfg.pipeline is not None:
        for name in cfg.pipeline:
            module = cfg.modules[name]
            inst_module = instantiate(module, device=device, tracking_dataset=tracking_dataset)
            modules.append(inst_module)

    global pipeline
    pipeline = Pipeline(models=modules)

# Initialize FastAPI app
app = FastAPI(title="TrackLab API", description="API for running tracking inference on videos")

def set_sharing_strategy():
    torch.multiprocessing.set_sharing_strategy(
        "file_system"
    )

# Load configuration and setup pipeline
cfg = load_config()
setup_pipeline()

@app.on_event("startup")
def startup_event():    
    log.info("FastAPI startup completed")

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
        frames_dir = Path("data/SoccerNetGS/gamestate-2024/challenge/SNGS-001/img1")
        
        frame_count = 0
        fps = 6
        frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / fps)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                cv2.imwrite(str(frames_dir / f"{int(frame_count/frame_interval):06d}.jpg"), frame)
            frame_count += 1
        cap.release()
        
        # Initialize tracker state
        tracking_set = tracking_dataset.sets[cfg.dataset.eval_set]
        tracker_state = TrackerState(
            tracking_set, 
            pipeline=pipeline,
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