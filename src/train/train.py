import os
import sys
import numpy as np
import torch
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
)
from torch.utils.data import DataLoader

# Add the parent directory of 'classes' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prepareModel import (
    addTokens,
    assertModelProcessorConfig,
    configureModelProcessor,
    load_model_and_processor,
    save_model_and_processor,
)
from classes.DonutDataset import DonutDataset
from classes.PLClass import DonutModelPLModule
from PushToHubCallback import PushToHubCallback
import json

# Load configuration from JSON file
with open("config.json", "r") as f:
    config = json.load(f)

# Access configuration values
project_name = config["project_name"]
wandb_config = config["wandb"]
callbacks_config = config["callbacks"]
model_config = config["model"]
image_size = model_config["image_size"]
max_length = model_config["max_length"]
training_config = config["training"]

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set matmul precision to 'medium' or 'high' to utilize Tensor Cores
torch.set_float32_matmul_precision("high")


def loadPretrained(modelPath: str):
    print("\n-- Loading model and processor --")
    config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
    config.encoder.image_size = image_size
    config.decoder.max_length = max_length

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained(
        "naver-clova-ix/donut-base", config=config
    )

    # add tokens to tokenizer
    addTokens(model, processor)

    # configure model and processor
    configureModelProcessor(model, processor, image_size, max_length)

    save_model_and_processor(model, processor, modelPath)
    model, processor = load_model_and_processor(
        VisionEncoderDecoderModel, DonutProcessor, modelPath, "cuda"
    )

    assertModelProcessorConfig(model, processor)
    print("-- Loaded model and processor --\n")

    return model, processor


def loadDataset(trainPath, valPath, processor: DonutProcessor):
    print("\n-- Loading dataset --")
    assert os.path.exists(trainPath), f"Training dataset not found at {trainPath}"

    train_dataset = DonutDataset(trainPath, processor, split="train")
    val_dataset = DonutDataset(valPath, processor, split="train")

    trainDataLoader = DataLoader(
        train_dataset,
        batch_size=training_config["train_batch_sizes"][0],
        shuffle=True,
        num_workers=4,
    )
    valDataLoader = DataLoader(
        val_dataset,
        batch_size=training_config["val_batch_sizes"][0],
        shuffle=False,
        num_workers=4,
    )

    print("-- Loaded dataset --\n")
    return trainDataLoader, valDataLoader


def train(model, processor, train_dataloader, val_dataloader):
    model.train()

    model_module = DonutModelPLModule(
        training_config,
        processor,
        model,
        0,
        model.config.decoder.max_length,
        train_dataloader,
        val_dataloader,
    )

    wandb_logger = WandbLogger(
        project=wandb_config["project"],
        name=wandb_config["name"],
        save_dir=wandb_config["save_dir"],
    )
    print(
        "Decoder start token ID:",
        processor.decode([model.config.decoder_start_token_id]),
    )

    early_stop_callback = EarlyStopping(
        monitor=callbacks_config["early_stopping"]["monitor"],
        patience=callbacks_config["early_stopping"]["patience"],
        verbose=callbacks_config["early_stopping"]["verbose"],
        mode=callbacks_config["early_stopping"]["mode"],
    )

    trainer = pl.Trainer(
        accelerator=training_config["accelerator"],
        devices=training_config["devices"],
        max_epochs=training_config["max_epochs"],
        val_check_interval=training_config["val_check_interval"],
        check_val_every_n_epoch=training_config["check_val_every_n_epoch"],
        gradient_clip_val=training_config["gradient_clip_val"],
        precision=training_config["precision"],
        num_sanity_val_steps=training_config["num_sanity_val_steps"],
        logger=wandb_logger,
        # callbacks=[PushToHubCallback(callbacks_config['push_to_hub']['save_model_path']), early_stop_callback],
        callbacks=[
            PushToHubCallback(callbacks_config["push_to_hub"]["save_model_path"])
        ],
    )

    trainer.fit(model_module)
