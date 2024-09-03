import train.evaluate as evaluate
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
import re
import time

DATASET_PATH = "../data/dataset/train"
local_save_dir = "saved_models/21"
image_path = "/home/roland/ml_playground-1/pytorch/document_intelligence/data/dataset/train/images/mb_hydro_1_20240818132320562224.jpg"

model, processor= evaluate.load_model_and_processor(
    VisionEncoderDecoderModel, DonutProcessor, local_save_dir, "cuda"
)


def inference_step(processor, model, device, pixel_values):
    model.eval()
    batch_size = pixel_values.shape[0]

    # Feed the prompt to the model
    decoder_input_ids = torch.full(
        (batch_size, 1),
        model.config.decoder_start_token_id,
        device=device,
    )

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=768,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    predictions = []
    for seq in processor.tokenizer.batch_decode(outputs.sequences):
        seq = seq.replace(processor.tokenizer.eos_token, "").replace(
            processor.tokenizer.pad_token, ""
        )
        seq = re.sub(
            r"<.*?>", "", seq, count=1
        ).strip()  # remove first task start token
        seq = processor.token2json(seq)
        predictions.append(seq)

    return predictions



def predict_single_image(model, processor, image_path, device):
    start_time = time.time()

    # Load the image from the given path
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    # Perform inference
    predictions = inference_step(processor, model, device, pixel_values)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken for prediction: {elapsed_time:.4f} seconds")

    return predictions

print(predict_single_image(model, processor, image_path, "cuda"))
