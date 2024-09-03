import json
import re
from donut import JSONParseEvaluator
import numpy as np
import torch
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
)
from datasets import load_dataset
from tqdm.auto import tqdm
import os

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load configuration from JSON file
with open("config.json", "r") as f:
    config = json.load(f)

model_config = config["model"]
task_prompt = model_config["task_prompt"]

def evaluate(
    datasetPath, localSaveDir, device, model=None, processor=None, printPercentage=0.2
):
    print("\n---- Evaluating model ----")
    if processor is None:
        processor = DonutProcessor.from_pretrained(
            os.path.join(localSaveDir, "processor")
        )
    if model is None:
        model = VisionEncoderDecoderModel.from_pretrained(
            os.path.join(localSaveDir, "model")
        )
    print("eval_image_size:", model.config.encoder.image_size)
    print("max_length:", model.config.decoder.max_length)
    print("max_position_embeddings:", model.config.decoder.max_position_embeddings)
    print("voab_size:", model.config.decoder.vocab_size)

    print("Special tokens:", processor.tokenizer.special_tokens_map)

    model.eval()
    model.to(device)

    output_list = []
    accs = []

    dataset = load_dataset(datasetPath, split="train")
    typeScore = {}
    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        # prepare encoder inputs
        pixel_values = processor(
            sample["image"].convert("RGB"), return_tensors="pt"
        ).pixel_values
        pixel_values = pixel_values.to(device)
        
        # prepare decoder inputs
        decoder_input_ids = processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        decoder_input_ids = decoder_input_ids.to(device)

        # autoregressively generate sequence
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        # turn into JSON
        seq = processor.batch_decode(outputs.sequences)[0]
        seq = seq.replace(processor.tokenizer.eos_token, "").replace(
            processor.tokenizer.pad_token, ""
        )
        seq = re.sub(
            r"<.*?>", "", seq, count=1
        ).strip()  # remove first task start token
        seq = processor.token2json(seq)

        ground_truth = json.loads(sample["ground_truth"])
        ground_truth = ground_truth["gt_parse"]
        evaluator = JSONParseEvaluator()
        score = evaluator.cal_acc(seq, ground_truth)

        accs.append(score)
        output_list.append(seq)

        if idx % int(len(dataset) * printPercentage) == 0:
            print(f"\n     Predict: {seq}")
            print(f"Ground truth: {ground_truth}")

        typeKey = ground_truth["t"]
        if typeKey not in typeScore.keys():
            typeScore[typeKey] = [score]
        else:
            typeScore[typeKey].append(score)

    scores = {"accuracies": accs, "mean_accuracy": np.mean(accs)}
    print(scores, f"length : {len(accs)}")

    print("\n-------")
    print("Mean accuracy:", np.mean(accs))
    for i in typeScore:
        print(f"Type {i} mean accuracy: {np.mean(typeScore[i])}")
    print("-------")

    return model, processor
