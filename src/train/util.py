import torch
from deepdiff import DeepDiff
from datasets import load_dataset
import json
from ast import literal_eval
import os

# Load configuration from JSON file
with open("config.json", "r") as f:
    config = json.load(f)

model_config = config["model"]
task_prompt = model_config["task_prompt"]


def verifyBatch(
    train_dataloader, train_dataset, val_dataloader, val_dataset, processor
):
    print("\n-- Verifying batch --")
    batch = next(iter(train_dataloader))
    pixel_values, labels, target_sequences = batch
    print(pixel_values.shape)

    for id in labels.squeeze().tolist()[:30]:
        if id != -100:
            print("id: {} == decode: {}".format([id], processor.decode([id])))
        else:
            print("id", id)

    print(f"train_len: {len(train_dataset)}")
    print(f"val_len: {len(val_dataset)}")

    # let's check the first validation batch
    batch = next(iter(val_dataloader))
    pixel_values, labels, target_sequences = batch
    print("validation batch shape", pixel_values.shape)
    print("target sequence", target_sequences[0])

    print("-- Verified batch --\n")


def checkTokens(added_tokens, processor):
    print("\n-- Checking tokens --")
    """Let's check which tokens are added:"""
    print(f"added tokens: {len(added_tokens)}")
    print(f"added_tokens: {added_tokens}")

    # the vocab size attribute stays constants (might be a bit unintuitive - but doesn't include special tokens)
    print("Original number of tokens (vocab):", processor.tokenizer.vocab_size)
    print("Number of tokens after adding special tokens:", len(processor.tokenizer))

    """You can verify that a token like `</s_unitprice>` was added to the vocabulary of the tokenizer (and the model):"""
    print(f"decode [57543]: {processor.decode([57543])}")
    print(f"decode [57540]: {processor.decode([57540])}")
    print("-- Checked tokens --\n")


def compareConfig(original_config, loaded_config):
    config_diff = DeepDiff(original_config, loaded_config, ignore_order=True)
    if config_diff:
        print("Differences in model configurations:")
        print(config_diff)
    else:
        print("Model configurations are identical.")


def compareTokenization(original_tokens, loaded_tokens):
    assert original_tokens and loaded_tokens, "Tokenization lists are empty."
    tokenizer_diff = DeepDiff(original_tokens, loaded_tokens, ignore_order=True)
    if tokenizer_diff:
        print("Differences in tokenization:")
        print(tokenizer_diff)
    else:
        print("Tokenization is identical.")


def compareStateDict(original_state_dict, loaded_state_dict):
    state_dict_diff = {}
    for key in original_state_dict:
        original_tensor = original_state_dict[key]
        loaded_tensor = loaded_state_dict[key].to(
            "cuda"
        )  # Move loaded tensor to the same device as original tensor
        if not torch.equal(original_tensor, loaded_tensor):
            state_dict_diff[key] = (original_tensor, loaded_tensor)

    if state_dict_diff:
        print("Differences in model weights:")
        for key, (original, loaded) in state_dict_diff.items():
            print(f"Key: {key}")
            print(f"Original: {original}")
            print(f"Loaded: {loaded}")
    else:
        print("Model weights are identical.")


def singleImageTokenization(processor, sample, device):
    # Prepare encoder inputs
    pixel_values = processor(
        sample["image"].convert("RGB"), return_tensors="pt"
    ).pixel_values
    pixel_values = pixel_values.to(device)

    # Prepare decoder inputs
    return processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids


def singleImageOutput(model, processor, sample, device):
    model.eval()
    model.to(device)

    # Prepare encoder inputs
    pixel_values = processor(
        sample["image"].convert("RGB"), return_tensors="pt"
    ).pixel_values
    pixel_values = pixel_values.to(device)

    # Prepare decoder inputs
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    decoder_input_ids = decoder_input_ids.to(device)

    # Autoregressively generate sequence
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
    return outputs.sequences


def compareModelOutput(
    original_model,
    original_processor,
    loaded_model,
    loaded_processor,
    datasetPath,
    device,
):
    original_model.to(device)
    loaded_model.to(device)

    dataset = load_dataset(datasetPath, split="train")
    sample = dataset[0]  # Select the first image from the dataset

    if (
        original_model.decoder.config.max_position_embeddings
        != loaded_model.decoder.config.max_position_embeddings
    ):
        print("The models have different max_position_embeddings.")

    if (
        original_model.config.encoder.image_size
        != loaded_model.config.encoder.image_size
    ):
        print("The models have different image sizes.")

    if (
        original_processor.tokenizer.pad_token_id
        != loaded_processor.tokenizer.pad_token_id
    ):
        print("The tokenizers have different pad_token_ids.")

    if (
        original_processor.tokenizer.eos_token_id
        != loaded_processor.tokenizer.eos_token_id
    ):
        print("The tokenizers have different eos_token_ids.")

    if (
        original_processor.tokenizer.unk_token_id
        != loaded_processor.tokenizer.unk_token_id
    ):
        print("The tokenizers have different unk_token_ids.")

    if singleImageTokenization(
        original_processor, sample, device
    ) != singleImageTokenization(loaded_processor, sample, device):
        print("The tokens are different.")

    original_predict = singleImageOutput(
        original_model, original_processor, sample, device
    )
    loaded_predict = singleImageOutput(loaded_model, loaded_processor, sample, device)

    # Compare the outputs
    if torch.equal(original_predict, loaded_predict):
        print("The outputs are identical.")
    else:
        print("The model outputs differ.")


# Function to compare model configurations
def compare_model_configs(model1, model2):
    config1 = model1.config.to_dict()
    config2 = model2.config.to_dict()

    if config1 == config2:
        print("The configurations are the same.")
    else:
        print("The configurations are different.")
        print("Config 1:", config1)
        print("Config 2:", config2)


def datasetCorrectFormat(dataset):
    example = dataset[0]
    ground_truth = example["ground_truth"]
    print(f"ground_truth: {ground_truth}")
    literal_eval(ground_truth)["gt_parse"]


def verifyTrainingDataCorrectFormat(trainingDataset):
    """As always, it's very important to verify whether our data is prepared correctly. Let's check the first training example:"""
    pixel_values, labels, target_sequence = trainingDataset[0]

    """This returns the `pixel_values` (the image, but prepared for the model as a PyTorch tensor), the `labels` (which are the encoded `input_ids` of the target sequence, which we want Donut to learn to generate) and the original `target_sequence`. The reason we also return the latter is because this will allow us to compute metrics between the generated sequences and the ground truth target sequences."""
    print(f"pixel_value.shape: {pixel_values.shape}")
    print(f"labels.shape: {labels.shape}")
    print(f"target_sequence: {target_sequence}")


# Save the model, processor, and optimizer state
def save_checkpoint(model, processor, optimizer, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "model")
    processor_save_path = os.path.join(save_dir, "processor")
    optimizer_save_path = os.path.join(save_dir, "optimizer.pt")

    model.save_pretrained(model_save_path)
    processor.save_pretrained(processor_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path)


# Load the model, processor, and optimizer state
def load_checkpoint(model_class, processor_class, save_dir, device):
    model_save_path = os.path.join(save_dir, "model")
    processor_save_path = os.path.join(save_dir, "processor")
    optimizer_save_path = os.path.join(save_dir, "optimizer.pt")

    model = model_class.from_pretrained(model_save_path).to(device)
    model.train()
    processor = processor_class.from_pretrained(processor_save_path)
    optimizer_state_dict = torch.load(optimizer_save_path)

    return model, processor, optimizer_state_dict
