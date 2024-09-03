import os
from classes.JsonKey import Key
from classes.JsonTokenConstants import TokenConstants
import json

# Load configuration from JSON file
with open("config.json", "r") as f:
    config = json.load(f)

model_config = config["model"]
task_prompt = model_config["task_prompt"]


def createTokens():
    tokens = []
    # Add tokens related to Key for XML purposes
    for i in Key.__dict__:
        if not i.startswith("__"):
            tokens.append(f"<s_{i}>")
            tokens.append(f"</s_{i}>")

    # Add other tokens that are part of the data
    for i in TokenConstants.__dict__:
        if not i.startswith("__"):
            tokens.append(f"{i}")

    tokens.append(task_prompt)
    return tokens


def addTokens(model, processor):
    addTokens = createTokens()
    new_tokens = [
        token for token in addTokens if token not in processor.tokenizer.get_vocab()
    ]

    if new_tokens:
        processor.tokenizer.add_tokens(new_tokens)
        model.decoder.resize_token_embeddings(len(processor.tokenizer))


def configureModelProcessor(model, processor, imageSize, maxLength):
    processor.image_processor.size = imageSize
    processor.image_processor.do_align_long_axis = False
    processor.tokenizer.model_max_length = maxLength

    model.config.encoder.image_size = imageSize
    model.config.decoder.max_length = maxLength
    model.config.decoder.vocab_size = len(processor.tokenizer)

    model.config.pad_token_id = processor.tokenizer.pad_token_id

    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        [task_prompt]
    )[0]


def save_model_and_processor(model, processor, saveDir):
    os.makedirs(saveDir, exist_ok=True)

    model_save_path = os.path.join(saveDir, "model")
    processor_save_path = os.path.join(saveDir, "processor")

    model.save_pretrained(model_save_path)
    processor.save_pretrained(processor_save_path)


def load_model_and_processor(model_class, processor_class, saveDir, device):
    model_save_path = os.path.join(saveDir, "model")
    processor_save_path = os.path.join(saveDir, "processor")

    model = model_class.from_pretrained(model_save_path)
    processor = processor_class.from_pretrained(processor_save_path)

    model.to(device)
    return model, processor


def assertModelProcessorConfig(model, processor):
    print("\n-- Asserting model and processor configurations --")
    print("model.config.encoder.image_size:", model.config.encoder.image_size)
    print("processor.image_processor.size:", processor.image_processor.size)
    print("model.config.decoder.max_length:", model.config.decoder.max_length)
    print("processor.tokenizer.model_max_length:", processor.tokenizer.model_max_length)
    print("model.config.decoder_start_token_id:", model.config.decoder_start_token_id)
    print("processor.cls_token_id:", processor.tokenizer.cls_token_id)
    print("processor.sep_token_id:", processor.tokenizer.sep_token_id)
    print("processor.pad_token_id:", processor.tokenizer.pad_token_id)
    print("processor.mask_token_id:", processor.tokenizer.mask_token_id)
    print("processor.unk_token_id:", processor.tokenizer.unk_token_id)

    assert (
        model.config.decoder.max_length == processor.tokenizer.model_max_length
    ), "Model max length and tokenizer max length are different."
    assert model.config.decoder.vocab_size == len(
        processor.tokenizer
    ), "Model vocab size and tokenizer vocab size are different."
    assert (
        model.config.pad_token_id == processor.tokenizer.pad_token_id
    ), "Model pad token ID and tokenizer pad token ID are different."
    assert (
        model.config.decoder_start_token_id
        == processor.tokenizer.convert_tokens_to_ids([task_prompt])[0]
    ), "Model decoder start token ID and tokenizer start token ID are different."

    print("image_size:", model.config.encoder.image_size)
    print("max_length:", model.config.decoder.max_length)
    print("-- end --\n")
