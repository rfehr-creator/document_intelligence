import re
import numpy as np
import pytorch_lightning as pl
import torch
from nltk import edit_distance


class DonutModelPLModule(pl.LightningModule):
    def __init__(
        self,
        config,
        processor,
        model,
        startValidate,
        max_length,
        train_dataloader,
        val_dataloader,
    ):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.startValidate = startValidate
        self.max_length = max_length
        self.trainDataloader = train_dataloader
        self.valDataloader = val_dataloader
        self.validation_results = []
        self.all_epoch_scores = []  # List to keep track of all epoch validation scores

    def training_step(self, batch, batch_idx):
        pixel_values, labels, _ = batch

        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        if self.current_epoch < self.startValidate:
            return

        pixel_values, labels, answers = batch
        batch_size = pixel_values.shape[0]
        # we feed the prompt to the model
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.model.config.decoder_start_token_id,
            device=self.device,
        )

        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=self.model.config.decoder.max_length,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(
                self.processor.tokenizer.pad_token, ""
            )
            seq = re.sub(
                r"<.*?>", "", seq, count=1
            ).strip()  # remove first task start token
            predictions.append(seq)

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            answer = answer.replace(self.processor.tokenizer.eos_token, "")
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1 and scores[0] > 0.5:
                print(f"\nPrediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        mean_score = np.mean(scores)
        self.validation_results.append(mean_score)
        return scores

    def on_validation_epoch_end(self):
        meanScore = np.mean(self.validation_results)
        # store for all epochs
        self.all_epoch_scores.append(meanScore)
        
        # log
        self.log("val_edit_distance", meanScore, prog_bar=True, logger=True)
        
        # print
        print(f"\n--- Validation Summary - Epoch {self.current_epoch} ---")
        print(f"Mean Validation Edit Distance = {meanScore:.2f}")
        print(f"Min Validation Edit Distance = {min(self.validation_results):.2f}")
        print(f"Max Validation Edit Distance = {max(self.validation_results):.2f}\n")

        # Clear the validation results for the next epoch
        self.validation_results.clear()
        
        print("All Epoch Validation Scores:")
        print(f"Mean Validation Edit Distance: {np.mean(self.all_epoch_scores):.2f}")
        for epoch, score in enumerate(self.all_epoch_scores):
            print(f"Epoch {epoch}: {score:.2f}")
    
    def on_train_end(self):
        print("\n--- Training Ended ---")
        print("All Epoch Validation Scores:")
        print(f"Mean Validation Edit Distance: {np.mean(self.all_epoch_scores):.2f}")
        for epoch, score in enumerate(self.all_epoch_scores):
            print(f"Epoch {epoch}: {score:.2f}")        
        
    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return self.trainDataloader

    def val_dataloader(self):
        return self.valDataloader
