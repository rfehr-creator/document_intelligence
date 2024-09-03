from pytorch_lightning import Callback
from prepareModel import save_model_and_processor

class PushToHubCallback(Callback):
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Saving Model... epoch {trainer.current_epoch}")
        save_model_and_processor(pl_module.model, pl_module.processor, self.save_dir)
        # pl_module.model.push_to_hub(
        #     huggingfaceAcct,
        #     commit_message=f"Training in progress, epoch {trainer.current_epoch}",
        # )

    def on_train_end(self, trainer, pl_module):
        print(f"Saving Model...")

        # Push to the hub
        # message = f"Training done. Epoch: {trainer.current_epoch}"
        # pl_module.processor.push_to_hub(huggingfaceAcct, commit_message=message)
        # pl_module.model.push_to_hub(huggingfaceAcct, commit_message=message)

        # Save locally
        save_model_and_processor(pl_module.model, pl_module.processor, self.save_dir)
