import evaluate
import train


SAVE_MODEL_PATH = "saved_models/pageOne"
DATASET_PATH_TRAIN = "../../data/dataset/train"
DATASET_PATH_VAL = "../../data/dataset/validation"


model, processor = train.loadPretrained(SAVE_MODEL_PATH)

trainDataLoader, valDataLoader = train.loadDataset(
    DATASET_PATH_TRAIN, DATASET_PATH_VAL, processor
)

train.train(model, processor, trainDataLoader, valDataLoader)

evaluate.evaluate(DATASET_PATH_VAL, SAVE_MODEL_PATH, "cuda", model, processor, 0.2)
