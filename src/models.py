import torch
import timesfm
from uni2ts.model.moirai import MoiraiForecast
from chronos import ChronosPipeline

# Die CPU-freundlichsten Checkpoints
CHRONOS_CHECKPOINT = "amazon/chronos-t5-tiny"
MOIRAI_CHECKPOINT = "Salesforce/moirai-1.0-R-small"
TIMESFM_CHECKPOINT = "google/timesfm-1.0-200m-pytorch"



class ModelFactory:
    @staticmethod
    def load_chronos(checkpoint=CHRONOS_CHECKPOINT):
        print(f"Lade Chronos von {checkpoint} auf CPU...")

        pipeline = ChronosPipeline.from_pretrained(
            checkpoint,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        return pipeline

    @staticmethod
    def load_moirai(checkpoint=MOIRAI_CHECKPOINT):
        print(f"Lade Moirai von {checkpoint} auf CPU...")

        model = MoiraiForecast.load_from_checkpoint(
            checkpoint_path=checkpoint,
            map_location="cpu"
        )
        return model

    @staticmethod
    def load_timesfm(checkpoint=TIMESFM_CHECKPOINT):
        print(f"Lade TimesFM von {checkpoint} auf CPU...")

        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend="cpu"),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=checkpoint),
        )
        return tfm


