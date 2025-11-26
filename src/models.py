import torch
import timesfm
from uni2ts.model.moirai import MoiraiForecast
from chronos import ChronosPipeline

# Die CPU-freundlichsten Checkpoints
CHRONOS_CHECKPOINT = "amazon/chronos-t5-tiny"  # Amazon Chronos (ca. 8M Parameter)
MOIRAI_CHECKPOINT = "Salesforce/moirai-1.0-R-small"  # Salesforce Moirai (kleinste Version)
TIMESFM_CHECKPOINT = "google/timesfm-1.0-200m"  # Google TimesFM (kleinere Version)


class ModelFactory:
    """
    Stellt Methoden bereit, um die drei Zeitreihen Foundation Models zu laden.
    Alle Modelle werden explizit auf der CPU initialisiert.
    """

    # ----------------------------------------------------
    # 1. CHRONOS (Amazon)
    # ----------------------------------------------------
    @staticmethod
    def load_chronos(checkpoint=CHRONOS_CHECKPOINT):
        print(f"Lade Chronos von {checkpoint} auf CPU...")

        # Chronos nutzt die HuggingFace Pipeline API
        pipeline = ChronosPipeline.from_pretrained(
            checkpoint,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        return pipeline

    # ----------------------------------------------------
    # 2. MOIRAI (Salesforce / uni2ts)
    # ----------------------------------------------------
    @staticmethod
    def load_moirai(checkpoint=MOIRAI_CHECKPOINT):
        print(f"Lade Moirai von {checkpoint} auf CPU...")

        # Moirai ist in uni2ts integriert und nutzt die gluonts-Prediction-Struktur
        model = MoiraiForecast.load_from_checkpoint(
            checkpoint_path=checkpoint,
            map_location="cpu"
        )
        return model

    # ----------------------------------------------------
    # 3. TIMESFM (Google)
    # ----------------------------------------------------
    @staticmethod
    def load_timesfm(checkpoint=TIMESFM_CHECKPOINT):
        print(f"Lade TimesFM von {checkpoint} auf CPU...")

        # TimesFM ist ein eigenständiges Python-Paket
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend="cpu"),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=checkpoint),
        )
        return tfm