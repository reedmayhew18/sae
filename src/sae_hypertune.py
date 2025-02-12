from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray import train, tune
from ray.tune import Tuner, with_resources
# from ray_lightning.tune import TuneReportCallback
from ray.train import RunConfig

from lightning_module import LightningSparseAutoencoder
from dataset import ActivationDataset, create_data_loaders

# Training Function with Ray Tune
def train_model(config):

    train_loader, val_loader = create_data_loaders(config["HB"]["batch_size"], config["scale_factor"])

    model = LightningSparseAutoencoder(
        input_dim=config["input_dim"], 
        hidden_dim=config["HB"]["hidden_dim"], 
        l1_lambda=config["l1_lambda"], 
        lr=config["lr"]
    )

    logger = TensorBoardLogger("tb_logs", name="SparseAutoencoder")

    trainer = pl.Trainer(
        max_epochs=config["num_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=logger,
        profiler="advanced",
        # val_check_interval=0.25,  # Check validation 4 times per epoch
        # max_time="00:30:00",  # Stop after 30 minutes
        enable_progress_bar=True, # Show progress bar
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(monitor="val_loss", patience=3, mode="min"),
            TuneReportCheckpointCallback(
                {
                    "train_loss": "train_loss",
                    "train_mse_loss": "train_mse_loss",
                    "train_l1_loss": "train_l1_loss",
                    "active_features": "active_features",
                    "val_loss": "val_loss",
                },
                filename="none",  # Do not save checkpoints
                save_checkpoints = False,
                on="train_batch_end",
            ),
            TuneReportCheckpointCallback(
                {
                    "val_loss": "val_loss",
                    "val_mse_loss": "val_mse_loss",
                    "val_l1_loss": "val_l1_loss",
                    "val_active_features": "val_active_features",
                },
                filename="none1",
                save_checkpoints = False,
                on="validation_end",
            ),
            # RayTrainReportCallback(),
        ],
        # strategy=RayDDPStrategy(), # Use Ray for distributed training, DDP stands for Distributed Data Parallel
        # callbacks=[RayTrainReportCallback()], # Report metrics to Ray
        # plugins=[RayLightningEnvironment()], # Use Ray for distributed training
    )
    trainer.fit(model, train_loader, val_loader)

# Ray Tune Hyperparameter Search
def tune_hyperparameters():

    possible_batch_sizes = [2048, 4096, 8192]
    possible_hidden_dims = [20000, 32768, 65536]

    valid_hb_pairs = []
    for hidden_dim in possible_hidden_dims:
        for batch_size in possible_batch_sizes:
            if hidden_dim * batch_size <= 441_000_000: # VRAM limit
                valid_hb_pairs.append({"hidden_dim": hidden_dim, "batch_size": batch_size})


    # search_space = {
    #     "hidden_dim": tune.choice([4096, 8192, 16384, 20000, 32768]),
    #     "batch_size": tune.choice([512, 1024, 2048, 4096, 8192]),
    #     "l1_lambda": tune.loguniform(1e-4, 1e-2),
    #     "lr": tune.loguniform(1e-4, 1e-2),
    # }

    search_space = {
        "num_epochs": 10,
        "input_dim": 3072,
        "scale_factor": 11.888623072966611,
        "HB": tune.choice(valid_hb_pairs),
        "lr": tune.loguniform(1e-5, 1e-1),
        "l1_lambda": tune.loguniform(1e-3, 1e-1),
    }

    scheduler_asha = tune.schedulers.ASHAScheduler(
        time_attr="training_iteration",
        metric="train_loss",
        mode="min",
        max_t=80,
        grace_period=40, # at least 25 batches
        reduction_factor=2,
    )

    # os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0" # Allows relative paths, but trials are not isolated

    trainable_with_resources = with_resources(
        train_model,
        {"cpu": 4, "gpu": 1}  # Adjust based on your available resources
    )

    tuner = Tuner(
        trainable=trainable_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=20, # Number of hyperparameter sets to try
            max_concurrent_trials=3, # Number of trials to run concurrently
            scheduler=scheduler_asha,
        ),
        run_config=RunConfig(
            name="hyperparameter_search",
            storage_path=str(Path("./results").resolve()),
        ),
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric="val_loss", mode="min")
    print("Best Hyperparameters Found:")
    print(best_result.config)
    return results

# Run Hyperparameter Search
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    tune_hyperparameters()


