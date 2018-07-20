from app.config import ResNetRunConfig

run_config = ResNetRunConfig(
    data_dir='/tmp/data',
    model_dir='/tmp/model',
    train_epochs=150,
    epochs_between_evaluation=30,
    batch_size=64,
    weight_decay=2e-4,
    use_synthetic_data=False,

    resnet_size=32,
    resnet_version=2,
)
