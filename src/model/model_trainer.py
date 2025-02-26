# model_trainer.py
from src.model.utils import weighted_binary_crossentropy

# adam优化器
def compile_model(model, weights):
    loss_fn = weighted_binary_crossentropy(weights)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])


def train_model(model, train_ds_4d, val_ds_4d, epochs, callbacks):
    # 确保数据集已经设置了批次大小
    assert train_ds_4d.element_spec[0].shape[1:] == (26, 13, 1), "Train dataset must have correct feature shape"
    assert val_ds_4d.element_spec[0].shape[1:] == (26, 13, 1), "Validation dataset must have correct feature shape"

    history = model.fit(
        train_ds_4d,
        validation_data=val_ds_4d,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
        use_multiprocessing=True,
        workers=4
    )
    return history