# model_trainer.py
from src.model.utils import weighted_binary_crossentropy, FocalLoss
import tensorflow as tf

# # adam优化器
# def compile_model(model, weights):
#     loss_fn = weighted_binary_crossentropy(weights)
#     model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])


def compile_model(model, gamma, alpha_balance):
    """
    weights: 正样本权重，等效于原加权交叉熵中的pos_weight
             推荐取值：负样本数 / 正样本数
    """
    loss_fn = FocalLoss(
        gamma=gamma,           # 困难样本聚焦参数
        alpha=alpha_balance            # 类别平衡系数
    )

    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='prec'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )



def train_model(model, train_ds_4d, val_ds_4d, epochs, callbacks):
    # 确保数据集已经设置了批次大小
    print(train_ds_4d.element_spec[0].shape[1:])
    assert train_ds_4d.element_spec[0].shape[1:] == (31, 400, 1), "Train dataset must have correct feature shape"
    assert val_ds_4d.element_spec[0].shape[1:] == (31, 400, 1), "Validation dataset must have correct feature shape"

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