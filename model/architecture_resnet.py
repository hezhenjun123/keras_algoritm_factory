from tensorflow.keras.applications import ResNet50 as resnet_pretrained
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model


def ResNet(input_shape,
           pretrained,
           channel_list,
           num_classes,
           activation='relu',
           dropout_prob=0,
           name='resnet',
           input_name='images',
           output_name='probs'):
    inp = Input(input_shape, name=input_name)
    resnet = resnet_pretrained(include_top=False,
                               weights='imagenet' if pretrained else None,
                               pooling='max')(inp)

    for dense in channel_list:
        resnet = Dense(dense, activation=activation)(resnet)
        if dropout_prob != 0.0:
            resnet = Dropout(dropout_prob)(resnet)

    preds = Dense(num_classes, activation='softmax', name=output_name)(resnet)
    return Model([inp], [preds], name=name)
