from segmentation_models_pytorch import Unet


class efficientnetb5(Unet):
    def __init__(
        self,    
        in_channels: int,
        classes: int
    ):
        Unet(
            encoder_name= 'efficientnet-b5',
            encoder_weights= 'imagenet',
            in_channels = in_channels,
            classes = classes
        )