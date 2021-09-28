import timm
import torch
from torch import nn



class ImgModel(nn.Module):
    def __init__(self, backbone='tf_efficientnet_b0_ns'):
        super(ImgModel, self).__init__()

        self.backbone = timm.create_model(model_name=backbone, pretrained=True, in_chans=1)
        self.pool = nn.AdaptiveMaxPool2d(1)

        out_features = self.backbone.classifier.in_features
        self.fc = nn.Linear(out_features, 7)

    def forward(self, x):
        out = self.backbone.forward_features(x)
        out = self.pool(out)
        out = self.fc(out[:,:,0,0])

        return out


if __name__ == '__main__':
    model = ImgModel()

    sample = torch.rand(32, 1, 128, 128)

    out = model(sample)

    print(out.shape)