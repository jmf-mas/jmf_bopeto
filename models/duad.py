import torch.nn as nn
from .ae import AEDetecting as AE
from .base import BaseModel


class DUAD(BaseModel):
    def __init__(self, params):
        self.params = params
        self.name = "duad"
        self.ae = None
        super(DUAD, self).__init__(params)
        self.cosim = nn.CosineSimilarity()

    def resolve_params(self):
        enc_layers = [
            (self.in_features, 60, nn.Tanh()),
            (60, 30, nn.Tanh()),
            (30, 10, nn.Tanh()),
            (10, self.params.ae_latent_dim, None)
        ]
        dec_layers = [
            (self.params.ae_latent_dim, 10, nn.Tanh()),
            (10, 30, nn.Tanh()),
            (30, 60, nn.Tanh()),
            (60, self.in_features, None)
        ]
        self.ae = AE(self.params, enc_layers, dec_layers).to(self.device)

    def encode(self, x):
        return self.ae.encoder(x)

    def decode(self, code):
        return self.ae.decoder(code)

    def forward(self, x):
        code = self.ae.encoder(x)
        x_prime = self.ae.decoder(code)
        h_x = self.cosim(x, x_prime)
        return code, x_prime, h_x

    def get_params(self) -> dict:
        return {
            "duad_p": self.p,
            "duad_p0": self.p0,
            "duad_r": self.r
        }
