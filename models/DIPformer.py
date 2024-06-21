import torch
from torch import nn
from layers.Encoder import Encoder, EncoderLayer
from layers.SelfAttention import FullAttention, AttentionLayer
from layers.InvertPatchEmbedding import PatchEmbedding, DataEmbedding_inverted


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(256, 17)
        self.linear2 = nn.Linear(4, 1)
        self.dropout = nn.Dropout(head_dropout)



    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x= x.permute(0,2,1)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

class AutoEncoder(nn.Module):   #x(bs, input_lenth, feature_num: 17)
    def __init__(self):
        super().__init__()
        self.Encoder1 = nn.Sequential(
            nn.Linear(17 , 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.ReLU()
        )
        self.Decoder1 = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 17),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.Encoder1(x)
        x = self.Decoder1(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, patch_len=4, stride=2):
        """
        patch_len: int, patch len for invert_patch_embedding,
        stride: int, stride for invert_patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # Invert Patch Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model,
                                                    configs.embed, configs.freq,configs.dropout)
        self.patch_embedding = PatchEmbedding(configs.d_model, patch_len, stride,
                                                    padding, configs.dropout)
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)

        self.head = FlattenHead(configs.enc_in, 128, configs.pred_len,head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # AutoEncoder
        # x_enc = Encoder1(x_enc)
        # x_enc = Decoder1(x_enc)

        # Invert Patch Embedding
        # bs * nvars x patch_num x d_model
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # bs * nvars x patch_num x d_model
        enc_out, attns = self.encoder(enc_out)
        # bs x nvars x patch_num x d_model
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # bs x nvars x d_model x patch_num
        enc_out = enc_out.permute(0, 1, 3, 2)
        # bs x nvars x target_window
        enc_out = self.head(enc_out)
        enc_out = enc_out.permute(0, 2, 1)
        # De-Normalization from Non-stationary Transformer
        enc_out = enc_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        enc_out = enc_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return out[:, -self.pred_len:, :]  # [B, L, D]

