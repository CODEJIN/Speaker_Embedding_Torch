from argparse import Namespace
import torch
import math

class GE2E(torch.nn.Module):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters

        self.prenet = Conv1d(
            in_channels= self.hp.Sound.Mel_Dim,
            out_channels= self.hp.GE2E.Embedding_Size,
            kernel_size= 1,
            bias= True,
            w_init_gain= 'relu'
            )
        self.relu = torch.nn.ReLU()
        
        self.positional_encoding = Positional_Encoding(
            max_position= self.hp.GE2E.Positional_Encoding.Max_Position,
            embedding_size= self.hp.GE2E.Embedding_Size,
            dropout_rate= self.hp.GE2E.Positional_Encoding.Dropout_Rate
            )
        
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer= torch.nn.TransformerEncoderLayer(
                d_model= self.hp.GE2E.Embedding_Size,
                nhead= self.hp.GE2E.Transformer.Head,
                dim_feedforward= self.hp.GE2E.Embedding_Size * 4,
                dropout= self.hp.GE2E.Transformer.Dropout_Rate
                ),
            num_layers= self.hp.GE2E.Transformer.Num_Layers,
            norm= torch.nn.LayerNorm(
                normalized_shape= self.hp.GE2E.Embedding_Size
                )
            )

        self.projection = Conv1d(
            in_channels= self.hp.GE2E.Embedding_Size,
            out_channels= self.hp.GE2E.Embedding_Size,
            kernel_size= 1,
            bias= True,
            w_init_gain= 'linear'
            )

    def forward(self, features, samples= 1):
        '''
        features: [Batch * Sample, Mel_dim, Time]
        '''
        x = self.prenet(features)   # [Batch, Emb_dim, Time]
        x = self.relu(x)
        x = self.positional_encoding(x)    # [Batch, Emb_dim, Time]
        x = self.transformer(x.permute(2, 0, 1)) # [Time, Batch, Emb_dim]
        x = x.permute(1, 2, 0)[:, :, :1] # [Batch, Emb_dim, 1], Use only first time
        x = x.view(-1, samples, x.size(1), x.size(2)).mean(dim= 1) # [Batch, Emb_dim, 1] -> [Batch, Emb_dim]
        x = self.projection(x).squeeze(2)   # [Batch, Emb_dim]
        x = torch.nn.functional.normalize(x, p=2, dim= 1)

        return x

class Conv1d(torch.nn.Conv1d):
    def __init__(self, w_init_gain= 'relu', *args, **kwargs):
        self.w_init_gain = w_init_gain
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        if self.w_init_gain in ['relu', 'leaky_relu']:
            torch.nn.init.kaiming_uniform_(self.weight, nonlinearity= self.w_init_gain)
        else:
            torch.nn.init.xavier_uniform_(self.weight, gain= torch.nn.init.calculate_gain(self.w_init_gain))
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# https://github.com/soobinseo/Transformer-TTS/blob/master/network.py
class Positional_Encoding(torch.nn.Module):
    def __init__(
        self,
        max_position: int,
        embedding_size: int,
        dropout_rate: float
        ):
        super().__init__()
        self.dropout = torch.nn.Dropout(p= dropout_rate)
        pe = torch.zeros(max_position, embedding_size)
        position = torch.arange(0, max_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(2, 1)
        self.register_buffer('pe', pe)

        self.alpha = torch.nn.Parameter(
            data= torch.ones(1),
            requires_grad= True
            )

    def forward(self, x):
        '''
        x: [Batch, Dim, Length]
        '''
        x = x + self.alpha * self.get_pe(x, self.pe)
        x = self.dropout(x)

        return x

    @torch.jit.script
    def get_pe(x, pe):
        return pe[:, :, :x.size(2)]


class GE2E_Loss(torch.nn.Module):
    def __init__(self, init_weight= 10.0, init_bias= -5.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(init_weight))
        self.bias = torch.nn.Parameter(torch.tensor(init_bias))

        self.consine_similarity = torch.nn.CosineSimilarity(dim= 2)
        self.cross_entroy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, embeddings, pattern_per_speaker):
        '''
        embeddings: [Batch, Emb_dim]
        The target of softmax is always 0.
        '''
        x = embeddings.view(
            embeddings.size(0) // pattern_per_speaker,
            pattern_per_speaker,
            -1
            )   # [Speakers, Pattern_per_Speaker, Emb_dim]

        centroid_for_within = x.sum(dim= 1, keepdim= True).expand(-1, x.size(1), -1)  # [Speakers, Pattern_per_Speaker, Emb_dim]
        centroid_for_between = x.mean(dim= 1)  # [Speakers, Emb_dim]

        within_cosine_similarities = self.consine_similarity(x, centroid_for_within) # [Speakers, Pattern_per_Speaker]
        within_cosine_similarities = self.weight * within_cosine_similarities - self.bias

        between_cosine_simiarity_filter = torch.eye(x.size(0)).to(embeddings.device)
        between_cosine_simiarity_filter = 1.0 - between_cosine_simiarity_filter.unsqueeze(1).expand(-1, x.size(1), -1) # [Speakers, Pattern_per_Speaker, speaker]
        between_cosine_simiarity_filter = between_cosine_simiarity_filter.bool()

        between_cosine_simiarities = self.consine_similarity( #[Speakers * Pattern_per_Speaker, speaker]
            embeddings.unsqueeze(dim= 1).expand(-1, centroid_for_between.size(0), -1),  # [Speakers * Pattern_per_Speaker, Speakers, Emb_dim]
            centroid_for_between.unsqueeze(dim= 0).expand(embeddings.size(0), -1, -1),  #[Speakers * Pattern_per_Speaker, Speakers, Emb_dim]
            )
        between_cosine_simiarities = self.weight * between_cosine_simiarities - self.bias
        between_cosine_simiarities = between_cosine_simiarities.view(x.size(0), x.size(1), x.size(0))   # [Speakers, Pattern_per_Speaker, Speakers]
        between_cosine_simiarities = torch.masked_select(between_cosine_simiarities, between_cosine_simiarity_filter)
        between_cosine_simiarities = between_cosine_simiarities.view(x.size(0), x.size(1), x.size(0) - 1)   # [Speakers, Pattern_per_Speaker, Speakers - 1]
        
        logits = torch.cat([within_cosine_similarities.unsqueeze(2), between_cosine_simiarities], dim = 2)
        logits = logits.view(embeddings.size(0), -1)    # [speaker * Pattern_per_Speaker, speaker]
        
        labels = torch.zeros(embeddings.size(0), dtype= torch.long).to(embeddings.device)
        
        return self.cross_entroy_loss(logits, labels)