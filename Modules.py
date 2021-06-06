from argparse import Namespace
import torch

class GE2E(torch.nn.Sequential):
    def __init__(self, hyper_parameters: Namespace):
        super().__init__()
        self.hp = hyper_parameters

        self.add_module('Prenet', Linear(
            in_features= self.hp.Sound.Mel_Dim,
            out_features= self.hp.GE2E.LSTM.Size,
            w_init_gain= 'relu'
            ))
        self.add_module('ReLU', torch.nn.ReLU())
        
        for index in range(self.hp.GE2E.LSTM.Stacks):
            module = Res_LSTM if index < self.hp.GE2E.LSTM.Stacks - 1 else LSTM
            self.add_module('LSTM_{}'.format(index), module(
                input_size= self.hp.GE2E.LSTM.Size,
                hidden_size= self.hp.GE2E.LSTM.Size,
                bias= True,
                batch_first= True
                ))
        self.add_module('Linear', Linear(
            in_features= self.hp.GE2E.LSTM.Size,
            out_features= self.hp.GE2E.Embedding_Size,
            w_init_gain= 'linear'
            ))

    def forward(self, mels, samples= 1):
        '''
        mels: [Batch * Sample, Mel_dim, Time]
        '''
        x = mels.transpose(2, 1)    # [Batch, Time, Mel_dim]
        x = super().forward(x) # [Batch, Time, Emb]
        x = x[:, -1, :] # [Batch, Emb]
        
        # if 'cuda' != x.device: torch.cuda.synchronize()

        x = x.view(-1, samples, x.size(1)).mean(dim= 1) # [Batch, Samples, Emb_dim] -> [Batch, Emb_dim]
        x = torch.nn.functional.normalize(x, p=2, dim= 1)

        return x

class Linear(torch.nn.Linear):
    def __init__(self, w_init_gain= 'linear', *args, **kwagrs):
        self.w_init_gain = w_init_gain
        super(Linear, self).__init__(*args, **kwagrs)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(
            self.weight,
            gain=torch.nn.init.calculate_gain(self.w_init_gain)
            )
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

class Res_LSTM(torch.nn.LSTM):
    def forward(self, input):
        return super().forward(input)[0] + input

class LSTM(torch.nn.LSTM):
    def forward(self, input):
        return super().forward(input)[0]


class GE2E_Loss(torch.nn.Module):
    def __init__(self, init_weight= 10.0, init_bias= -5.0):
        super(GE2E_Loss, self).__init__()
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