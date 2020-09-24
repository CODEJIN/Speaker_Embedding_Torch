import torch
import logging

# I don't use hyper parameter dict in here
# because several moudles become a submodule
# on other repository.

class GE2E(torch.nn.Module):
    def __init__(self, mel_dims, lstm_size, lstm_stacks, embedding_size):
        super(GE2E, self).__init__()   
        self.lstm_stacks = lstm_stacks

        self.layer_Dict = torch.nn.ModuleDict()

        self.layer_Dict['Prenet'] = Linear(
            in_features= mel_dims,
            out_features= lstm_size,
            )

        for index in range(self.lstm_stacks):
            self.layer_Dict['LSTM_{}'.format(index)] = torch.nn.LSTM(
                input_size= lstm_size,
                hidden_size= lstm_size,
                bias= True,
                batch_first= True
                )
            for name, parameter in self.layer_Dict['LSTM_{}'.format(index)].named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(parameter)
                elif 'bias' in name:
                    torch.nn.init.zeros_(parameter)
        self.layer_Dict['Linear'] = Linear(
            in_features= lstm_size,
            out_features= embedding_size,
            )

    def forward(self, mels, samples= 1):
        '''
        mels: [Batch, Mel_dim, Time]
        '''
        x = mels.transpose(2, 1)    # [Batch, Time, Mel_dim]
        x = self.layer_Dict['Prenet'](x)    # [Batch, Time, LSTM_dim]
        if 'cuda' != x.device: torch.cuda.synchronize()

        for index in range(self.lstm_stacks):
            self.layer_Dict['LSTM_{}'.format(index)].flatten_parameters()
            x = self.layer_Dict['LSTM_{}'.format(index)](x)[0] + \
                (x if index < self.lstm_stacks - 1 else 0)    # [Batch, Time, LSTM_dim]
            if 'cuda' != x.device: torch.cuda.synchronize()

        x = self.layer_Dict['Linear'](x[:, -1, :])   # [Batch, Emb_dim]
        
        if 'cuda' != x.device: torch.cuda.synchronize()

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

class GE2E_Loss(torch.nn.Module):
    def __init__(self, init_weight= 10.0, init_bias= -5.0):
        super(GE2E_Loss, self).__init__()
        self.weight = torch.nn.Parameter(torch.tensor(init_weight))
        self.bias = torch.nn.Parameter(torch.tensor(init_bias))

        self.layer_Dict = torch.nn.ModuleDict()
        self.layer_Dict['Consine_Similarity'] = torch.nn.CosineSimilarity(dim= 2)
        self.layer_Dict['Cross_Entroy_Loss'] = torch.nn.CrossEntropyLoss()

    def forward(self, embeddings, pattern_per_Speaker):
        '''
        embeddings: [Batch, Emb_dim]
        The target of softmax is always 0.
        '''
        x = embeddings.view(
            embeddings.size(0) // pattern_per_Speaker,
            pattern_per_Speaker,
            -1
            )   # [Speakers, Pattern_per_Speaker, Emb_dim]

        centroid_for_Within = x.sum(dim= 1, keepdim= True).expand(-1, x.size(1), -1)  # [Speakers, Pattern_per_Speaker, Emb_dim]
        centroid_for_Between = x.mean(dim= 1)  # [Speakers, Emb_dim]

        within_Cosine_Similarities = self.layer_Dict['Consine_Similarity'](x, centroid_for_Within) # [Speakers, Pattern_per_Speaker]
        within_Cosine_Similarities = self.weight * within_Cosine_Similarities + self.bias    

        between_Cosine_Simiarity_Filter = torch.eye(x.size(0)).to(embeddings.device)
        between_Cosine_Simiarity_Filter = 1.0 - between_Cosine_Simiarity_Filter.unsqueeze(1).expand(-1, x.size(1), -1) # [speaker, pattern_per_Speaker, speaker]
        between_Cosine_Simiarity_Filter = between_Cosine_Simiarity_Filter.bool()

        between_Cosine_Simiarities = self.layer_Dict['Consine_Similarity']( #[speaker * pattern_per_Speaker, speaker]
            embeddings.unsqueeze(dim= 1).expand(-1, centroid_for_Between.size(0), -1),  # [Speakers * Pattern_per_Speaker, Speakers, Emb_dim]
            centroid_for_Between.unsqueeze(dim= 0).expand(embeddings.size(0), -1, -1),  #[Speakers * Pattern_per_Speaker, Speakers, Emb_dim]
            )
        between_Cosine_Simiarities = self.weight * between_Cosine_Simiarities + self.bias
        between_Cosine_Simiarities = between_Cosine_Simiarities.view(x.size(0), x.size(1), x.size(0))   # [speaker, pattern_per_Speaker, speaker]
        between_Cosine_Simiarities = torch.masked_select(between_Cosine_Simiarities, between_Cosine_Simiarity_Filter)
        between_Cosine_Simiarities = between_Cosine_Simiarities.view(x.size(0), x.size(1), x.size(0) - 1)   # [speaker, pattern_per_Speaker, speaker - 1]
        
        logits = torch.cat([within_Cosine_Similarities.unsqueeze(2), between_Cosine_Simiarities], dim = 2)
        logits = logits.view(embeddings.size(0), -1)    # [speaker * pattern_per_Speaker, speaker]
        
        labels = torch.zeros(embeddings.size(0), dtype= torch.long).to(embeddings.device)
        
        return self.layer_Dict['Cross_Entroy_Loss'](logits, labels)