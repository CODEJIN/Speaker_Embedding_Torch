import torch
import pickle
from random import sample
import numpy as np


from Logger import Logger

def Feature_Stack(features):
    max_feature_length = max([feature.shape[0] for feature in features])
    features = np.stack(
        [np.pad(feature, [[0, max_feature_length - feature.shape[0]], [0, 0]], constant_values= -10.0) for feature in features],
        axis= 0
        )
    return features

ge2e = torch.jit.load('./traced/ge2e.pts', map_location='cpu')
logger = Logger('./temp')

metadata_dict = pickle.load(open(
    '/data/22K.External_No_GPYou/Train/METADATA.PICKLE', 'rb'
    ))

pattern_dict = {
    speaker: Feature_Stack([
        pickle.load(open('/data/22K.External_No_GPYou/Train/{}'.format(x), 'rb'))['Mel'][:120]
        for x in sample(patterns, min(len(patterns), 50))
        ])
    for speaker, patterns in metadata_dict['File_List_by_Speaker_Dict'].items()
    }

embedding_dict = {
    speaker: ge2e(torch.from_numpy(patterns).permute(0, 2, 1))
    for index, (speaker, patterns) in enumerate(pattern_dict.items())
    if index < 30
    }

speakers = []
embeddings = []
for speaker, embedding in embedding_dict.items():
    speakers.extend([speaker] * embedding.size(0))
    embeddings.append(embedding)
embeddings = torch.cat(embeddings, dim= 0).cpu().numpy()

logger.add_embedding(
    embeddings,
    metadata= speakers,
    global_step= 0,
    tag= 'Embeddings'
    )