import logging, yaml, os, sys, argparse, math
import torch

from Modules import GE2E
from Arg_Parser import Recursive_Parse

class Tracer(torch.nn.Module):
    def __init__(self, hp_path: str, checkpoint_path: str):
        super().__init__()
        self.hp = Recursive_Parse(yaml.load(
            open(hp_path, encoding='utf-8'),
            Loader=yaml.Loader
            ))

        self.model = GE2E(self.hp)
        self.Load_Checkpoint(path= checkpoint_path)
        self.model.eval()

        for param in self.model.parameters(): 
            param.requires_grad = False

    def Load_Checkpoint(self, path):
        state_dict = torch.load(path, map_location= 'cpu')
        self.model.load_state_dict(state_dict['Model'])
        self.steps = state_dict['Steps']

        logging.info('Checkpoint loaded at {} steps.'.format(self.steps))

    def forward(self, x):
        return self.model(x, samples= 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    parser.add_argument('-c', '--checkpoint_file', required=True, type= str)
    args = parser.parse_args()

    tracer = Tracer(args.hyper_parameters, args.checkpoint_file)    
    x = torch.rand(1, tracer.hp.Sound.Mel_Dim, 400)
    traced_model = torch.jit.trace(tracer, x)
    os.makedirs('traced', exist_ok= True)
    traced_model.save('./traced/ge2e_x.pts')

