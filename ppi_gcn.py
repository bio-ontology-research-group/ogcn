import click as ck
import pandas as pd

@ck.command()
@ck.option(
    '--train-inter-file', '-trif', default='data/4932.train_interactions.pkl',
    help='Interactions file (deepint_data.py)')
@ck.option(
    '--test-inter-file', '-tsif', default='data/4932.test_interactions.pkl',
    help='Interactions file (deepint_data.py)')
@ck.option(
    '--data-file', '-df', default='data/swissprot.pkl',
    help='Data file with protein sequences')
@ck.option(
    '--deepgo-model', '-dm', default='data/deepgoplus.h5',
    help='DeepGOPlus prediction model')
@ck.option(
    '--model-file', '-mf', default='data/9606.model.h5',
    help='DeepGOPlus prediction model')
@ck.option(
    '--batch-size', '-bs', default=32,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=16,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
def main(train_inter_file, test_inter_file, data_file, deepgo_model, model_file, batch_size, epochs, load):
    pass



def load_ppi_data():
    pass

if __name__ == '__main__':
    main()
