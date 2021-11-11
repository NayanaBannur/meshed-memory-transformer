import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
from pycocotools.coco import COCO as pyCOCO
import evaluation
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from imageio import imread

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def predict_captions(model, dataloader, text_field, test_dataset):
    import itertools
    model.eval()
    
    file_list = [dict_dataset_test.examples[i].__dict__['image'] for i in range(len(dict_dataset_test.examples))]
    file_list = list(dict.fromkeys(file_list))
    
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            error_list = []
            gen = {}
            gts = {}
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

            gts = evaluation.PTBTokenizer.tokenize(gts)
            gen = evaluation.PTBTokenizer.tokenize(gen)
            _, image_scores = evaluation.compute_scores(gts, gen)
            errors = np.argwhere(np.array(image_scores['BLEU'][0]) < 0.5)
            
            for error in errors:
                idx = error[0]
                filename = file_list[it*10+idx]
                error_list.append((filename, gts['%d_%d' % (it, idx)], gen['%d_%d' % (it, idx)]))
                
            display_errors(error_list)


def display_errors(error_list):
    for (fn, gts, gen) in error_list:
        file = fn.split('/')[-1]
        img = Image.open(f'ms-coco/images/val2014/{file}')
        img_id = int(fn[-16:-4])
        fig, ax = plt.subplots()
        ax.imshow(img)
        all_gt = ', '.join(gts)
        fig.text(0.5, 0.3, f'Ground truth: {all_gt}. Generated: {gen}', 
                    wrap=True, horizontalalignment='center', fontsize=8)
        plt.subplots_adjust(bottom=0.5)
        plt.savefig(f'outputs/{file}')


if __name__ == '__main__':
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_folder', type=str)
    args = parser.parse_args()

    print('Meshed-Memory Transformer Evaluation')

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    _, _, test_dataset = dataset.splits
    text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))

    # Model and dataloaders
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': 40})
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    data = torch.load('meshed_memory_transformer.pth')
    model.load_state_dict(data['state_dict'])

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    error_list = predict_captions(model, dict_dataloader_test, text_field, dict_dataset_test)
    
    display_errors(error_list)
