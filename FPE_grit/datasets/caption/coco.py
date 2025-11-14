# ------------------------------------------------------------------------
# Modified from Meshed Memory Transformer
# https://github.com/aimagelab/meshed-memory-transformer
# ------------------------------------------------------------------------
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

from .transforms import *
from .transforms import get_transform
from .transforms.utils import MinMaxResize
from .field import ImageField, TextField


from .example import Example
from pycocotools.coco import COCO as pyCOCO
from engine.utils import nested_tensor_from_tensor_list

from .get_frequency_map import Get_frequency_map

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler, BatchSampler

OVERFIT_SIZE = 500
OVERFIT_SIZE_SC=500

FILTER_MODEL= 1   #    [0,1,2] means [low, high, both]
CUT_OFF_FREQUENCY_SET= [20,70]  

class CPairedDataset:

    def __init__(self, examples, image_field, overfit=False):
        self.examples = examples
        self.image_field = image_field
        self.overfit = overfit
        self.get_freq_map = Get_frequency_map(cut_off_frequency = CUT_OFF_FREQUENCY_SET, filter_model= FILTER_MODEL)


    def __getitem__(self, idx):
        example = self.examples[idx]    
        img_path, caption = example.image, example.tokens
        image_id = example.image_id

        img = self.image_field.preprocess(img_path)   

        img_freq = self.get_freq_map.get_img_fft(img)   
        
        return img, caption, image_id, img_freq

    def __len__(self):
        if self.overfit:
            return OVERFIT_SIZE
        return len(self.examples)   


class CDictionaryDataset:

    def __init__(self, examples, image_field, overfit=False):
        self.overfit = overfit
        self.image_field = image_field
        self.get_freq_map = Get_frequency_map(cut_off_frequency=CUT_OFF_FREQUENCY_SET, filter_model= FILTER_MODEL)
        self.img2captions = {} 
        self.img2image_id = {}
 
        for example in examples:
            if example.image not in self.img2captions:   
                self.img2captions[example.image] = []
            self.img2captions[example.image].append(example.text)  
            self.img2image_id[example.image] = example.image_id    
        self.img_paths = list(self.img2captions.keys())

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image_id = self.img2image_id[img_path]
        captions = self.img2captions[img_path]

        img = self.image_field.preprocess(img_path)
        img_freq = self.get_freq_map.get_img_fft(img)  

        return img, captions, image_id , img_freq



    def __len__(self):
        if self.overfit:
            return OVERFIT_SIZE_SC
        return len(self.img_paths)   




class DictionaryCollator:

    def __init__(self, img_field, device='cpu'):
        self.img_field = img_field
        self.device = device

    def __call__(self, batch):
        imgs = [item[0] for item in batch]
        captions = [item[1] for item in batch]
        image_ids = [item[2] for item in batch]
        imgs_freq = [item[3] for item in batch]

        outputs = {}
        if self.img_field.use_hdf5_feat:
            samples = {}
            if self.img_field.use_gri_feat:
                samples['gri_feat'] = torch.stack([im['gri_feat'] for im in imgs]).to(self.device)
                samples['gri_mask'] = torch.stack([im['gri_mask'] for im in imgs]).to(self.device)
            if self.img_field.use_reg_feat:
                samples['reg_feat'] = torch.stack([im['reg_feat'] for im in imgs]).to(self.device)
                samples['reg_mask'] = torch.stack([im['reg_mask'] for im in imgs]).to(self.device)
            outputs['samples'] = samples
        else:
            outputs['samples'] = nested_tensor_from_tensor_list(imgs).to(self.device)
            # nested_tensor_from_tensor_list 接收一个由 Tensor 对象组成的列表 tensor_list，并返回一个 NestedTensor 对象。
            # NestedTensor 是一个包含张量和掩码的容器，常用于处理不同尺寸的图像批次
            outputs['samples_freq'] = nested_tensor_from_tensor_list(imgs_freq).to(self.device)


        outputs['captions'] = captions #列表， 1句caption的原始tokens列表(XE)  或 5句文本列表(SC)
        outputs['image_id'] = image_ids
        outputs["use_hdf5_feat"] = self.img_field.use_hdf5_feat 

        return outputs
        # ['samples']   从hdf5读的gri和reg特征  或  加载transform变换后、且尺寸统一的image 
        # ['captions']  caption 
        # ['image_id']  图像索引
        # ['samples_freq']    加载transform变换后、且尺寸统一,经过dft的image


class PairedCollator(DictionaryCollator):

    def __init__(self, img_field, device='cpu', max_len=54, pad_idx=1, bos_idx=2, eos_idx=3):
        super().__init__(img_field, device)
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def __call__(self, batch):
        b = super().__call__(batch)

        # truncate
        captions = [c[:self.max_len] for c in b['captions']]       # 截取最大长度 54
        max_len = max([len(c) for c in b['captions']])

        padded = []
        for c in captions:
            #              开始  2            结束 3           填充 1
            caption = [self.bos_idx] + c + [self.eos_idx] + [self.pad_idx] * (max_len - len(c))
            padded.append(caption)

        padded = [torch.Tensor(caption).long() for caption in padded]
        padded = pad_sequence(padded, batch_first=True).to(self.device)

        b['captions'] = padded
        # b['captions'] 张量，经截断、填充、以及添加开始和结束标记，形状为 [batch_size, max_len]。
        
        return b
        # ['samples']    从hdf5读的gri和reg特征  或  加载transform变换后、且尺寸统一的image 
        # ['captions']   1个经过截断、增加占位符、pad符后的caption token (长度一致)
        # ['image_id']   图像索引
        # ['samples_freq']    加载transform变换后、且尺寸统一,经过dft的image



class TestCollator:

    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, batch):
        imgs = [item[0] for item in batch]
        image_ids = [item[1] for item in batch]


        outputs = {}
        outputs['samples'] = nested_tensor_from_tensor_list(imgs).to(self.device)
        outputs['image_id'] = image_ids
        # 推理时没有加频率
        return outputs


class TestDataset:

    def __init__(self, root, anno_file, transform=None, overfit=False, from_idx=0, to_idx=-1):
        annotations = json.load(open(anno_file))['images']
        total_images = len(annotations)
        print("Total images:", total_images)
        print(f"from idx: {from_idx} to idx: {to_idx}")

        if to_idx >= total_images - 1 or to_idx == -1:
            self.annotations = annotations[from_idx:]
        else:
            self.annotations = annotations[from_idx:to_idx]

        self.root = root
        self.transform = transform

        if self.transform is None:
            self.transform = Compose([MinMaxResize((384, 640)), ToTensor(), normalize()])
        

    def __getitem__(self, idx):
        item = self.annotations[idx]
        img_path = os.path.join(self.root, item['file_name'])
        img_id = item['id']
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img_trans = self.transform(img)
        
        return img, img_id

    def __len__(self):
        # return len(self.annotations)
        return



class COCO(CPairedDataset):

    def __init__(
        self,
        image_field,
        text_field,
        img_root,
        ann_root,
        use_restval=True,
        cut_validation=False,
        overfit=False,
        **kwargs,
    ):
        self.image_field = image_field
        self.text_field = text_field
        self.overfit = overfit

        # Karpathy split  , where 113,287, 5,000, and 5,000 images are used for training, validation, and testing respectively.
        roots = {}
        roots['train'] = {
            'img': os.path.join(img_root, 'train2014'),
            'cap': os.path.join(ann_root, 'captions_train2014.json')
        }
        roots['valid'] = {
            'img': os.path.join(img_root, 'val2014'),
            'cap': os.path.join(ann_root, 'captions_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(img_root, 'val2014'),
            'cap': os.path.join(ann_root, 'captions_val2014.json')
        }
        # trainrestval 将训练和验证数据一起塞进去
        roots['trainrestval'] = {                                       #   82783 images      40504 images
            'img': (roots['train']['img'], roots['valid']['img']),      # ( .../train2014 , .../val2014 )
            'cap': (roots['train']['cap'], roots['valid']['cap'])       # ( .../captions_train2014.json , .../captions_val2014.json )
        }

        # In trainrestval, 123287 images, 616767 captions.              all COCO (with val/test 5000/5000)
        # In trainrestval(train), 82783 images,  414113 captions.
        # In trainrestval(valid), 40504 images,  202654 captions.

        if ann_root is not None:
            ids = {}
            ids['train'] = np.load(os.path.join(ann_root, 'coco_train_ids.npy'))    
            ids['valid'] = np.load(os.path.join(ann_root, 'coco_dev_ids.npy'))
            if cut_validation:
                ids['valid'] = ids['valid'][:5000]
            ids['test'] = np.load(os.path.join(ann_root, 'coco_test_ids.npy'))
            ids['trainrestval'] = (ids['train'], np.load(os.path.join(ann_root, 'coco_restval_ids.npy')))

            if use_restval:
                roots['train'] = roots['trainrestval']
                ids['train'] = ids['trainrestval']
        else:
            ids = None

        self.train_examples, self.valid_examples, self.test_examples = self._get_samples(roots, ids)

    def split_examples(self):
        return {
            'train': self.train_examples,
            'valid': self.valid_examples,
            'test': self.test_examples,
        }

    def splits(self):# 无用
        train_split = CPairedDataset(self.train_examples, self.image_field, self.text_field, overfit=self.overfit)
        valid_split = CPairedDataset(self.valid_examples, self.image_field, self.text_field, overfit=self.overfit)
        test_split = CPairedDataset(self.test_examples, self.image_field, self.text_field, overfit=self.overfit)
        return train_split, valid_split, test_split

    def _get_samples(self, roots, ids_dataset=None):
        train_samples = []
        valid_samples = []
        test_samples = []
        heights = []
        widths = []

        # self.overfit = ture 仅分两组，无train 
        splits = ['valid', 'test'] if self.overfit else ['train', 'valid', 'test']
        for split in splits:
            if isinstance(roots[split]['cap'], tuple): 
                coco_dataset = (pyCOCO(roots[split]['cap'][0]), pyCOCO(roots[split]['cap'][1]))  
                root = roots[split]['img']                     
            else:
                coco_dataset = (pyCOCO(roots[split]['cap']),)    
                root = (roots[split]['img'],)                     

            if ids_dataset is None:
                ids = list(coco_dataset.anns.keys())     
            else:
                ids = ids_dataset[split]        #  ids['train'] = ids['trainrestval'] = (413915, 152520)  (Karpathy split train 113,287）

            if isinstance(ids, tuple):
                bp = len(ids[0])
                ids = list(ids[0]) + list(ids[1])    #  ids['train'] = ids['trainrestval'] = 413915 + 152520,  (Karpathy split train 113,287）
            else:
                bp = len(ids)

            # print(f"load {len(ids)} img/cap in {split}. ")  # load 566435 img/cap in train  (Karpathy split train 113,287）
            

            for index in tqdm(range(len(ids))):
                if index < bp:
                    coco = coco_dataset[0]
                    img_root = root[0]
                else:
                    coco = coco_dataset[1]
                    img_root = root[1]

                ann_id = ids[index]
                caption = coco.anns[ann_id]['caption']
                img_id = coco.anns[ann_id]['image_id']
                filename = coco.loadImgs(img_id)[0]['file_name']
                height = coco.loadImgs(img_id)[0]['height']
                width = coco.loadImgs(img_id)[0]['width']
                heights.append(height)
                widths.append(width)

                example = Example.fromdict({
                    'image_id': img_id,
                    'image': os.path.join(img_root, filename),
                    'text': caption,  
                    'tokens': [self.text_field.vocab.stoi[w] for w in self.text_field.preprocess(caption)]
                })

                if split == 'train':
                    train_samples.append(example)
                elif split == 'valid':
                    valid_samples.append(example)
                elif split == 'test':
                    test_samples.append(example)

        if self.overfit:
            train_samples = valid_samples

        return train_samples, valid_samples, test_samples


def build_coco_dataloaders(config=None, mode='freezing', device='cpu'):
    overfit = config.dataset.overfit
    transform = get_transform(config.dataset.transform_cfg)
    


    text_field = TextField(vocab_path=config.dataset.vocab_path)

    valid_field = ImageField(transform=transform['valid'], **config.dataset)
    train_field = ImageField(transform=transform['train'], **config.dataset)
    

    examples = COCO(None, text_field, **config.dataset).split_examples()



    if mode == 'freezing' and config.optimizer.freezing_xe_epochs > 0:
        valid_field.init_hdf5_feat()
        train_field.init_hdf5_feat()

    if mode == 'finetune':
        valid_field.use_hdf5_feat = False
        train_field.use_hdf5_feat = False


    datasets = {
        'train': CPairedDataset(examples['train'], train_field, overfit=overfit),           
        'valid': CPairedDataset(examples['valid'], valid_field, overfit=overfit),
        'train_dict': CDictionaryDataset(examples['train'], train_field, overfit=overfit),   
        'valid_dict': CDictionaryDataset(examples['valid'], valid_field, overfit=overfit),
        'test_dict': CDictionaryDataset(examples['test'], valid_field, overfit=overfit),
    }

    collators = {
        'train': PairedCollator(train_field, device=device),  
        'valid': PairedCollator(valid_field, device=device),
        'train_dict': DictionaryCollator(train_field, device=device),  
        'valid_dict': DictionaryCollator(valid_field, device=device),
        'test_dict': DictionaryCollator(valid_field, device=device),
    }

    batch_size = config.optimizer.batch_size * 4 if mode == 'freezing' else config.optimizer.batch_size
    sc_batch_size = config.optimizer.batch_size if mode == 'freezing' else config.optimizer.batch_size // 4

    dataloaders = {}
    
    # 对 test and valid 无Sampler，顺序采样
    dataloaders['valid_dict'] = DataLoader(
        datasets['valid_dict'],
        batch_size=max(1, sc_batch_size * 2),
        num_workers=config.optimizer.num_workers,
        collate_fn=collators['valid_dict'],   # 特征堆叠  cap无截断
    )
    dataloaders['test_dict'] = DataLoader(
        datasets['test_dict'],
        batch_size=max(1, sc_batch_size * 2),
        num_workers=config.optimizer.num_workers,
        collate_fn=collators['test_dict'],  # 特征堆叠  cap无截断
    )

    if getattr(config.exp, 'eval', False):
        return dataloaders

    samplers = {
        'train': DistributedSampler(datasets['train'], shuffle=True),    # XE
        'valid': DistributedSampler(datasets['valid'], shuffle=False),
        'train_dict': DistributedSampler(datasets['train_dict'], shuffle=True)  # SC
    }


    batch_train_sampler = BatchSampler(samplers['train'], batch_size, drop_last=True)
    batch_train_dict_sampler = BatchSampler(samplers['train_dict'], max(2, sc_batch_size), drop_last=True)


    dataloaders['train'] = DataLoader(
        datasets['train'],
        batch_sampler=batch_train_sampler,     # 有batch_sampler，批量随机（分布式）采样
        collate_fn=collators['train'],
        num_workers=config.optimizer.num_workers,
    )
    dataloaders['valid'] = DataLoader(
        datasets['valid'],
        batch_size=batch_size,
        sampler=samplers['valid'],     # 有sampler，非批量顺序（分布式）采样
        collate_fn=collators['valid'],
        num_workers=config.optimizer.num_workers,
    )
    dataloaders['train_dict'] = DataLoader(
        datasets['train_dict'],
        batch_sampler=batch_train_dict_sampler,  # 有batch_sampler，批量随机（分布式）采样
        collate_fn=collators['train_dict'],
        num_workers=config.optimizer.num_workers,
    )
    return dataloaders, samplers



def build_test_dataloaders(config=None, device='cpu', from_idx=0, to_idx=-1):

    datasets = {
        'test':
            TestDataset(
                root=os.path.join(os.environ['DATA_ROOT'], 'test2014'),
                anno_file=os.path.join(os.environ['DATA_ROOT'], 'annotations/image_info_test2014.json'),
                from_idx=from_idx,
                to_idx=to_idx,
            ),
        'valid':
            TestDataset(
                root=os.path.join(os.environ['DATA_ROOT'], 'val2014'),
                anno_file=os.path.join(os.environ['DATA_ROOT'], 'annotations/captions_val2014.json'),
                from_idx=from_idx,
                to_idx=to_idx,
            ),
        'coco_Ka_split_test':
            TestDataset(
                root=os.path.join(os.environ['DATA_ROOT'], 'val2014'),
                anno_file="/gemini/code/hericap/data/coco_karpathy_test_gt.json",
                from_idx=from_idx,
                to_idx=to_idx,
            ),
        
    }
    collator = TestCollator(device=device)
    sc_batch_size = config.optimizer.batch_size

    dataloaders = {}
    # test and valid
    dataloaders['test'] = DataLoader(
        datasets['test'],
        batch_size=16,  # max(2, sc_batch_size),
        num_workers=4,  # config.optimizer.num_workers,
        collate_fn=collator,
    )
    dataloaders['valid'] = DataLoader(
        datasets['valid'],
        batch_size=16,  # max(2, sc_batch_size),
        num_workers=4,  # config.optimizer.num_workers,
        collate_fn=collator,
    )

    dataloaders['coco_Ka_split_test'] = DataLoader(
        datasets['coco_Ka_split_test'],
        batch_size=16,  # max(2, sc_batch_size),
        num_workers=2,  # config.optimizer.num_workers,
        collate_fn=collator,
    )

    return dataloaders
