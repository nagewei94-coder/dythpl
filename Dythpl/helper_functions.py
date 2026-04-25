import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
from copy import deepcopy
import random
import xml.etree.ElementTree as ET
import time
from copy import deepcopy
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import datasets as datasets
import torch
from PIL import ImageDraw
from pycocotools.coco import COCO
import json
import torch.utils.data as data
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import pickle
from sklearn.metrics import average_precision_score,roc_auc_score
import scipy.io as sio  # 【新增】用于读取 .mat 文件

 
def get_auc(target,preds):
    total_auc=0.
    auc=0
    for i in range(target.shape[1]):
        try:
            auc = roc_auc_score(target[:, i], preds[:, i])
        except ValueError:
            pass
        total_auc += auc

    multi_auc = total_auc / target.shape[1]

    return multi_auc

def one_error(target,preds):
    # 【新增】检查输入类型，如果是 NumPy 就转成 Tensor
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)

    ins_num=preds.shape[0]
    class_num=preds.shape[1]
    err=0
    for i in range(ins_num):
        idx=torch.argmax(preds[i])
        if target[i][idx]==0:
            err+=1
    return err/ins_num


def micro_f1(mcm):
    class_num=mcm.shape[0]
    tp=0
    fp=0
    tn=0
    fn=0
    for i in range(class_num):
        tn=tn+mcm[i][0][0]
        fp=fp+mcm[i][0][1]
        fn=fn+mcm[i][1][0]
        tp=tp+mcm[i][1][1]
    OP=tp/(fp+tp)
    OR=tp/(tp+fn)
    OF1=(2*OP*OR)/(OP+OR)
    print("OP,OR,OF1",OP,OR,OF1)
    return OF1,OP,OR
def macro_f1(mcm):
    class_num=mcm.shape[0]

    CP=0
    CR=0
    for i in range(class_num):
        if mcm[i][0][1]+mcm[i][1][1] == 0:
            CP=CP
        else :
            CP=CP+(mcm[i][1][1]/(mcm[i][0][1]+mcm[i][1][1]))
        if mcm[i][1][0]+mcm[i][1][1] == 0:
            CR=CR
        else:
            CR=CR+(mcm[i][1][1]/(mcm[i][1][0]+mcm[i][1][1]))

   
    CP=CP/class_num
    CR=CR/class_num
    CF1=(2*CR*CP)/(CR+CP)
    print("CP,CR,CF1",CP,CR,CF1)
    return CF1,CP,CR


def compute_mAP(y_true, y_pred):
    AP = []
    # y_true = y_true.max(axis=1).astype(np.float64)
    for i in range(y_true.shape[1]):
        AP.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(AP)*100
def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds,write=False):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))


    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)

    return 100 * ap.mean()

class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01


    def __len__(self):
        return len(self.ids)

    def _load_label(self, img_id):
        """实时解析 XML 获取 One-hot 标签"""
        xml_path = os.path.join(self.anno_dir, img_id + '.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        label_vec = torch.zeros(20)
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in self.class_to_ind:
                idx = self.class_to_ind[name]
                label_vec[idx] = 1.0
                
        return label_vec

    def __getitem__(self, index):
        img_id = self.ids[index]
        
        # 1. 加载图片
        img_path = os.path.join(self.img_dir, img_id + '.jpg')
        image = Image.open(img_path).convert('RGB')
        
        # 2. 获取基础分类标签 (20维)
        target_label = self._load_label(img_id) # 返回 Tensor(20)

        # 3. 如果是训练集，拼接主题标签
        if self.is_train:
            # 获取主题，如果没找到则给默认值 [0, 0]
            # 注意: 这里的主题索引需要是 float 形式拼接到 target 里，
            # 后续在 engine.py 或 TPLoss 里会被拆分并转回 long/int
            topics = self.topic_dict.get(img_id, [0.0, 0.0]) 
            topic_tensor = torch.tensor(topics, dtype=torch.float32)
            
            # 拼接: [20个分类标签, 2个主题标签] -> 总长 22
            target = torch.cat((target_label, topic_tensor), dim=0)
        else:
            # 测试集不需要主题，保持原样即可，或者为了统一格式也可以补0
            # 这里保持原样，因为 validate_tathpl_voc 会切片取前20个
            target = target_label

        if self.transform is not None:
            image = self.transform(image)
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make TATHPL copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def get_class_ids_split(json_path, classes_dict):
    with open(json_path) as fp:
        split_dict = json.load(fp)
    if 'train class' in split_dict:
        only_test_classes = False
    else:
        only_test_classes = True

    train_cls_ids = set()
    val_cls_ids = set()
    test_cls_ids = set()

    # classes_dict = self.learn.dbunch.dataset.classes
    for idx, (i, current_class) in enumerate(classes_dict.items()):
        if only_test_classes:  # base the division only on test classes
            if current_class in split_dict['test class']:
                test_cls_ids.add(idx)
            else:
                val_cls_ids.add(idx)
                train_cls_ids.add(idx)
        else:  # per set classes are provided
            if current_class in split_dict['train class']:
                train_cls_ids.add(idx)
            # if current_class in split_dict['validation class']:
            #     val_cls_ids.add(i)
            if current_class in split_dict['test class']:
                test_cls_ids.add(idx)

    train_cls_ids = np.fromiter(train_cls_ids, np.int32)
    val_cls_ids = np.fromiter(val_cls_ids, np.int32)
    test_cls_ids = np.fromiter(test_cls_ids, np.int32)
    return train_cls_ids, val_cls_ids, test_cls_ids


def update_wordvecs(model, train_wordvecs=None, test_wordvecs=None):
    if hasattr(model, 'fc'):
        if train_wordvecs is not None:
            model.fc.decoder.query_embed = train_wordvecs.transpose(0, 1).cuda()
        else:
            model.fc.decoder.query_embed = test_wordvecs.transpose(0, 1).cuda()
    elif hasattr(model, 'head'):
        if train_wordvecs is not None:
            model.head.decoder.query_embed = train_wordvecs.transpose(0, 1).cuda()
        else:
            model.head.decoder.query_embed = test_wordvecs.transpose(0, 1).cuda()
    else:
        print("model is not suited for ml-decoder")
        exit(-1)


def default_loader(path):
    img = Image.open(path)
    return img.convert('RGB')
    # return Image.open(path).convert('RGB')

class DatasetFromList(data.Dataset):
    """From List dataset."""

    def __init__(self, root, impaths, labels, idx_to_class,
                 transform=None, target_transform=None, class_ids=None,
                 loader=default_loader):
        """
        Args:

            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on TATHPL sample.
        """
        self.root = root
        self.classes = idx_to_class
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.samples = tuple(zip(impaths, labels))
        self.class_ids = class_ids
        self.get_relevant_samples()

    def __getitem__(self, index):
        impath, target = self.samples[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform([target])
        target = self.get_targets_multi_label(np.array(target))
        if self.class_ids is not None:
            target = target[self.class_ids]
        return img, target

    def __len__(self):
        return len(self.samples)

    def get_targets_multi_label(self, target):
        # Full (non-partial) labels
        labels = np.zeros(len(self.classes))
        labels[target] = 1
        target = labels.astype('float32')
        return target

    def get_relevant_samples(self):
        new_samples = [s for s in
                       self.samples if any(x in self.class_ids for x in s[1])]
        # new_indices = [i for i, s in enumerate(self.samples) if any(x in self.class_ids for x
        #                                                             in s[1])]
        # omitted_samples = [s for s in
        #                    self.samples if not any(x in self.class_ids for x in s[1])]

        self.samples = new_samples



def parse_csv_data(dataset_local_path, metadata_local_path):
    try:
        df = pd.read_csv(os.path.join(metadata_local_path, "data.csv"))
    except FileNotFoundError:
        # No data.csv in metadata_path. Try dataset_local_path:
        metadata_local_path = dataset_local_path
        df = pd.read_csv(os.path.join(metadata_local_path, "data.csv"))
    images_path_list = df.values[:, 0]
    # images_path_list = [os.path.join(dataset_local_path, images_path_list[i]) for i in range(len(images_path_list))]
    labels = df.values[:, 1]
    image_labels_list = [labels.replace('[', "").replace(']', "").split(', ') for labels in
                             labels]

    if df.values.shape[1] == 3:  # split provided
        valid_idx = [i for i in range(len(df.values[:, 2])) if df.values[i, 2] == 'val']
        train_idx = [i for i in range(len(df.values[:, 2])) if df.values[i, 2] == 'train']
    else:
        valid_idx = None
        train_idx = None

    # logger.info("em: end parsr_csv_data: num_labeles: %d " % len(image_labels_list))
    # logger.info("em: end parsr_csv_data: : %d " % len(image_labels_list))

    return images_path_list, image_labels_list, train_idx, valid_idx


def multilabel2numeric(multilabels):
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(multilabels)
    classes = multilabel_binarizer.classes_
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    multilabels_numeric = []
    for multilabel in multilabels:
        labels = [class_to_idx[label] for label in multilabel]
        multilabels_numeric.append(labels)
    return multilabels_numeric, class_to_idx, idx_to_class


def get_datasets_from_csv(dataset_local_path, metadata_local_path, train_transform,
                          val_transform, json_path):

    images_path_list, image_labels_list, train_idx, valid_idx = parse_csv_data(dataset_local_path, metadata_local_path)
    labels, class_to_idx, idx_to_class = multilabel2numeric(image_labels_list)

    images_path_list_train = [images_path_list[idx] for idx in train_idx]
    image_labels_list_train = [labels[idx] for idx in train_idx]

    images_path_list_val = [images_path_list[idx] for idx in valid_idx]
    image_labels_list_val = [labels[idx] for idx in valid_idx]

    train_cls_ids, _, test_cls_ids = get_class_ids_split(json_path, idx_to_class)

    train_dl = DatasetFromList(dataset_local_path, images_path_list_train, image_labels_list_train,
                               idx_to_class,
                               transform=train_transform, class_ids=train_cls_ids)

    val_dl = DatasetFromList(dataset_local_path, images_path_list_val, image_labels_list_val, idx_to_class,
                             transform=val_transform, class_ids=test_cls_ids)

    return train_dl, val_dl, train_cls_ids, test_cls_ids

# --- 追加在 helper_functions.py 末尾 ---

class VOC2007_Simple(datasets.VOCDetection):
    """
    一个简化的 VOC 数据加载器，不需要依赖 LDA 主题文件。
    专门用于跑 Baseline。
    """
    def __init__(self, root, year='2007', image_set='trainval', transform=None, target_transform=None):
        # 自动下载并解压，省去路径配置烦恼
        super().__init__(root, year=year, image_set=image_set, download=False, transform=transform, target_transform=target_transform)
        self.VOC_CLASSES = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
            'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

    def __getitem__(self, index):
        # 调用父类获取原始图片和 xml 字典
        img, target_dict = super().__getitem__(index)
        
        # 将 xml 字典转换为 One-Hot 标签向量
        label_vec = self._encode_target(target_dict)
        
        # 构造一个假的 topic 数据 (全0)，为了骗过模型的前向传播检查
        # 假设我们只需要跑通 ViT，不关心 Topic，这里给个空占位符即可
        dummy_topic = torch.zeros(1) 
        
        # 返回格式必须是 image, [label, topic]，为了兼容 engine.py 的逻辑
        return img, [label_vec, dummy_topic]

    def _encode_target(self, target_dict):
        objects = target_dict['annotation']['object']
        if not isinstance(objects, list):
            objects = [objects]
        
        label_vec = torch.zeros(len(self.VOC_CLASSES))
        for obj in objects:
            class_name = obj['name']
            if class_name in self.VOC_CLASSES:
                idx = self.VOC_CLASSES.index(class_name)
                label_vec[idx] = 1.0
        return label_vec
    

    # --- 追加在 helper_functions.py 末尾 ---

class voc2007_DyTHPL(Dataset):
    """
    一个专为 DyT-HPL 设计的干净的数据加载器。
    它只加载图片和分类标签，不读取任何主题文件。
    """
    def __init__(self, data_path, transform=None, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.root = data_path
        self.img_dir = os.path.join(self.root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        self.anno_dir = os.path.join(self.root, 'VOCdevkit', 'VOC2007', 'Annotations')
        
        # 读取官方划分文件
        if is_train:
            txt_file = os.path.join(self.root, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'trainval.txt')
        else:
            txt_file = os.path.join(self.root, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'test.txt')
            
        with open(txt_file, 'r') as f:
            self.ids = [line.strip() for line in f.readlines()]

        self.transform = transform
        
        self.VOC_CLASSES = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
            'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        self.class_to_ind = dict(zip(self.VOC_CLASSES, range(len(self.VOC_CLASSES))))

    def __len__(self):
        return len(self.ids)

    def _load_label(self, img_id):
        xml_path = os.path.join(self.anno_dir, img_id + '.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        label_vec = torch.zeros(20)
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in self.class_to_ind:
                idx = self.class_to_ind[name]
                label_vec[idx] = 1.0
        return label_vec

    def __getitem__(self, index):
        img_id = self.ids[index]
        
        # 加载图片
        img_path = os.path.join(self.img_dir, img_id + '.jpg')
        image = Image.open(img_path).convert('RGB')
        
        # 只加载分类标签
        target_label = self._load_label(img_id)

        if self.transform is not None:
            image = self.transform(image)

        # 【关键】只返回图片和分类标签，不再拼接任何东西
        return image, target_label
    



class Corel5k_DyTHPL(Dataset):
    def __init__(self, data_path, transform=None, is_train=True):
        super().__init__()
        self.is_train = is_train
        
        # 1. 定位根目录
        self.root = os.path.join(data_path, 'Corel5k', 'Corel5k')
        if not os.path.exists(self.root):
            self.root = os.path.join(data_path, 'Corel5k')

        # 2. 读取类别名称
        words_file = os.path.join(self.root, 'corel5k_words.txt')
        if os.path.exists(words_file):
            print(f"Loading class names from: {words_file}")
            with open(words_file, 'r', encoding='utf-8') as f:
                self.VOC_CLASSES = [line.strip() for line in f.readlines()]
        else:
            self.VOC_CLASSES = [f"Class_{i}" for i in range(260)]
            
        self.num_classes = len(self.VOC_CLASSES)

        # 3. 读取图片列表和标签矩阵
        if self.is_train:
            list_filename = 'corel5k_train_list.txt'
            annot_filename = 'corel5k_train_annot.mat'
        else:
            list_filename = 'corel5k_test_list.txt'
            annot_filename = 'corel5k_test_annot.mat'

        list_path = os.path.join(self.root, list_filename)
        annot_path = os.path.join(self.root, annot_filename)

        with open(list_path, 'r') as f:
            self.img_files = [line.strip() for line in f.readlines()]

        try:
            mat_data = sio.loadmat(annot_path)
            self.labels = None
            for key in mat_data.keys():
                if 'annot' in key:
                    self.labels = mat_data[key]
                    break
            
            if self.labels is None:
                raise ValueError(f"Cannot find annotation key in {annot_path}")
                
            self.labels = np.array(self.labels, dtype=np.float32)
            
        except Exception as e:
            print(f"Error loading .mat file: {e}")
            raise

        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # 1. 获取图片路径 (txt里可能是 "103000/103030.jpeg")
        img_name_raw = self.img_files[index]
        
        # 去掉可能存在的后缀，方便我们自己尝试 jpg/jpeg
        img_base = os.path.splitext(img_name_raw)[0]
        
        # 定义所有可能的路径组合
        # 1. 尝试 root/images/路径.jpeg
        # 2. 尝试 root/images/路径.jpg
        # 3. 尝试 root/路径.jpeg (没有images文件夹的情况)
        # 4. 尝试 root/路径.jpg
        
        candidates = [
            os.path.join(self.root, 'images', img_base + '.jpeg'),
            os.path.join(self.root, 'images', img_base + '.jpg'),
            os.path.join(self.root, img_base + '.jpeg'),
            os.path.join(self.root, img_base + '.jpg'),
            # 处理反斜杠问题 (Windows)
            os.path.join(self.root, 'images', img_base.replace('/', '\\') + '.jpeg'),
            os.path.join(self.root, 'images', img_base.replace('/', '\\') + '.jpg'),
        ]

        img_path = None
        for p in candidates:
            if os.path.exists(p):
                img_path = p
                break
        
        if img_path is None:
            # 如果都找不到，打印详细报错信息，方便你去文件夹里看
            raise FileNotFoundError(
                f"Image not found! ID: {index}, Raw: {img_name_raw}\n"
                f"Tried paths:\n" + "\n".join(candidates)
            )

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image: {img_path}")
            raise e

        # 2. 获取标签
        target = torch.tensor(self.labels[index], dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image)
            
        return image, target

from copy import deepcopy

from copy import deepcopy
import torch

class PartialModelEma:
    """
    【修复版】：标准全局 EMA (Global EMA)
    (保留类名 PartialModelEma 以兼容现有代码，但逻辑已更正为全局平滑)
    
    机制:
    对模型的所有参数（包括 Backbone, Head, 以及 Prompt Pool）进行统一的 EMA 平滑。
    这彻底消除了“主干网络处于过去时间态，而 Prompt 处于现在时间态”带来的系统性时空错位问题。
    """
    def __init__(self, model, decay=0.9995, device=None):
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            # 获取当前训练模型的参数字典
            msd = model.state_dict()
            # 获取 EMA 模型的参数字典
            esd = self.module.state_dict()
            
            for k, v in msd.items():
                if self.device is not None:
                    v = v.to(device=self.device)
                
                # 【核心修复】：删除原本的 if 'pool' or 'prompt' 的硬拷贝特判逻辑
                # 对所有参数一视同仁，全部应用传入的 update_fn (即 EMA 平滑公式)
                # 保证 Query (Backbone生成) 和 Key (Prompt池) 永远在同一个平滑特征空间内对齐
                esd[k].copy_(update_fn(esd[k], v))

    def update(self, model):
        # 定义标准 EMA 更新公式: old * decay + new * (1 - decay)
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        # 强制同步所有参数 (通常在初始化或特殊重置时使用)
        self._update(model, update_fn=lambda e, m: m)


# --- 追加在 helper_functions.py 末尾 ---
from pycocotools.coco import COCO


class Coco_DyTHPL(data.Dataset):
    def __init__(self, root, annFile, transform=None):
        """
        Args:
            root: 图片文件夹路径 (例如 .../train2014)
            annFile: json 标注文件路径 (例如 .../instances_train2014.json)
        """
        self.root = root
        self.coco = COCO(annFile)
        
        # 获取所有包含图片的 ID
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        
        # --- 建立类别映射 ---
        # COCO 的 category_id 是 1~90，中间有断号（比如没有 id=12）
        # 我们需要把它映射到连续的 0~79 索引
        cats = self.coco.loadCats(self.coco.getCatIds())
        # 按 id 排序确保顺序一致
        cats.sort(key=lambda x: x['id'])
        
        # 映射表: COCO_ID -> 0~79 Index
        self.cat_id_to_index = {cat['id']: i for i, cat in enumerate(cats)}
        
        # 保存类别名，用于后续可视化
        self.VOC_CLASSES = [cat['name'] for cat in cats] 
        self.num_classes = len(self.VOC_CLASSES) # 应该是 80

    def __getitem__(self, index):
        img_id = self.ids[index]
        
        # 1. 加载图片
        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        path = os.path.join(self.root, file_name)
        
        # 容错处理：确保转为 RGB (COCO 有少量灰度图)
        image = Image.open(path).convert('RGB')
        
        # 2. 生成标签 (Standard Multi-hot: 80-dim)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        
        for ann in anns:
            cat_id = ann['category_id']
            # 将 COCO ID 转换为 0-79 的索引
            if cat_id in self.cat_id_to_index:
                idx = self.cat_id_to_index[cat_id]
                target[idx] = 1.0
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, target

    def __len__(self):
        return len(self.ids)