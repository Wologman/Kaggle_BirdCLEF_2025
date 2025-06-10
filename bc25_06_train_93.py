use_case = {
                'experiment' : '93',
                'project_root': '/home/olly/Desktop/Kaggle_BC25',
                'num_cores': 2, 
                'run_training' : True,
                'epochs' : 24,
                'pretrained_ckpt': '/home/olly/Desktop/Kaggle_BC25/Data/Experiments/Exp_93_pretrain/Results/epoch=14-step=40440.ckpt',
                'exclude': 'birds'
                }

#General Python
import gc
from pathlib import Path
from tqdm import tqdm
import ast
from ast import literal_eval
from functools import reduce
import yaml
import warnings
import math
warnings.filterwarnings("ignore", category=UserWarning, message='A new version')

#Math & Plotting
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import plotly.express as px

#Machine Learning 
import albumentations as A
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics as skm

#Torch and PyTorch specific
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,  EarlyStopping
from torch.utils.data import  DataLoader, Dataset, WeightedRandomSampler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchaudio.functional import compute_deltas

#Audio
import librosa
import torchaudio
import colorednoise as cn


class FilePaths:
    def __init__(self, options=None):
        self.PROJECT_DIR = Path(options['project_root'])
        self.DATA_DIR = self.PROJECT_DIR / 'Data' 
        self.TRAIN_METADATA = self.DATA_DIR / 'Train_Metadata'
        self.LABELS_PATH = str(self.TRAIN_METADATA / 'train.parquet')
        
        _experiment = options['experiment']
        self.temp_dir = str(self.DATA_DIR / f'Experiments/Exp_{_experiment}' / 'Temp')
        self.chkpt_dir = self.temp_dir  + '/checkpoints'
        self.out_dir = self.DATA_DIR / f'Experiments/Exp_{_experiment}' / 'Results'
        self.model_deploy = self.DATA_DIR / f'Experiments/Exp_{_experiment}/Exp_{_experiment}_Deploy'
        self.last_weights_path = str(Path(self.chkpt_dir) / 'last.ckpt')
        self.bird_names_map = str(self.TRAIN_METADATA  / 'naming.csv')
        self.bird_map_for_model = self.model_deploy / f'exp_{_experiment}_bird_map.csv'
        self.model_config = self.model_deploy / f'exp_{_experiment}_config.yaml'
        self.val_df_path = self.out_dir / 'val_labels.csv'
        self.SHORT_FILES_PATH = self.DATA_DIR / 'Train_Metadata/audio_labels_under_6_sec.parquet'
        _background_audio_dir = self.DATA_DIR / 'Original_Data/birdclef-2025/train_soundscapes'
        self.background_audios = list(_background_audio_dir.glob("*.ogg"))
        #self.BACKGROUND_MDL_CKPT = '/home/olly/Desktop/Kaggle_BC25/Data/Experiments/Exp_69/Results/epoch=15-step=16911.ckpt'
        self.background_pseudolabels = self.TRAIN_METADATA / 'background_labels.parquet'
        self.background_crops_dir = self.DATA_DIR / 'Cropped_Train_Soundscapes'
        self.pretrained_ckpt = options['pretrained_ckpt']


class TrainingParameters:
    def __init__(self, options=None):
        self.TRAIN = options['run_training']
        self.EPOCHS = options['epochs'] 
        self.YEAR = 25
        self.EXPERIMENT = options['experiment']
        self.NUM_WORKERS = options['num_cores']
        self.BATCH_SIZE = 32 #32 # 12,  16, 32, 64 for sizes 512, 348, 32-larger network, 256
        self.TEST_BATCH_SIZE = 16
        self.PATIENCE = 6
        self.RESET_EPOCH = 9
        self.KEEP_LAST= 10
        self.MIN_DELTA = 0
        self.SEED = 2024
        self.MODEL = 'tf_efficientnet_b0.ns_jft_in1k' #, #'eca_nfnet_l0' #'tf_efficientnet_b0.ns_jft_in1k' #'convnext_tiny.in12k_ft_in1k' #'convnext_tiny.fb_in22k', 'eca_nfnet_l0' #  # 'tf_efficientnetv2_s.in21k_ft_in1k'
        self.WEIGHTED_SAMPLING = False
        self.WEIGHT_DECAY = 1e-6
        self.WARMUP_EPOCHS = 2
        self.INITIAL_LR = 1e-4 
        self.LR = 1e-3
        self.MIN_LR = 1e-5
        self.LR_CYCLE_LENGTH = 8
        self.LR_DECAY = 0.5
        self.EPOCHS_TO_UNFREEZE_BACKBONE = 6
        self.START_BACKGROUNDS_EPOCH = 0  #None   #Not using backgrounds
        self.PSEUDOLABEL_WEIGHT = .8 #None   #Not using pseudolabels
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.GPU = 'gpu' if torch.cuda.is_available() else 'cpu' #for Pytorch Lightning
        self.PRECISION = '16-mixed' if self.GPU == 'gpu' else 32
        self.LOSS_FUNCTION_NAME =  'BCEFocal2WayLoss' #'BCEWithLogitsLoss', 'BCEFocalLoss',
        self.USE_MIXUP = True
        self.MIXUP_ALPHA = .25 #Tried .4 and performance droped slightly from .64 to .63 (so inconclusive)
        self.LOW_ALPHA = 0.8 #For Focal Loss, for the most common classes, we downweigt the 'easy' prediction of 'false'
        self.MID_ALPHA = 0.8 
        self.HIGH_ALPHA = 0.8 #For the rare classes, we want the decision to have more impact on the loss compared to the common ones.
        self.FOCAL_GAMMA = 2
        self.FIRST_AUGMENTATION_UPDATE = 10
        self.SECOND_AUGMENTATION_UPDATE = 16                  


class BirdData:
    BIRDS = ['']
    NON_BIRDS = ['']
    N_FOLDS = 10
    USE_SECONDARY = True
    RARE_THRESHOLD = 20 #10 Classes with less samples than this will not be allowed in validation dataset, and will be up-sampled to this value
    SPATIAL_LIMITS = None #Filter the dataset by lat and long. For example: {'WEST':0, 'EAST':10, 'NORTH': -20, 'SOUTH':-30}
    MAX_PER_CLASS = 30000   #Cap the maximum number of samples allowed in any particular class to prevent extreme imbalance
    MAX_PER_CLASS_VAL = 300  #Cap the max for the val classes so that the val score isn't too dominated by the common classes
    EXCLUDED_CLASSES = []
    LOW_ALPHA_CLASSES = [],
    HIGH_ALPHA_CLASSES = [], 
    #LABEL_SMOOTHING = 0.1
    SECONDARY_WEIGHTS_TRAIN = 0.3
    SECONDARY_WEIGHTS_VAL = 0.3


class DefaultAudio:
    IMAGE_SHAPE = (2,0.5)#(1,2)  #5 second chunks position in final image: height x width
    DURATION = 12  # Duration the loaded sound file will be randomly cropped or padded to for training.
    SR = 32000
    IMAGE_WIDTH = 384 #384 #512 # 256 #The spectrogram will get cropped/padded to this square regardless of any audio considerations
    CHUNK_WIDTH = 768 #768 #IMAGE_WIDTH if IMAGE_SHAPE[1] == 1 else IMAGE_WIDTH // 2  #Number of frames wide for each sub-image
    N_MELS = 192 #192 #IMAGE_WIDTH // 2 if IMAGE_SHAPE[0] == 2 else IMAGE_WIDTH #Height of the chunk spectrograms
    N_FFT = 2048 #3072 #2048 *2 #3072 or 2048 #N_fft/2 + 1 bins will get made prior to downsampling to the value of N_MELS
    DOUBLE_AUDIO = True
    BUFFER_AUDIO = 1
    FMIN = 20
    FMAX = 14000 
    HOP_LENGTH = 500 #500 #750 #1500 #1243  #1500  #
    #826 #620 #310, 620, 826, 1243, for chunks widths of 128, 192, 256, 516 respectively for 10 seconds
    PCEN = False
    USE_DELTAS = True

class ResettingEarlyStopping(EarlyStopping):
    def __init__(self, reset_epoch=12, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_epoch = reset_epoch
        self._score_has_been_reset = False

    def on_validation_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch

        if current_epoch == self.reset_epoch and not self._score_has_been_reset:
            print(f"\n[EarlyStopping] Resetting best score and wait counter at epoch {self.reset_epoch}\n")
            self.wait_count = 0
            self.stopped_epoch = 0

            # reset best_score properly to inf/-inf tensor depending on mode
            if self.mode == "min":
                self.best_score = torch.tensor(float("inf"), device=pl_module.device)
            else:
                self.best_score = torch.tensor(float("-inf"), device=pl_module.device)

            self._score_has_been_reset = True

        super().on_validation_end(trainer, pl_module)



class ResettingModelCheckpoint(ModelCheckpoint):
    def __init__(self, reset_epoch=12, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_epoch = reset_epoch
        self._score_has_been_reset = False

    def on_validation_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch

        # Reset the best score at the specified epoch
        if current_epoch == self.reset_epoch and not self._score_has_been_reset:
            print(f"\n[Checkpoint Callback] Resetting best score at epoch {self.reset_epoch}\n")
            self.best_model_score = None
            self.kth_best_model_score = None
            self.best_k_models = {}  # Optional: clear tracked models too
            self._score_has_been_reset = True

        super().on_validation_end(trainer, pl_module)


def save_model_config(paths, audio_cfg, train_cfg):
    '''{'basename':'tf_efficientnet_b0.ns_jft_in1k', 
                         'ckpt_path': f"{options['project_root']}/Data/Experiments/Exp_{options['experiment']}/Results/last.ckpt",
                         'image_shape': (1,2), #The layout of 5-sec spectrograms stacked into the final image (height x width)
                         'image_time': 10,
                         'n_mels': 256,
                         'n_fft': 2048,
                         'use_deltas' : True,
                         'hop_length': 1243,
                         '5_sec_width': 128,
                         'aggregation': 'mean',
                         }'''
    
    model_config = {'basename': train_cfg.MODEL, 
                    'image_shape': audio_cfg.IMAGE_SHAPE, #The layout of 5-sec spectrograms stacked into the final image (height x width)
                    'image_time': audio_cfg.DURATION,
                    'n_mels': audio_cfg.N_MELS,
                    'n_fft': audio_cfg.N_FFT,
                    'double_audio': audio_cfg.DOUBLE_AUDIO,
                    'buffer_audio': audio_cfg.BUFFER_AUDIO,
                    'use_deltas' : audio_cfg.USE_DELTAS,
                    'hop_length': audio_cfg.HOP_LENGTH,
                    '5_sec_width': audio_cfg.CHUNK_WIDTH,
                    'aggregation': 'mean',  #Aggregation within the attention block, currently not a variable in training
                    }
    
    with paths.model_config.open("w") as f:
        yaml.dump(model_config, f, default_flow_style=False)

class Stop: #bold red
    S = '\033[1m' + '\033[91m'
    E = '\033[0m'
    
class Go: #bold green
    S = '\033[1m' + '\033[32m'
    E = '\033[0m'
    
class Blue: #for general info
    S = '\033[1m' + '\033[94m'
    E = '\033[0m'


def get_pseudos(path):
    '''
    To train where the dataset has been pre-classified by other models
    Returns a dict of list of lists.  Each sub-list is the prediction values, 
    the position of the sub-list corresponts to the time-position
    in the sample, with each list representing a chunk of 5 seconds
    '''
    pseudo_df = pd.read_csv(path)

    if 'latitude' in pseudo_df.columns and 'longitude' in pseudo_df.columns:
        pseudo_df.drop(columns=['latitude', 'longitude'], inplace=True)

    #drop any rows where all the values are 0
    cols_after_4th = pseudo_df.columns[4:]
    mask = (pseudo_df[cols_after_4th] == 0).all(axis=1)
    pseudo_df = pseudo_df[~mask]

    print(pseudo_df.iloc[:,:6].head())

    grouped = pseudo_df.groupby('filename')
    birdlist = pseudo_df.iloc[:,4:].columns.tolist()
    print(f'There are {len(birdlist)} birds in the value columns')
    pseudo_dict = {}

    for filename, group in grouped:
        group_sorted = group.sort_values(by='time')
        values = group_sorted[birdlist].values.tolist()
        pseudo_dict[filename] = values

    return pseudo_dict


def load_sf(wav_path):
    y, _ = torchaudio.load(wav_path)
    y = y.squeeze().numpy() 
    if not np.isfinite(y).all():
        y[np.isnan(y)] = np.mean(y)
    return y
    

def balance_primary_label(df, label_column='primary_label', max_count=200):
    value_counts = df[label_column].value_counts()
    balanced_df = pd.DataFrame(columns=df.columns)
    for value, count in value_counts.items():
        value_df = df[df[label_column] == value]
        if count > max_count:
            value_df = value_df.sample(n=max_count, random_state=1)
        balanced_df = pd.concat([balanced_df, value_df], axis=0)
    balanced_df = balanced_df.sample(frac=1, random_state=1).reset_index(drop=True)
    
    return balanced_df

    
def compute_pcen(y, audio_cfg):
    if not np.isfinite(y).all():
        y[np.isnan(y)] = np.zeros_like(y)
        y[np.isinf(y)] = np.max(y)
    
    melspec = librosa.feature.melspectrogram(y=y, 
                                             sr=audio_cfg.SR, 
                                             n_mels=audio_cfg.N_MELS, 
                                             n_fft= audio_cfg.N_FFT, 
                                             fmin=audio_cfg.FMIN, 
                                             fmax=audio_cfg.FMAX
                                            )
    pcen = librosa.pcen(melspec, 
                        sr=audio_cfg.SR, 
                        gain=0.98, 
                        bias=2, 
                        power=0.5, 
                        time_constant=0.4, 
                        eps=0.000001
                       )
    return pcen.astype(np.float32)


def compute_melspec(y, audio_cfg):
    if not np.isfinite(y).all():
        y[np.isnan(y)] = np.zeros_like(y)
        y[np.isinf(y)] = np.max(y)
    
    melspec = librosa.feature.melspectrogram(y=y, 
                                             sr=audio_cfg.SR, 
                                             n_mels=audio_cfg.N_MELS, 
                                             n_fft=audio_cfg.N_FFT, 
                                             hop_length = audio_cfg.HOP_LENGTH, 
                                             fmin=audio_cfg.FMIN, 
                                             fmax=audio_cfg.FMAX
                                            ) 
    return librosa.power_to_db(melspec)


def mono_to_color(X, eps=1e-6, use_deltas=False):
    _min, _max = X.min(), X.max()
    if (_max - _min) > eps:
        X = (X - _min) / (_max - _min) #scales to a range of [0,1]
        X = X.astype(np.float32)
    else:
        X = np.zeros_like(X, dtype=np.float32)

    if use_deltas:
        T = torch.tensor(X, dtype=torch.float32)
        delta = compute_deltas(T)
        delta_2 = compute_deltas(delta)
        delta, delta_2 = delta.numpy(), delta_2.numpy()
        X = np.stack([X, delta, delta_2], axis=-1)
    else:
        X = np.stack([X, X, X], axis=-1) #puts the chanels last, like a normal image
    
    return X


def crop_or_pad(y, length,  train='train'):
    initial_length = len(y)
    max_vol = np.abs(y).max()
    if max_vol == 0:
        print('Warning, there was training sample of all zeros before padding')
    if initial_length == 0:
        print('Warning, there was a sample of initial length zero before padding')
    if 3 * initial_length < length:
        random_values = np.random.random(initial_length)
        y = np.concatenate([y,random_values,y])
    elif 2 * initial_length < length:
        random_values = np.random.random(initial_length//2)
        y = np.concatenate([y,random_values,y])
    if len(y) < length:
        y = np.concatenate([y, y]) 
    
    def Normalize(array):
        max_vol = np.abs(array).max()
        if max_vol == 0:
            length = len(array)
            array = np.random.random(length)
            print('Warning, there was a final training sample of all zeros, replacing with random noise')
            return array  # or return array filled with zeros, if appropriate
        return array * 1 / max_vol

    if len(y) < length:
        difference = length - len(y)
        fill=np.zeros(difference)
        y = np.concatenate([y, fill])
    else:
        if train != 'train':
            start = 0
        else:
            start = 0
            start = np.random.randint(len(y) - length)
        y = y[start: start + length]
    y = Normalize(y)
    return y


def random_crop(arr, length):
    '''For cropping backgrounds from a larger clip to a chosen length'''
    if len(arr) > length:
        start = np.random.randint(len(arr) - length)
        arr = arr[start: start + length]
    return arr


def padded_cmap(solution, submission, padding_factor=5):
    solution = solution.fillna(0).replace([np.inf, -np.inf], 0)
    submission = submission.fillna(0).replace([np.inf, -np.inf], 0)
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    score = skm.average_precision_score(
        padded_solution.values,
        padded_submission.values,
        average='macro')    
    return score


def padded_cmap_by_class(solution, submission, padding_factor=5):
    solution = solution.fillna(0).replace([np.inf, -np.inf], 0)
    submission = submission.fillna(0).replace([np.inf, -np.inf], 0)
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    
    column_headers = list(solution.columns)
    scores = {}
    
    for column in column_headers:
        score = skm.average_precision_score(
            padded_solution[[column]].values,
            padded_submission[[column]].values,
            average='macro')    
        scores[column] = score
    return scores


def map_score(solution, submission):
    solution = solution.fillna(0).replace([pd.np.inf, -pd.np.inf], 0)
    submission = submission.fillna(0).replace([pd.np.inf, -pd.np.inf], 0)
    score = skm.average_precision_score(
        solution.values,
        submission.values,
        average='micro')  
    return score


def plot_by_class(df_target, df_pred):
    cmap5_by_class = padded_cmap_by_class(df_target, df_pred, padding_factor=5)
    col_sums = [(col, df_target[col].sum()) for col in df_target.columns]
    names_by_frequency = sorted(col_sums, key=lambda x: x[1], reverse=True)
    names = [name for name, _ in names_by_frequency]
    counts = [count for _, count in names_by_frequency]
    scores = [cmap5_by_class[name] for name in names]
    df = pd.DataFrame({'names': names, 'counts': counts, 'scores': scores})
    df["scores"] = pd.to_numeric(df["scores"])
    df["counts"] = pd.to_numeric(df["counts"])
    fig = px.bar(df, x='scores', y='names', color='counts', orientation='h', hover_data=['counts', 'scores'], range_x=[0, 1])
    fig.update_layout(height=1200)
    fig.show()
    return names, scores, counts


def save_naming_scheme(train_df, val_df, class_names, naming_csv_path):
    '''Saves out the naming scheme, with only the relevent 
    names, and some counts for future analysis.'''

    train_counts = train_df['primary_label'].value_counts().reset_index()
    val_counts = val_df['primary_label'].value_counts().reset_index()
    train_counts.columns = ['eBird', 'TrainSamples']
    val_counts.columns = ['eBird', 'ValSamples']
    use_cols = ['eBird', 'CommonName', 'ScientificName', 'ExtraName']
    name_map_df = pd.read_csv(paths.bird_names_map)

    mapped_birds = set(list(name_map_df['eBird'].unique()))
    train_birds = set(list(train_df['primary_label'].unique()))

    missing_from_mapping = train_birds - mapped_birds

    

    name_map_df = name_map_df[name_map_df['eBird'].isin(class_names)]
    name_map_df = name_map_df.drop_duplicates(subset='eBird', keep='first')
    name_map_df = name_map_df.merge(train_counts, on='eBird', how='left')
    name_map_df = name_map_df.merge(val_counts, on='eBird', how='left')
    name_map_df['ValSamples'] = name_map_df['ValSamples'].fillna(0)
    name_map_df = name_map_df.sort_values(by='eBird')
    if (__name__ == '__main__'):
        if len(missing_from_mapping) != 0:
            print(f'The following eBirds are missing from the mapping CSV: {missing_from_mapping}')
        print(f"In total there are {name_map_df['eBird'].nunique()} unique primary labels saved to the naming csv.")
        print(f"In total there are {train_df['primary_label'].nunique()} unique primary labels in the training csv.")
        print(f'There are {len(list(set(class_names)))} unique classes in the classes attribure of cfg')
    name_map_df.to_csv(naming_csv_path, index=False)
    return


############################################# Data Augmentation # ######################################
########################################################################################################

class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray, sr):
        for trns in self.transforms:
            y = trns(y, sr)
        return y
    

class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray, sr):
        if self.always_apply:
            return self.apply(y, sr=sr)
        else:
            if np.random.rand() < self.p:
                return self.apply(y, sr=sr)
            else:
                return y

    def apply(self, y: np.ndarray, **params):
        raise NotImplementedError
        
        
class OneOf(Compose):
    def __init__(self, transforms, p=0.5):
        super().__init__(transforms)
        self.p = p
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, y: np.ndarray, sr):
        data = y
        if self.transforms_ps and (np.random.random() < self.p):
            random_state = np.random.RandomState(np.random.randint(0, 2 ** 16 - 1))
            t = random_state.choice(self.transforms, p=self.transforms_ps)
            data = t(y, sr)
        return data
    
    
class Normalize(AudioTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray, **params):
        max_vol = np.abs(y).max()
        y_vol = y * 1 / max_vol
        #y_vol = y_vol - y_vol.mean()  Try adding this some time
        return np.asfortranarray(y_vol)
    
    
class RandomNoiseInjection(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_level=1):
        super().__init__(always_apply, p)

        self.noise_level = (0.0, max_noise_level)

    def apply(self, y: np.ndarray, **params):
        noise_level = np.random.uniform(*self.noise_level)
        noise = np.random.randn(len(y))
        augmented = (y + noise * noise_level).astype(y.dtype)
        return augmented
    
    
class GaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented
    
#https://github.com/felixpatzelt/colorednoise
class PinkNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)
        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented
    
    
class BrownNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)
        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        brown_noise = cn.powerlaw_psd_gaussian(2, len(y))
        a_brown = np.sqrt(brown_noise ** 2).max()
        augmented = (y + brown_noise * 1 / a_brown * a_noise).astype(y.dtype)
        return augmented
    

#https://www.kaggle.com/code/hidehisaarai1213/rfcx-audio-data-augmentation-japanese-english
#https://medium.com/@makcedward/data-augmentation-for-audio-76912b01fdf6
class AddBackround(AudioTransform):
    def __init__(self, 
                 duration,
                 sr,
                 background_noise_paths,
                 always_apply=True, 
                 p=0.6, 
                 min_snr=1, 
                 max_snr=3,
                 ):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr
        self.back_pths = background_noise_paths
        self.background = load_sf(random.choice(self.back_pths))
        self.d_len = duration * sr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        if random.random() < 0.2:
            self.background = load_sf(random.choice(self.back_pths))
        
        cropped_background = random_crop(self.background, self.d_len)

        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))  
        l_signal = len(y)

        a_background = np.sqrt(cropped_background ** 2).max()
        l_background = len(cropped_background)

        if l_signal > l_background:
            ratio = l_signal//l_background
            cropped_background = np.tile(cropped_background, ratio+1 )
            cropped_background = cropped_background[0:l_signal]

        if l_signal < l_background:    
            cropped_background = cropped_background[0:l_signal]

        augmented = (y + cropped_background * 1 / a_background * a_noise).astype(y.dtype)
        return augmented  
    
    
def spec_augment(spec: np.ndarray, 
                 num_mask=3, 
                 freq_masking_max_percentage=0.1,
                 time_masking_max_percentage=0.1, 
                 p=0.5):
    if random.uniform(0, 1) > p:
        return spec

    # frequency masking
    num_freq_masks = random.randint(1, num_mask)
    for i in range(num_freq_masks):
        freq_percentage = random.uniform(0, freq_masking_max_percentage)
        freq_mask_size = int(freq_percentage * spec.shape[0])
        freq_mask_pos = random.randint(0, spec.shape[0] - freq_mask_size)
        spec[freq_mask_pos:freq_mask_pos+freq_mask_size, :] = 0

    # time masking
    num_time_masks = random.randint(1, num_mask)
    for i in range(num_time_masks):
        time_percentage = random.uniform(0, time_masking_max_percentage)
        time_mask_size = int(time_percentage * spec.shape[1])
        time_mask_pos = random.randint(0, spec.shape[1] - time_mask_size)
        spec[:, time_mask_pos:time_mask_pos+time_mask_size] = 0

    return spec

class MinMaxNormalize(A.ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, image, **params):
        min_val = image.min()
        max_val = image.max()
        denom = max_val - min_val + 1e-6  # avoid divide by zero
        return (image - min_val) / denom


class AbluTransforms():
    MEAN = (0.485, 0.456, 0.406) # RGB
    STD = (0.229, 0.224, 0.225) # RGB
    
    def __init__(self, audio):
        self.image_width = audio.IMAGE_WIDTH

        self.train = A.Compose([
                        A.CoarseDropout(num_holes_range=[1,4], p=0.4),
                        A.PadIfNeeded(min_height=self.image_width, min_width=self.image_width),
                        A.CenterCrop(width=self.image_width, height=self.image_width), 
                        A.Normalize(self.MEAN, self.STD, max_pixel_value=1.0, p=1), 
                        ])
        
        self.valid = A.Compose([
                        A.PadIfNeeded(min_height=self.image_width, min_width=self.image_width),
                        A.CenterCrop(width=self.image_width, height=self.image_width),  
                        A.Normalize(self.MEAN, self.STD, max_pixel_value=1.0,p=1),
                        ])


class PrepareImage():
    mean = .5
    std = .22
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.prep = A.Compose([
            A.PadIfNeeded(min_height=self.height, min_width=self.width),
            A.CenterCrop(width=self.width, height=self.height), 
            A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0, p=1),
        ])


def mixup_data(x, y, alpha, device):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def fold_image(arr, shape): 
    '''chop the image in half along the temporal dimension and stack to a square image
    Goal is to allow more pixels and segments in the temporal domain than frequency'''
    length = arr.shape[1]
    num_vertical = shape[0]
    cols = length//num_vertical
    remainder = length % num_vertical
    if num_vertical == 2:
        half0 = arr[:, :cols + remainder]   #added the .T v55
        half1 = arr[:, cols:]  #added the .T v53
        arr =  np.vstack((half0, half1))  #changed to h-stack v55
    elif num_vertical == 4:
        half0 = arr[:, :cols + remainder]
        half1 = arr[:, cols:]
        half2 = arr[:, cols:]
        half3 = arr[:, cols:]
        arr =  np.vstack((half0, half1, half2, half3))  #changed to h-stack v55
    return arr


############################################# Dataset Definition  ######################################
########################################################################################################

def sumix(waves: torch.Tensor, labels: torch.Tensor, max_percent: float = 1.0, min_percent: float = 0.3):
    batch_size = len(labels)
    perm = torch.randperm(batch_size, device=waves.device)

    # Random scaling factors for each waveform
    coeffs_1 = torch.rand(batch_size, device=waves.device).view(-1, 1) * (max_percent - min_percent) + min_percent
    coeffs_2 = torch.rand(batch_size, device=waves.device).view(-1, 1) * (max_percent - min_percent) + min_percent

    # Mix waveforms
    waves = coeffs_1 * waves + coeffs_2 * waves[perm]
    eps = 1e-8
    max_vals = waves.abs().amax(dim=1, keepdim=True)
    waves = waves / (max_vals + eps)

    # Take per-class max for each label
    labels = torch.max(labels, labels[perm])

    return waves, labels

def backmix(waves: torch.Tensor, backgrounds: torch.Tensor, max_percent: float = 1, min_percent: float = 0.3):
    batch_size = waves.shape[0]
    backgrounds = backgrounds.to(waves.device)

    # Random scaling factors for each waveform
    #coeffs_1 = torch.rand(batch_size, device=waves.device).view(-1, 1) * (max_percent - min_percent) + min_percent
    coeffs_2 = torch.rand(batch_size, device=waves.device).view(-1, 1) * (max_percent - min_percent) + min_percent

    # Mix waveforms
    waves = waves + coeffs_2 * backgrounds
    eps = 1e-8
    max_vals = waves.abs().amax(dim=1, keepdim=True)
    waves = waves / (max_vals + eps)

    return waves



#concatmix
#https://github.com/dazzle-me/birdclef-2022-3rd-place-solution/blob/main/experiments/new-metrics-exp-17/fold_4/saved_src/models/layers/concat_mix.py


class WaveformDataset(Dataset):
    def __init__(self, 
                 df, #This is the default dataframe with only human-labelled data
                 audio,
                 epoch=0,
                 train=True, 
                 soundscape_path=None,
                 augmentation_updates = [6,11],
                 short_df = None
                ): 
        self.epoch=epoch
        self.sr = audio.SR
        self.train = train
        self.classes = list(df.columns)[3:]
        self.df = df
        self.duration = audio.DURATION
        self.d_len = self.duration * self.sr
        self.soundscape_path=soundscape_path
        self.back_pths = None
        self.height = audio.N_MELS
        self.width = audio.CHUNK_WIDTH
        self.image_shape = audio.IMAGE_SHAPE
        self.num_chunks = 1 #self.image_shape[0] * self.image_shape[1]
        self.chunk_lenth = self.d_len // self.num_chunks
        self.prep_image = PrepareImage(height=self.height, width = self.width)
        self.pcen = audio.PCEN
        self.use_deltas = audio.USE_DELTAS
        self.audio_cfg = audio
        self.first_aug_reset = augmentation_updates[0]
        self.second_aug_reset = augmentation_updates[1]
        self.short_df = short_df
        #print(f'The shape of the dataframe is currently {self.df.shape}')

        if self.train:
            self.wave_transforms = Compose(
                [
                    OneOf(
                        [
                            RandomNoiseInjection(p=1, max_noise_level=0.04),
                            GaussianNoise(p=1, min_snr=1, max_snr=5),
                            PinkNoise(p=1, min_snr=1, max_snr=5),
                            BrownNoise(p=1, min_snr=1, max_snr=5),
                        ],
                        p=.25,
                    ),
                    #AddBackround(self.duration, self.sr, self.back_pths, p=.25, min_snr=1.5, max_snr=3),
                    Normalize(p=1),
                ]
            )
        else:
            self.wave_transforms = Compose([Normalize(p=1)])
        
        if self.soundscape_path is not None:
            self.ss_array = self.open_audio_clip(soundscape_path)
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    def update_df(self, df_idx=1):
        self.df = self.df_list[df_idx]
        
    def reset_wave_augmentation(self, epoch):
        if self.train and self.first_aug_reset <= epoch < self.second_aug_reset:    #self.first_aug_reset  was 5 & 10
            print(f'Using medium waveform augmentation on epoch {epoch}')
            self.wave_transforms = Compose(
                [
                    OneOf(
                        [
                            RandomNoiseInjection(p=1, max_noise_level=0.04),
                            GaussianNoise(p=1, min_snr=1, max_snr=5),
                            PinkNoise(p=1, min_snr=1, max_snr=5),
                            BrownNoise(p=1, min_snr=1, max_snr=5),
                        ],
                        p=.15,
                    ),
                    #AddBackround(self.duration, self.sr, self.back_pths, p=.15, min_snr=1.5, max_snr=3),  #Tried various SNR and p combinations.  Adding background noise just doesn't seem to do much.
                    Normalize(p=1),
                ]
            )
        elif self.train and epoch >= self.second_aug_reset:
            print(f'Using minimal waveform augmentation on epoch {epoch}')
            self.wave_transforms = Compose(
                [
                    OneOf(
                        [
                            RandomNoiseInjection(p=1, max_noise_level=0.04),
                            GaussianNoise(p=1, min_snr=1.5, max_snr=5),
                            PinkNoise(p=1, min_snr=1.5, max_snr=5),
                            BrownNoise(p=1, min_snr=1.5, max_snr=5),
                        ],
                        p=.1,
                    ),
                    Normalize(p=1),
                ]
            )

    def open_audio_clip(self, path, centres=None):
        '''Opens an audio clip and returns the whole thing.  If centres are marked then only the 
           portions around those centres are returned.'''    
        try:  
            y, sr = torchaudio.load(path)
            if y.ndim == 2 and y.shape[0] == 2:
                #print(f'converting {path} to mono')
                y = torch.mean(y, dim=0).unsqueeze(0)  # from stereo to mono
            y = y.squeeze().numpy() 
        except:
            y = np.random.randn(5*320000) 
            print(f'could not open {path}')
        
        if not np.isfinite(y).all():
            y[np.isnan(y)] = np.mean(y)
            y[np.isinf(y)] = np.mean(y)

        if self.soundscape_path is None:
            if len(y) > 0: 
                if centres is not None and len(centres) > 0:
                    chunk_length = self.d_len // self.num_chunks
                    centre = random.choice(centres)
                    start = max(int(centre * sr) - chunk_length//2, 0)
                    end = min(int(centre * sr) + chunk_length//2, len(y))
                    y = y[start:end]
                else:
                    y = y[:self.d_len*4]  #so we don't get memory or speed issues from long clips
        return y
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        '''For training open the path from the row filepath, 
        or for a soundscape we just need to crop out the 
        relevent chunk of the array opened in the init.'''
        row = self.df.iloc[idx]
        centres = row.iloc[2]

        if self.soundscape_path:
            targets = idx 
        else:
            target_vals = row[3:].values.astype(np.uint8)
            targets = torch.tensor(target_vals)

        #print(f'getting item from this row: \n {row}')

        if self.soundscape_path is not None:
            start = self.sr * row.start
            stop = self.sr * row.stop
            y = self.ss_array[start: stop]  #arrays of 0 after the first iteration       
        else:
            wav_path = row.filepath 
            y = self.open_audio_clip(wav_path, centres=centres)
            ##########################################################################################################
            if len(y) <= 32000 * 6 and self.short_df is not None:
                #print(self.short_df.head(3))
                row = self.short_df.sample(n=1).iloc[0]
                filepath = row['filepath']
                _targets = row.iloc[3:].values.astype(np.uint8)
                _targets = torch.tensor(_targets)
                _y = self.open_audio_clip(filepath, centres=None)

                targets = torch.max(targets, _targets)
                y = np.concatenate((y, _y))
            ########################################################################################################

        y = crop_or_pad(y, self.d_len, train=self.train)
        y = self.wave_transforms(y, sr=self.sr)
        
        #image = get_image(y, self.audio_cfg, self.epoch, self.prep_image, train=self.train)
        #image = self.image_transform(image=image)['image']
        #image = image.transpose(2,0,1).astype(np.float32) # swapping the image channels to the first axis

        return y, targets


############################################# Loss Functions ######################################
##################################################################################################

# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
# https://www.kaggle.com/code/thedrcat/focal-multilabel-loss-in-pytorch-explained
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, loss_alphas=None):
        """
        :param alpha: Default alpha value if no class-specific alphas are provided.
        :param gamma: Gamma value for focal loss.
        :param loss_alphas: List of alpha values, one for each class.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_alphas = loss_alphas

    def forward(self, logits, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(logits, targets)
        probas = torch.sigmoid(logits)

        if self.loss_alphas is not None:
            alpha = torch.tensor(self.loss_alphas, device=logits.device)[targets.long()]
        else:
            alpha = self.alpha
        loss = targets * alpha * (1. - probas) ** self.gamma * bce_loss + \
               (1. - targets) * probas ** self.gamma * bce_loss
        loss = loss.mean()

        return loss


class BCEFocal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], gamma=2, loss_alphas=None):
        super().__init__()

        self.focal = BCEFocalLoss(gamma=gamma, loss_alphas=loss_alphas)
        self.weights = weights

    def forward(self, input, target):
        input_ = input["logit"]
        target = target.float()
        loss = self.focal(input_, target)

        #my simplified version, using the segment logits directly instead of the interpolated function from original code
        #segmentwise_logit = input['segmentwise_logit'].mean(dim=1) #also tried mean, but it didn't work for some reason
        segmentwise_logit = input['segmentwise_logit'].mean(dim=1)
        aux_loss = self.focal(segmentwise_logit, target)   

        return self.weights[1] * loss + self.weights[1] * aux_loss


class LossFunctions():
    '''A wrapper class, that incudes various loss function types and takes a dictionary
    as an input with the various outputs from the model'''
    def __init__(self, loss_fn_nm, gamma=2, loss_alphas=None):
        loss_dict = {
                'BCEFocal2WayLoss': BCEFocal2WayLoss(gamma=gamma, loss_alphas=loss_alphas),
                'BCEFocalLoss': BCEFocalLoss(),
                'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(),
                'CrossEntropyLoss': nn.CrossEntropyLoss()
                }
        self.loss_fn_nm = loss_fn_nm
        self.loss_fn = loss_dict.get(loss_fn_nm, nn.CrossEntropyLoss())
        
    def loss(self, preds_dict, target):
        if self.loss_fn_nm == 'BCEFocal2WayLoss':     
            loss_val = self.loss_fn(preds_dict, target)  #'BCEFocal2WayLoss'
        else:   # ['BCEFocalLoss', 'BCELossWithLogits','CrossEntropyLoss']
            loss_val = self.loss_fn(preds_dict['logit'], target)
        return loss_val


def open_background_clip(path):
    '''Opens an audio clip and returns the whole thing.  If centres are marked then only the 
        portions around those centres are returned.'''    
    y, sr = torchaudio.load(path)
    if y.ndim == 2 and y.shape[0] == 2:
        #print(f'converting {path} to mono')
        y = torch.mean(y, dim=0)#.unsqueeze(0)  # from stereo to mono
    return y.squeeze()



import concurrent.futures

def merge_pairs(X, Y):
    # Drop last if odd
    if X.shape[0] % 2 != 0:
        X = X[:-1]
        Y = Y[:-1]

    # Split into even and odd indices
    X_even, X_odd = X[::2], X[1::2]
    Y_even, Y_odd = Y[::2], Y[1::2]

    # Concatenate audio along time axis (dim=2)
    X_merged = torch.cat([X_even, X_odd], dim=1)

    # Logical OR on labels
    Y_merged = torch.logical_or(Y_even > 0, Y_odd > 0).float()

    return X_merged, Y_merged


def get_background_crops(df, batch_size, duration=6):   #get_backgrounds_batch(
    samples_df = df.sample(n=batch_size)
    target_length = duration * 32000
    audio_tensors = []
    label_tensors = []
    #label_columns = samples_df.columns[1:]   #This was probably a mistake for several experiments!
    label_columns = samples_df.columns.difference(['filepath'])
    
    for _, row in samples_df.iterrows():
        y = open_background_clip(row['filepath'])
        if target_length is not None:
            length = y.shape[0]
            if length < target_length:
                pad_size = target_length - length
                y = torch.nn.functional.pad(y, (0, pad_size))
            else:
                y = y[:target_length]
        audio_tensors.append(y)
        label_values = pd.to_numeric(row[label_columns], errors='coerce').values
        label_tensors.append(torch.tensor(label_values, dtype=torch.float32))

    X = torch.stack(audio_tensors)  # shape: [batch, 1, time]
    Y = torch.stack(label_tensors)  # shape: [batch, num_classes]
    return X, Y


def get_backgrounds_batch(path_list, batch_size, duration):   #get_backgrounds_batch(
    chosen_paths = random.sample(path_list, 8)
    length = duration * 32000
    substacks = []

    # Divide batch_size approximately evenly across 8 clips
    split_sizes = [batch_size // 8] * 8
    remainder = batch_size % 8
    for i in range(remainder):
        split_sizes[i] += 1  # distribute leftover samples

    def process_clip(path, num_samples):
        y = open_background_clip(path)
        clip_length = y.shape[0]
        starts = [random.randint(0, clip_length - length) for _ in range(num_samples)]
        return torch.stack([y[start:start+length] for start in starts])

    # Parallelize the processing of clips using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_clip, chosen_paths, split_sizes))

    # Combine all the resulting stacks into a single batch
    batch = torch.cat(results, dim=0)
    return batch

def get_backgrounds(path_list, num_backgrounds, duration):
    '''single core version for smaller number of samples'''
    chosen_paths = random.sample(path_list, num_backgrounds)
    length = duration * 32000

    def process_clip(path, num_samples):
        y = open_background_clip(path)
        clip_length = y.shape[0]
        starts = [random.randint(0, clip_length - length) for _ in range(num_samples)]
        return torch.stack([y[start:start+length] for start in starts])

    wavs = []
    for wav_path in chosen_paths:
        wav = process_clip(wav_path, 1)
        wavs.append(wav)

    # Combine all the resulting stacks into a single batch
    batch = torch.cat(wavs, dim=0)
    return batch


############################################# Model Definition  ######################################
##################################################################################################

def get_one_image(y, audio_cfg, epoch, image_preparer, ablu_transforms, train=True):
    use_pcen = audio_cfg.PCEN
    width = audio_cfg.CHUNK_WIDTH
    shape = audio_cfg.IMAGE_SHAPE
    use_deltas = audio_cfg.USE_DELTAS
    num_chunks = int(shape[0] * shape[1])

    if use_pcen: 
        image = compute_pcen(y, audio_cfg)
    else:
        image = compute_melspec(y, audio_cfg)
    
    #note that the point of this is to normalise each chunk seperately, and re-assemble the image the way it already was
    normalised = []
    if num_chunks == 4:
        for i in range(4):
            sub_image = image[:, i * width : (i + 1) * width] #four quarters of the final image
            sub_image = image_preparer.prep(image=sub_image)['image']
            normalised.append(sub_image)
        image = np.concatenate(normalised, axis=1)
    elif num_chunks == 2:
        for i in range(2):
            sub_image = image[:, i * width  : (i + 1) * width]  #stacking the two halfs of the final image
            sub_image = image_preparer.prep(image=sub_image)['image']
            normalised.append(sub_image)
        image = np.concatenate(normalised, axis=1)
    else:
        image = image_preparer.prep(image=image)['image']
    #from here it would be safest not to do any shuffling in the time direction, as chunks were normalised seperately

    if train and epoch <=8:
        image = spec_augment(image, 
                            p=0.25, 
                            num_mask=3,
                            freq_masking_max_percentage=0.1,
                            time_masking_max_percentage=0.1)
    
    if shape[0]==2:
        image = fold_image(image, shape=shape)
    image = mono_to_color(image, use_deltas=use_deltas)
    image = ablu_transforms(image=image)['image']

    return image

def audio_batch_to_image_tensor(batch_audio, audio_cfg, epoch, image_preparer, ablu_transforms, train=True):
    batch_images = []
    for y in batch_audio:
        image = get_one_image(
            y=y.cpu().numpy(),
            audio_cfg=audio_cfg,
            epoch=epoch,
            image_preparer=image_preparer,
            ablu_transforms = ablu_transforms,
            train=train
        )
        image = torch.from_numpy(image).float()  # (H, W, C)
        image = image.permute(2, 0, 1)           # → (C, H, W)
        batch_images.append(image)

    return torch.stack(batch_images).to(batch_audio.device)  # (B, C, H, W) 

class ClassifierHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout_rate=0.2):
        super().__init__()
        self.linear = nn.Linear(in_channels, in_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output = nn.Linear(in_channels // 2, num_classes)

    def forward(self, x):          # [Batch, Chanels, Time]
        x = x.permute(0, 2, 1)     # [Batch,  Time, Chanels,]
        x = self.linear(x)    
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        x = x.permute(0, 2, 1)      # [B, num_classes, T]
        return x


class BirdSoundModel(pl.LightningModule):
    
    def init_layer(self, layer):
        nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

    def init_bn(self, bn):
        bn.bias.data.fill_(0.)
        bn.weight.data.fill_(1.0)

    def init_weights(self, model):
        '''Currently not in use'''
        classname = model.__class__.__name__
        if classname.find("Conv2d") != -1:
            nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
            model.bias.data.fill_(0)
        elif classname.find("BatchNorm") != -1:
            model.weight.data.normal_(1.0, 0.02)
            model.bias.data.fill_(0)
        elif classname.find("GRU") != -1:
            for weight in model.parameters():
                if len(weight.size()) > 1:
                    nn.init.orghogonal_(weight.data)
        elif classname.find("Linear") != -1:
            model.weight.data.normal_(0, 0.01)
            model.bias.data.zero_()


    class AttentionBlock(nn.Module):
        def __init__(self,
                     in_features: int,
                     out_features: int,
                     activation="linear",
                     image_shape = (1,1),
        ):
            super().__init__()

            self.activation = activation
            
            self.attention = nn.Conv1d(
                in_channels=in_features,
                out_channels=out_features,  #So we're doing per-class attention, because number of classes per sample is unknown
                kernel_size=3, #was 1 originally, changed to 3 with good results
                stride=1,
                padding=1,  #was 0 originally, changed to 1 to match above
                bias=True)
            
            with torch.no_grad():
                self.attention.weight.fill_(1.0 / (self.attention.kernel_size[0] * in_features))
                self.attention.bias.zero_()
            
            self.classify = ClassifierHead(in_channels=1280, num_classes=206)
            self.image_shape=image_shape
            self.num_chunks = int(self.image_shape[0]*self.image_shape[1])

        def init_layer(self, layer): #could access the outer class init_layer method instead
            nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, "bias"):
                if layer.bias is not None:
                    layer.bias.data.fill_(0.)    
            
        def init_weights(self):
            self.init_layer(self.attention)
            #self.init_layer(self.classify)

        def nonlinear_transform(self, x):
            if self.activation == 'linear':
                return x
            elif self.activation == 'sigmoid':
                return torch.sigmoid(x)

        def forward(self, x):

            # x: (batch_size, n_features, n_chunks * n_segments_per_chunk)

            # We can reshape to convolve only along the frequency dimension to operate on the time chunks independently. 
            # We don't need to do this for the logits, but keeping the same form in case we want to change the activation, 
            # or kernel size in a way that they are not independent of each other.

            batch_size = x.shape[0]  # Split along the third dimension
            split_length = x.shape[2] // self.num_chunks
            
            
            x = torch.split(x, split_length, dim=2)
            x = torch.cat(x, dim=0)  #  (128, 1280, 4)
            
            attn = self.attention(x) #.squeeze(1)
            norm_att = torch.softmax(torch.tanh(attn), dim=-1)/self.num_chunks  #so that they have a mean value of 1/16 each
            split_attn = torch.split(norm_att, batch_size, dim=0) #Put the weights back to their original shape
            norm_att = torch.cat(split_attn, dim=2)#.unsqueeze(-1) 

            seg_logits = self.classify(x)

            seg_logits = F.dropout(seg_logits, p=0.3, training=self.training)
            classify = self.nonlinear_transform(seg_logits)  #note - this is OK, because we're just doing a sigmoid, would be

            split_logits = torch.split(seg_logits, batch_size, dim=0)
            seg_logits = torch.cat(split_logits, dim=2)

            split_classify = torch.split(classify, batch_size, dim=0)
            classify = torch.cat(split_classify, dim=2)
            
            weighted_preds = norm_att * classify
            weighted_seg_logits = norm_att * seg_logits
            preds = weighted_preds.sum(dim=-1)   
            logit = weighted_seg_logits.sum(dim=-1)  #equivalent to mean because the weights are /16
            seg_logits = seg_logits.transpose(1, 2)

            return logit, seg_logits, preds

    def __init__(self,
                 cfg, 
                 classes,
                 audio,
                 paths=None,
                 in_channels=3,
                 gamma=2,
                 loss_alphas=None,
                 backbone_checkpoint_path=None,
                 #background_model=None,
                 background_df=None,
                ):
        super().__init__()

        self.classes = classes
        self.num_classes = len(self.classes)
        self.bn0 = nn.BatchNorm2d(3) #(audio.IMAGE_WIDTH) #if cfg.RESHAPE_IMAGE else nn.BatchNorm2d(audio.N_MELS)
        self.base_model = timm.create_model(
                                    cfg.MODEL, 
                                    pretrained=True, 
                                    in_chans=in_channels
                                    )
        self.start_backgrounds_epoch = cfg.START_BACKGROUNDS_EPOCH
        self.pseudolabel_weight = cfg.PSEUDOLABEL_WEIGHT

        layers = list(self.base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        self.train_image_transform = AbluTransforms(audio).train
        self.val_image_transform = AbluTransforms(audio).valid
        #self._background_model = background_model
        self.background_df = background_df


        if backbone_checkpoint_path is not None:
            checkpoint = torch.load(backbone_checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get("state_dict", checkpoint)

            # Filter to only load encoder/backbone weights
            encoder_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items() if "encoder." in k}

            missing, unexpected = self.encoder.load_state_dict(encoder_dict, strict=False)
            print(f"Backbone weights loaded with {len(missing)} missing and {len(unexpected)} unexpected keys.")

        if hasattr(self.base_model, "fc"):
            in_features = self.base_model.fc.in_features
        elif cfg.MODEL == 'eca_nfnet_l0':
            in_features = self.base_model.head.fc.in_features
        elif cfg.MODEL == 'convnext_tiny.in12k_ft_in1k':
            in_features = self.base_model.head.fc.in_features
        else:
            in_features = self.base_model.classifier.in_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.image_shape = audio.IMAGE_SHAPE
        self.att_block = self.AttentionBlock(in_features, 
                                            self.num_classes, 
                                            activation="sigmoid",
                                            image_shape=self.image_shape
                                            )
        self.loss_function = LossFunctions(cfg.LOSS_FUNCTION_NAME, gamma=gamma, loss_alphas=loss_alphas).loss
        self.init_weight()
        self.val_outputs = []
        self.train_outputs = []
        self.metrics_list = []
        self.val_epoch = 0
        self.epoch_to_unfreeze_backbone = cfg.EPOCHS_TO_UNFREEZE_BACKBONE,
        self.lr = cfg.LR
        self.initial_lr = cfg.INITIAL_LR
        self.min_lr = cfg.MIN_LR
        self.warmup_epochs = cfg.WARMUP_EPOCHS
        self.cycle_length = cfg.LR_CYCLE_LENGTH
        self.lr_decay = cfg.LR_DECAY
        self.printed_shapes = False
        self.use_data_updates = False #cfg.USE_UNLABELED_SOUNDSCAPES
        self.data_update_1 = None #cfg.FIRST_DATA_UPDATE
        self.data_update_2 = None #cfg.SECOND_DATA_UPDATE
        self.use_mixup = cfg.USE_MIXUP
        self.mixup_alpha = cfg.MIXUP_ALPHA
        if paths is not None:
            self.temp_dir = Path(paths.temp_dir)
            self.results_dir = Path(paths.out_dir)
        self.image_preparer = PrepareImage(height=audio_cfg.N_MELS, width = audio_cfg.CHUNK_WIDTH)
        self.audio = audio
        self.positive_backgrounds_used = 0
        self.negative_backgrounds_used = 0
        self.all_backgrounds_used = 0
        
    
    def init_weight(self):
        self.init_bn(self.bn0)
        self.init_layer(self.fc1)

    def forward(self, x):

        #example = x[0][0]  # shape: (Freq, Time)
        #plt.figure(figsize=(10, 4))
        #plt.imshow(example.cpu().numpy(), origin='lower', aspect='auto', cmap='magma')
        #plt.title("Mel Spectrogram")
        #plt.xlabel("Time")
        #plt.ylabel("Frequency")
        #plt.colorbar(label='Amplitude (dB)')
        #plt.tight_layout()
        #plt.show()

        # (batch_size, channels, mel_bins,  time_steps)
        x = self.bn0(x)
        x = self.encoder(x)  #This is the image passing through the base model  8x8 out with a 256x256 image
        
        #This is the guts of the SED part.  So first we need to unfold back into original shape so agregation has it's spatial meaning
        
        if self.image_shape == (2,2):  #Stack the (1,2) and (2,2) scenarios in the frequency direction
            half = x.shape[2]//2
            x0 = x[:,:,:half,:half]
            x1 = x[:,:,:half,half:]
            x2 = x[:,:,half:,:half]
            x3 = x[:,:,half:,half:]
            x = torch.cat((x0,x1,x2,x3), dim=2) #stack vertically along the frequency direction, so now it's 16 high, 4 wide for a 256x256 input image
        elif self.image_shape == (1,4):  #Stack the (1,2) and (2,2) scenarios in the frequency direction
            quarter = x.shape[2]//4
            x0 = x[:,:,:,:quarter]
            x1 = x[:,:,:,quarter:2*quarter]
            x2 = x[:,:,:,2*quarter:3*quarter]
            x3 = x[:,:,:,3*quarter:]
            x = torch.cat((x0,x1,x2,x3), dim=2) #stack vertically along the frequency direction, so now it's 16 high, 4 wide for a 256x256 input image
        elif self.image_shape == (1,2):
            half = x.shape[3]//2
            x0 = x[:,:,:,:half]
            x1 = x[:,:,:,half:]
            x = torch.cat((x0,x1), dim=2) #For a 128x128 (2,1) image, we'd now have 8 high in frequency, 2 wide in time
        elif self.image_shape == (2, 0.5):
            half = x.shape[2]//2
            x0 = x[:,:,:half,:]
            x1 = x[:,:,half:,:]
            
            x = torch.cat((x0,x1), dim=3)  #For a 256x256 (2, 0.5) this should be now 4 high in frequency, 16 wide in time
            #print(f'concatenated shape {x.shape}')
            #x = x.transpose()  #should be batch, features, freq

        #For the (2,1) and (1,1) cases we donn't need to do anything here, there is only one chunk represented along the horizontal axis.
        dimension = 2 if self.image_shape == (2, 0.5) else 3
        x = torch.mean(x, dim=dimension) # Aggregate in short axis, but only over each chunk
        #print(f'shape after mean {x.shape}')
        x = F.dropout(x, p=0.2, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.3, training=self.training)

        (logit, segment_logits, preds) = self.att_block(x) #  for (2,1) in: [64, 1280, 16], out: [64, 182], [64, 16, 182],  [64, 182]
        
        output_dict = {
            'clip_preds': preds,  #predictions for AP and CMAP calculation
            'segmentwise_logit': segment_logits,   #[64, 16, 182]  # doesn't use the attn or activation function
            'logit': logit,  #torch.Size([64, 182]) used for the loss calculation, includes attention   
        }

        return output_dict

    def configure_optimizers(self):
        def custom_lr_scheduler(epoch):
            '''CosineAnealingWarmRestarts but with a decay between cycles and a warmup'''
            initial = self.initial_lr / self.lr 
            rel_min = self.min_lr / self.lr
            step_size = (1-initial) / self.warmup_epochs
            warmup = initial + step_size * epoch if epoch <= self.warmup_epochs else 1
            cycle = epoch-self.warmup_epochs
            decay = 1 if epoch <= self.warmup_epochs else self.lr_decay ** (cycle // self.cycle_length)
            phase = np.pi * (cycle % self.cycle_length) / self.cycle_length
            cos_anneal = 1 if epoch <= self.warmup_epochs else  rel_min + (1 - rel_min) * (1 + np.cos(phase)) / 2
            return warmup * decay * cos_anneal #this value gets multipleid by the initial lr (self.lr)
        
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = LambdaLR(optimizer, lr_lambda=custom_lr_scheduler)
        return [optimizer], [scheduler]
    
    def train_with_mixup(self, X, y):
        X, y_a, y_b, lam = mixup_data(X, y, self.mixup_alpha, self.device)
        preds_dict = self(X)
        loss_mixup = mixup_criterion(self.loss_function, preds_dict, y_a, y_b, lam)
        return loss_mixup, preds_dict



    def training_step(self, batch, batch_idx):
        audios, targets = batch

        #random_choices = random.choice(['sumix', 'sumix',  'nomix']) #'cutmix',
        #random_choices = random.choices(['sumix', 'cutmix', 'nomix'], weights=[2, 0, 1])[0]

        num_sumix = random.choice([0,1,1,2])

        if num_sumix == 1:
            audios, targets = sumix(audios, targets, min_percent=0.3)
        elif num_sumix == 2:
            audios, targets = sumix(audios, targets, min_percent=0.5)
            audios, targets = sumix(audios, targets, min_percent=0.5)

        if self.pseudolabel_weight is not None: 
            if self.current_epoch >= self.start_backgrounds_epoch: # and random.random() < .2:
                batch_size = audios.shape[0]
                num_backgrounds = 8
                duration = audios.shape[1]//32000
                X, Y = get_background_crops(self.background_df, num_backgrounds*2)  #return backgrounds in the form of [num_backgrounds, wave_array], [num_backgrounds, labels]
                backgrounds, background_preds = merge_pairs(X, Y) 
                backgrounds = backgrounds.to(self.device)
                background_preds = background_preds.to(self.device) 


                
                #backgrounds = get_backgrounds(paths.background_audios, background_size, duration)  #return backgrounds in the form of [batch, wave_array]
                #background_images = audio_batch_to_image_tensor(
                #                                                backgrounds, 
                #                                                self.audio, 
                #                                                self.current_epoch, 
                #                                                self.image_preparer,
                #                                                ablu_transforms = self.val_image_transform,
                #                                                train=False
                #                                                )
                #background_images = background_images.to(self.device)
                #with torch.no_grad():
                #    if self.current_epoch <=25:
                #        positive_threshold = 0.6
                #        negative_threshold = 0.1
                #        background_preds = self._background_model(background_images)['clip_preds']
                #    else:
                #        positive_threshold = 0.5
                #        negative_threshold = 0.1
                #        background_preds = self(background_images)['clip_preds']
                #background_preds = background_preds.detach()

                #positive_mask = background_preds >= positive_threshold  # shape [batch_size, num_classes], bool
                #negative_mask = background_preds <= negative_threshold 
                #combined_mask = positive_mask | negative_mask  #Selects the samples to 

                #background_preds = torch.where(positive_mask, background_preds, torch.zeros_like(background_preds)) 
                #background_preds = torch.where(positive_mask, torch.ones_like(background_preds), torch.zeros_like(background_preds))

                #positive_samples = (positive_mask != 0).any(dim=1)  # At least one non-zero value
                #negative_samples = (~positive_samples) & negative_mask.all(dim=1)    # All values are zero
                #num_negative = negative_samples.sum().item()
                #num_positive = positive_samples.sum().item()
                #class_positive_counts = positive_mask.sum(dim=0)
                #self.negative_backgrounds_used += num_negative
                #self.positive_backgrounds_used += num_positive
                #if not hasattr(self, "class_positive_counts"):
                #    self.class_positive_counts = torch.zeros_like(class_positive_counts)
                #self.class_positive_counts += class_positive_counts
                
                #rarity = 1.0 / (self.class_positive_counts.float() + 1e-6)  # shape [num_classes]
                #rarity_scores = (positive_mask.float() * rarity).sum(dim=1)  # shape [batch_size]
                #rsorted_scores, sorted_indices = rarity_scores.sort(descending=True)

                #top_k = background_size // 4
                #mid_k = background_size  // 2

                # Get indices
                #top_indices = sorted_indices[:top_k]     # Top 25%
                #mid_indices = sorted_indices[top_k:top_k + mid_k]  # Middle 50%

                # Combine: keep middle + duplicate top
                #final_indices = torch.cat([mid_indices, top_indices, top_indices], dim=0).to(backgrounds.device)
                #final_indices = torch.cat([top_indices, top_indices, top_indices, top_indices], dim=0).to(backgrounds.device)
                
                # Apply to all relevant tensors
                #backgrounds = backgrounds[final_indices]
                #background_preds = background_preds[final_indices]
                #combined_mask = combined_mask[final_indices]

                #zero_samples = (combined_mask.sum(dim=1) == 0)  # Samples that were neither postive nor negative mask
                #backgrounds[zero_samples] = 0

                #used_samples = positive_samples | negative_samples
                #num_used = used_samples.sum().item()
                #self.all_backgrounds_used += num_used


                padding_rows = batch_size - num_backgrounds
                if padding_rows > 0:
                    prediction_zeros = torch.zeros((padding_rows, background_preds.shape[1]), device=background_preds.device, dtype=background_preds.dtype)
                    background_preds = torch.cat([background_preds, prediction_zeros], dim=0)
                    background_zeros = torch.zeros((padding_rows, backgrounds.shape[1]), device=backgrounds.device, dtype=backgrounds.dtype)
                    backgrounds = torch.cat([backgrounds, background_zeros], dim=0)

                targets = torch.clamp(targets*self.pseudolabel_weight + background_preds, max=1) #Allows secondary labels to be summed with pseudolabels, to a max of 1

                audios = backmix(audios, backgrounds, max_percent=0.8, min_percent=0.2) #Only the backgrounds get scaled
        elif self.start_backgrounds_epoch is not None:
            batch_size = audios.shape[0]
            duration = audios.shape[1]//32000
            backgrounds = get_backgrounds(paths.background_audios, batch_size, duration)  #return backgrounds in the form of [batch, wave_array]
            audios = backmix(audios, backgrounds, max_percent=.8, min_percent=0.2)

        images = audio_batch_to_image_tensor(
            audios, 
            self.audio, 
            self.current_epoch, 
            self.image_preparer,
            ablu_transforms = self.train_image_transform,
            train=True
            )

        #if random_choices == 'mixup':
        #    loss, preds_dict = self.train_with_mixup(images, targets)
        #else:
        preds_dict = self(images)
        loss = self.loss_function(preds_dict, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        train_output = {"train_loss": loss, "logits": preds_dict['clip_preds'], "targets": targets}
        self.train_outputs.append(train_output) 
        return loss        

    def validation_step(self, batch, batch_idx):
        audios, targets = batch 

        images = audio_batch_to_image_tensor(
            audios, 
            self.audio, 
            self.current_epoch, 
            self.image_preparer,
            ablu_transforms = self.val_image_transform,
            train=False
            )

        preds_dict = self(images)  #Should the loss definately be based on framewise output?
        if not self.printed_shapes:
            print(f'logit', self(images)['logit'].shape)                        #[64, 79]  Batch,num_classes
            print(f'segmentwise_logit shape', self(images)['segmentwise_logit'].shape)    # [64, 8, 79]  
            self.printed_shapes = True
        val_loss = self.loss_function(preds_dict, targets)  #Why the need to convert to float here
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        output = {"val_loss": val_loss, "logits": preds_dict['logit'], "targets": targets}
        self.val_outputs.append(output)  #new line
        return output

    def train_dataloader(self):
        return self._train_dataloader

    def validation_dataloader(self):
        return self._validation_dataloader
    
    def on_train_epoch_start(self):
        epoch = self.current_epoch
        train_loader = self.trainer.train_dataloader
        train_loader.dataset.reset_wave_augmentation(epoch)
        train_loader.dataset.set_epoch(epoch)

    def on_train_epoch_end(self, *args, **kwargs):  
        epoch = self.current_epoch

        #print(f'The number of negative background samples used as augmentation during this epoch was {self.negative_backgrounds_used}')
        #print(f'The number of positive background samples used as augmentation during this epoch was {self.positive_backgrounds_used}')
        #print(f'The total number of backgrounds mixed for thie epoch was {self.all_backgrounds_used}')
        #print("Positive class counts:", self.class_positive_counts.tolist())

        #self.negative_backgrounds_used = 0
        #self.positive_backgrounds_used = 0
        #self.all_backgrounds_used = 0

        if epoch == self.epoch_to_unfreeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = True
            print(f'Unfreezing the backbone after {epoch} epochs')

        if self.use_data_updates:   
            if (epoch + 1) == self.data_update_1:
                train_loader = self.trainer.train_dataloader
                train_loader.dataset.update_df(df_idx=1)
                print(f'Using the second dataset from epoch {epoch+1}')
            elif (epoch + 1) == self.data_update_2:
                train_loader = self.trainer.train_dataloader
                train_loader.dataset.update_df(df_idx=2)
                print(f'Using the third dataset from epoch {epoch+1}')

    def on_validation_epoch_end(self):
        val_outputs = self.val_outputs
        avg_val_loss = torch.stack([x['val_loss'] for x in val_outputs]).mean().cpu().detach().numpy()
        output_val_logits = torch.cat([x['logits'] for x in val_outputs],dim=0)
        val_targets = torch.cat([x['targets'] for x in val_outputs],dim=0).cpu().detach().numpy() 
        
        train_outputs = self.train_outputs
        if train_outputs:
            train_losses = [x['train_loss'].cpu().detach().numpy() for x in train_outputs]
            avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
        else: 
            avg_train_loss = avg_val_loss #we need this because the first time it's an empty list
        
        val_probs = output_val_logits.sigmoid().cpu().detach().numpy()

        val_pred_df = pd.DataFrame(val_probs, columns = self.classes)
        val_target_df = pd.DataFrame(val_targets, columns = self.classes)

        avg_score = padded_cmap(val_target_df, val_pred_df, padding_factor = 5)  #Val CMAP5
        avg_score3 = skm.label_ranking_average_precision_score(val_targets,val_probs)  #Val LRAP
        optimizer_lr = self.trainer.optimizers[0].param_groups[0]['lr']  
        print(f"Learning rate from optimiser at epoch {self.current_epoch}: {optimizer_lr}")

        if self.val_epoch > 0:
            self.metrics_list.append({'train_loss':avg_train_loss, 
                                      'val_loss': avg_val_loss, 
                                      'val_map' : avg_score,    #Val CMAP5
                                      'val_prec':avg_score3,    #Val LRAP
                                      #'train_map':avg_score4,  #Train CMAP5
                                      'learning_rate':optimizer_lr
                                     })

        print(f'Epoch {self.current_epoch} train loss {avg_train_loss}')
        print(f'Epoch {self.current_epoch} validation loss {avg_val_loss}')
        print(f'Epoch {self.current_epoch} validation C-MAP score pad 5 {avg_score}')
        print(f'Epoch {self.current_epoch} validation AP score {avg_score3 or None}')
        
        
        val_target_df.to_pickle(self.results_dir / 'val_target_weighted_df.pkl')
        val_target_df = val_target_df.map(lambda x: 1 if x != 0 else 0)
        val_target_df.to_pickle(self.results_dir / 'val_target_df.pkl')
        val_pred_df.to_pickle(self.results_dir / 'val_pred_df.pkl') 

        self.val_outputs = []
        self.train_outputs = []
        self.val_epoch +=1

        return
    
    def get_my_metrics_list(self):
        return self.metrics_list


############################################# Training Functions ######################################
######################################################################################################

def get_model(ckpt_path, cfg, audio, paths, classes):
    model = BirdSoundModel(cfg, classes, audio, paths)
    checkpoint = torch.load(ckpt_path)
    full_state_dict = checkpoint['state_dict']
    filtered_state_dict = {
        k: v for k, v in full_state_dict.items() if not k.startswith("_background_model.")
    }

    model.load_state_dict(filtered_state_dict, strict=True)  # strict=False for safety
    model.to(cfg.DEVICE)
    return model


def save_models(paths, train_cfg, audio_cfg, classes, deploy_ckpt_selection=1):
    '''This is overkill, but I imagine wanting to modify to pickle 
    the whole model instead of just the checkpoints'''
    checkpoints = [path for path in Path(paths.chkpt_dir).glob('*.ckpt')]
    latest_ckpt_first = sorted(checkpoints, key=lambda p: p.stat().st_ctime, reverse=True)
    selection_idx = max(deploy_ckpt_selection, len(latest_ckpt_first))-1
    
    for idx, ckpt_path in tqdm(enumerate(latest_ckpt_first)) :
        model = get_model(ckpt_path, train_cfg, audio_cfg, paths, classes)
        save_path = str(Path(paths.out_dir) / ckpt_path.name)
        deploy_path = str(Path(paths.model_deploy) / ckpt_path.name)
        torch.save(model.state_dict(), save_path)
        if idx == selection_idx:
            torch.save(model.state_dict(), deploy_path)
        print(Blue.S + 'Weights checkpoint saved to: ' + Blue.E + save_path)

    return save_path  #just returns what ever came last, to check for functionality


def get_class_weights(df):
    df = df.iloc[:, 2:] # removing the 'filepath' and 'primary_label' columns
    col_sums = df.sum()
    counts_array = col_sums.values
    counts_array = np.sqrt(300 + counts_array) 
    class_weights = counts_array.tolist()
    sample_idxs = np.argmax(df.values, axis=1).tolist()
    sampling_weights = [1 / class_weights[idx] for idx in sample_idxs] 
    return sampling_weights


def get_dataloaders(df_train, 
                    df_valid,
                    train_df_short,
                    audio, 
                    batch_size, 
                    num_workers,
                    augmentation_updates=[6,12]):
    

    ds_train = WaveformDataset(df_train, 
                               audio,
                               train=True,
                               augmentation_updates=augmentation_updates,
                               short_df=train_df_short)
    
    ds_val = WaveformDataset(df_valid, 
                             audio,
                             train=False)
    
    #if weighted_sampling is not None:
    #    sample_weights = get_class_weights(df_train)
    #    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(ds_train))
    #    dl_train = DataLoader(ds_train, batch_size=batch_size, sampler=sampler, num_workers=num_workers)   #persistent_workers=True
    #else:
    
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)    #persistent_workers=True
    dl_val = DataLoader(ds_val, batch_size=batch_size, num_workers = num_workers)   

    return dl_train, dl_val, ds_train, ds_val


def get_loss_alphas(classes, data_cfg, cfg):
    '''Returns individualised alpha parameters for BCE focal loss'''
    low_indices = [classes.index(x) for x in data_cfg.LOW_ALPHA_CLASSES if x in classes]
    high_indices = [classes.index(x) for x in data_cfg.HIGH_ALPHA_CLASSES if x in classes]
    alphas = np.full(len(classes), cfg.MID_ALPHA)
    alphas[low_indices] = cfg.LOW_ALPHA
    alphas[high_indices] = cfg.HIGH_ALPHA
    return alphas


def run_training(dl_train, dl_val, data_cfg, train_cfg, audio_cfg, checkpoint_dir, background_df, pretrained_path):  #background_model
    reset_epoch = train_cfg.RESET_EPOCH
    #pl.seed_everything(train_cfg.SEED, workers=True)
    torch.set_flush_denormal(True)
    torch.set_float32_matmul_precision('medium')  
    print(f"Running training...")
    logger = None
    classes = dl_train.dataset.classes
    loss_alphas = get_loss_alphas(classes, data_cfg, train_cfg)


    audio_model = BirdSoundModel(train_cfg,
                                 classes,
                                 audio_cfg,
                                 paths,
                                 gamma=train_cfg.FOCAL_GAMMA,
                                 loss_alphas=loss_alphas,
                                 backbone_checkpoint_path=pretrained_path,
                                 #background_model = background_model
                                 background_df = background_df
                                 )


    resetting_early_stop_callback = ResettingEarlyStopping(monitor="val_loss", 
                                        reset_epoch=9,
                                        min_delta=train_cfg.MIN_DELTA, 
                                        patience=train_cfg.PATIENCE, 
                                        verbose= True, 
                                        mode="min")
    
    resetting_checkpoint_callback = ResettingModelCheckpoint(
                                        reset_epoch=reset_epoch,
                                        save_top_k=train_cfg.KEEP_LAST,
                                        monitor="val_loss",
                                        mode="min",
                                        dirpath=checkpoint_dir,
                                        save_last=True,
                                        save_weights_only=True,
                                        verbose=True,
                                        )
    
    early_stop_callback = ResettingEarlyStopping(monitor="val_loss", 
                                        min_delta=train_cfg.MIN_DELTA, 
                                        patience=train_cfg.PATIENCE, 
                                        verbose= True, 
                                        mode="min")


    # saves top- checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(save_top_k=train_cfg.KEEP_LAST,
                                          monitor="val_loss",
                                          mode="min",
                                          dirpath=checkpoint_dir,
                                          save_last= True,
                                          save_weights_only=True, 
                                          verbose= True,
                                          # filename=f'birdCLEF23-{epoch:02d}-{val_loss:.4f}', need to figure this out so It can update to a dataset
                                          )



    callbacks_to_use = [checkpoint_callback]  #, early_stop_callback

    trainer = pl.Trainer(
        val_check_interval=0.5,
        deterministic=True,
        max_epochs=train_cfg.EPOCHS,
        logger=logger,
        callbacks=callbacks_to_use,
        precision=train_cfg.PRECISION, 
        accelerator=train_cfg.GPU,
        reload_dataloaders_every_n_epochs=1
        )

    print("Running trainer.fit")
    trainer.fit(audio_model, train_dataloaders=dl_train, val_dataloaders=dl_val)       
    gc.collect()
    torch.cuda.empty_cache()

    return audio_model.get_my_metrics_list()


def extract_results(metrics, paths):
    train_losses = [x['train_loss'] for x in metrics]
    val_losses = [x['val_loss'] for x in metrics]  
    #train_map = [x['train_map'] for x in metrics]  
    val_map = [x['val_map'] for x in metrics]  
    val_precision = [x['val_prec'] for x in metrics] 
    learning_rates =  [x['learning_rate'] for x in metrics]
    time_axis = [0.5*x for x in range(1,len(val_losses)+1)]

    fig, ax = plt.subplots()
    plt.plot(time_axis, train_losses, 'r', label='Train Loss')
    plt.plot(time_axis, val_losses, '--k', label='Val Loss')
    plt.legend()
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ax.tick_params('both', colors='r')

    ax2 = ax.twinx()
    #plt.plot(time_axis, train_map, 'b', label='Train cmap5')
    plt.plot(time_axis, val_precision, '--g', label='Val LRAP')
    plt.plot(time_axis, val_map, ':m', label='Val cmap5')
    ax2.set_ylabel('Accuracy')
    plt.legend()
    plt.legend(loc='lower left')
    ax.tick_params('both', colors='b')
    plt.savefig(Path(paths.out_dir) / f"exp_{use_case['experiment']}_training_metrics.jpg")
    plt.close()

    pred = pd.read_pickle(Path(paths.out_dir) / f'val_pred_df.pkl')
    true = pd.read_pickle(Path(paths.out_dir) / f'val_target_df.pkl')

    print(f' The final Val CMAP score is {padded_cmap(true, pred, padding_factor = 5)}')

    plt.plot(time_axis[::2], learning_rates[::2], marker='o')  #no need to plot the half-epoch rates
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.savefig(Path(paths.out_dir) / 'learning_rate.jpg')
    plt.close()


############################################# Prepare Data  ######################################
##################################################################################################

def load_training_data(labels_path, data_path, excluded_classes, use_secondary):
    '''Load the datframe from csv, clean any irrelevent secondary labels, and varify that all the files actually exist'''
    use_cols = ['filename', 'primary_label', 'secondary_labels']  #'secondary_labels'  'latitude', 'longitude'
    in_df = pd.read_parquet(labels_path, engine='pyarrow')
    in_df['filepath'] = str(data_path) + '/' + in_df['filename']
    #print(f'There are {len(in_df)} audio samples in the labels original dataframe')
    in_df = in_df[~in_df['primary_label'].isin(excluded_classes)]
    unique_birds = sorted(in_df['primary_label'].unique()) 

    #print(f'Training + Validation with {in_df.shape[0]} audio samples')
    #print(f'There are {len(unique_birds)} primary class labels')

    def remove_unused_birds(second_bird_list):
        return [string for string in second_bird_list if string in unique_birds]
    if use_secondary:
        in_df['secondary_labels'] = in_df['secondary_labels'].apply(ast.literal_eval)
        in_df['secondary_labels'] = in_df['secondary_labels'].apply(remove_unused_birds)

    #Let's check that all the training samples in the dataframe actually exist.  Remove any rows that can't be found.
    #original_length = len(in_df)
    training_samples = set([path for path in Path(data_path).rglob('*') if path.suffix in {'.ogg', '.flac'}])
    in_df['filepath'] = in_df['filepath'].apply(Path)
    in_df = in_df[in_df['filepath'].isin(training_samples)]
    in_df['filepath'] = in_df['filepath'].apply(str)
    #new_length = len(in_df)

    #print(Blue.S + 'The original length was: ' +Blue.E, original_length)
    #print(Blue.S + 'The final length was: ' +Blue.E, new_length)
    #if original_length > new_length:
    #    print(Blue.S + 'Samples removed: ' + Blue.E, original_length - new_length)

    in_df['centres'] = in_df['centres'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    return in_df


def filter_by_location(df, limits=None):
    if limits is not None:
        df = df[
                (df['longitude'] >= limits['WEST']) &
                (df['longitude'] <= limits['EAST']) &
                (df['latitude'] >= limits['SOUTH']) &
                (df['latitude'] <= limits['NORTH'])]
    return df


def limit_max_per_class(df, max_per_class = None):
    '''Put an upper limit on class size to prevent extreme class imbalance'''
    if max_per_class is not None:
        class_counts = df['primary_label'].value_counts()
        classes_to_reduce = class_counts[class_counts > max_per_class].index
        def downsample_class(df, class_label, max_rows):
            df_class = df[df['primary_label'] == class_label]
            return df_class.sample(n=max_rows)
        df_list = [downsample_class(in_df, class_label, max_per_class) if class_label in classes_to_reduce 
                else df[df['primary_label'] == class_label]
                for class_label in df['primary_label'].unique()]
        df = pd.concat(df_list)
    return df


def split_classes_by_size(df, threshold):
    '''Temporarily drop any super rare classes from the dataframe, so they don't end up 
    loosing precious samples from training due to location or splitting.'''
    mask = df['primary_label'].map(df['primary_label'].value_counts()) > threshold
    common_df = df[mask]
    common_df = common_df.reset_index(drop=True)
    mask = df['primary_label'].map(df['primary_label'].value_counts()) <= threshold
    rare_df = df[mask]
    rare_df = rare_df.reset_index(drop=True)
    return common_df, rare_df


def duplicate_rare_rows(df, min_samples):
    '''Upsample the super-rare classes to some minimum'''
    value_counts = df['primary_label'].value_counts()
    duplication_needed = {label: min_samples - count for label, count in value_counts.items()}

    duplicated_rows = []
    for label, count in duplication_needed.items():
        label_rows = df[df['primary_label'] == label]
        num_duplicates = count // len(label_rows)  # Number of full duplications needed
        remainder = count % len(label_rows)        # Remaining duplications needed

        if num_duplicates > 0:
            duplicated_full_sets = pd.concat([label_rows] * num_duplicates, ignore_index=True)
            duplicated_rows.append(duplicated_full_sets)

        if remainder > 0:
            duplicated_remainder = label_rows.sample(n=remainder, replace=True)
            duplicated_rows.append(duplicated_remainder)

    df = pd.concat([df] + duplicated_rows, ignore_index=True)
    #final_counts = df['primary_label'].value_counts()
    #print(final_counts[-10:])
    return df


def train_val_split(common_df, rare_df, n_folds=10, max_per_class_val=None):
    '''Split training and validation samples, but limiting the max in the validation set
    and also not using any super-rare sample in the validation set'''
    skf =StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2024)
    target = common_df['primary_label'] 

    for train_index, val_index in skf.split(common_df, target):
        tn_df, val_df = common_df.iloc[train_index].copy(), common_df.iloc[val_index].copy()
    train_df = pd.concat([tn_df, rare_df])

    #print(f'The training dataframe has {train_df.shape[0]} rows\n'
    #    f'The validation dataframe has {val_df.shape[0]} rows')
    excess_samples_list = []
    def limit_class_samples(group):
            if len(group) > max_per_class_val:
                # Split into excess (to move back to tn_df) and limited samples (to keep in val_df)
                excess_samples = group.iloc[max_per_class_val:]
                excess_samples_list.append(excess_samples)
                return group.iloc[:max_per_class_val]
            return group

    val_df = val_df.groupby('primary_label').apply(limit_class_samples).reset_index(drop=True)

    if excess_samples_list:
        excess_samples_df = pd.concat(excess_samples_list, ignore_index=True)
        train_df = pd.concat([train_df, excess_samples_df], ignore_index=True)

    #print(f'After rebalancing the training dataframe has {train_df.shape[0]} rows\n'
    #      f'The validation dataframe has {val_df.shape[0]} rows')
    return train_df, val_df


def multi_binarize(df, unique_birds, secondary_weights):
    keep_cols = ['primary_label', 'filepath', 'centres']
    
    mlb = MultiLabelBinarizer(classes=unique_birds)
    df_primary = pd.concat([df, pd.get_dummies(df['primary_label']).astype('uint8')], axis=1)
    missing_birds = list(set(unique_birds).difference(list(df.primary_label.unique())))
    df_primary = pd.concat([df_primary, pd.DataFrame(0, index=df_primary.index, columns=missing_birds)], axis=1)
    df_primary = df_primary[unique_birds] # To synchronise the column order
    #df['combined_labels'] = df.apply(lambda row: [row['primary_label']] + row['secondary_labels'], axis=1)
    secondary_array = mlb.fit_transform(df['secondary_labels']).astype('uint8')
    combined_array = secondary_array * secondary_weights + df_primary[unique_birds].values
    label_df = pd.DataFrame(combined_array, columns=unique_birds)
    df = df[keep_cols]
    df = df.reset_index(drop=True)
    df = pd.concat([df, label_df], axis=1)

    return df


def encode_data(train_df, val_df, secondary_weights_train, secondary_weights_val, unique_birds):
    keep_cols = ['primary_label', 'secondary_labels', 'filepath', 'centres']
    keep_no_secondary =  ['primary_label', 'filepath', 'centres']
    train_df = train_df[keep_cols]
    val_df = val_df[keep_cols]

    if secondary_weights_train > 0:
        train_df_0 = multi_binarize(train_df, unique_birds, secondary_weights_train)
        val_df =  multi_binarize(val_df, unique_birds, secondary_weights=secondary_weights_val)
    else:
        train_df = train_df[keep_no_secondary]
        val_df = val_df[keep_no_secondary]
        train_df_0 = pd.concat([train_df, pd.get_dummies(train_df['primary_label']).astype('uint8')], axis=1)
        val_df = pd.concat([val_df, pd.get_dummies(val_df['primary_label']).astype('uint8')], axis=1)
        missing_birds = list(set(unique_birds).difference(list(val_df.primary_label.unique())))
        val_df = pd.concat([val_df, pd.DataFrame(0, index=val_df.index, columns=missing_birds)], axis=1)
        val_df = val_df[train_df_0.columns] # To synchronise the column order

    df_numeric = train_df_0.iloc[:, 2:].values
    non_uint8_values = df_numeric[df_numeric.dtype != np.uint8]
    return train_df_0, val_df


############################################# Main Script  #######################################
##################################################################################################

train_cfg = TrainingParameters(options=use_case)
data_cfg = BirdData()
audio_cfg = DefaultAudio()
paths = FilePaths(options=use_case)

Path(paths.out_dir).mkdir(parents=True, exist_ok=True)
Path(paths.temp_dir).mkdir(parents=True, exist_ok=True)
Path(paths.chkpt_dir).mkdir(parents=True, exist_ok=True)
Path(paths.model_deploy).mkdir(parents=True, exist_ok=True)

if (__name__ == '__main__'):
    print(Blue.S + f'Training with: ' + Blue.E + train_cfg.GPU)
    print(Blue.S + 'CPUs for available for dataloading: ' + Blue.E + str(train_cfg.NUM_WORKERS))

#This could all be moved into a data prep class
in_df = load_training_data(paths.LABELS_PATH, paths.DATA_DIR, data_cfg.EXCLUDED_CLASSES, data_cfg.USE_SECONDARY)
in_df = filter_by_location(in_df, limits=data_cfg.SPATIAL_LIMITS)
in_df = limit_max_per_class(in_df, data_cfg.MAX_PER_CLASS)
unique_birds = sorted(list(in_df['primary_label'].unique()))
common_df, rare_df = split_classes_by_size(in_df, data_cfg.RARE_THRESHOLD)  
rare_df = duplicate_rare_rows(rare_df, data_cfg.RARE_THRESHOLD)
train_df, val_df = train_val_split(common_df, rare_df, n_folds=data_cfg.N_FOLDS, max_per_class_val=data_cfg.MAX_PER_CLASS_VAL)

short_df = pd.read_parquet(paths.SHORT_FILES_PATH, engine='pyarrow')
short_df['filepath'] = str(paths.DATA_DIR) + '/' + in_df['filename']
val_files = val_df['original_file'].to_list()
short_df = short_df[~short_df['original_file'].isin(val_files)]

val_df.to_csv(paths.val_df_path, index=False)

train_df, val_df = encode_data(train_df, 
                                 val_df, 
                                 data_cfg.SECONDARY_WEIGHTS_TRAIN, 
                                 data_cfg.SECONDARY_WEIGHTS_VAL, 
                                 unique_birds)


short_df_files = short_df['filepath'].to_list()
normalized_files = [str(Path(f).resolve()) for f in short_df_files]
train_df['filepath'] = train_df['filepath'].apply(lambda x: str(Path(x).resolve()))
train_df_short = train_df[train_df['filepath'].isin(normalized_files)]

augmentation_updates = [train_cfg.FIRST_AUGMENTATION_UPDATE, train_cfg.SECOND_AUGMENTATION_UPDATE]

dl_train, dl_val, ds_train, ds_val = get_dataloaders(train_df, 
                                                     val_df,
                                                     train_df_short,
                                                     audio_cfg,
                                                     batch_size=train_cfg.BATCH_SIZE,
                                                     num_workers=train_cfg.NUM_WORKERS,
                                                     augmentation_updates=augmentation_updates)

save_naming_scheme(train_df, val_df, ds_train.classes, paths.bird_map_for_model)
save_model_config(paths, audio_cfg, train_cfg)

if (__name__ == '__main__') and (train_cfg.TRAIN):

    #background_model = BirdSoundModel(train_cfg, unique_birds, audio_cfg, paths)
    #model_state_dict = torch.load(paths.BACKGROUND_MDL_CKPT)
    #background_model.load_state_dict(model_state_dict)  
    #background_model.eval()
    background_df = pd.read_parquet(paths.background_pseudolabels)
    background_df["filepath"] = background_df["filename"].apply(lambda fn: paths.background_crops_dir / fn)
    background_df = background_df.drop(columns=['filename'])
    '/home/olly/Desktop/Kaggle_BC25/Data/Experiments/Exp_62_pretrain/Results/epoch=14-step=40440.ckpt'
    metrics = run_training(dl_train, dl_val, data_cfg, train_cfg, audio_cfg, paths.chkpt_dir, background_df, pretrained_path=paths.pretrained_ckpt)   # background_model
    extract_results(metrics, paths)
    last_path = save_models(paths, train_cfg, audio_cfg, unique_birds)
    final_model = BirdSoundModel(train_cfg, unique_birds, audio_cfg, paths)  # Now checking it loads OK
    model_state_dict = torch.load(last_path)
    final_model.load_state_dict(model_state_dict)  