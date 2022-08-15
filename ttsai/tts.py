# 数値演算
import numpy as np
import torch
from torch import nn
# 音声波形の読み込み
from scipy.io import wavfile
# フルコンテキストラベル、質問ファイルの読み込み
#from nnmnkwii.io import hts
# 音声分析
#import pyworld
# 音声分析、可視化
#import librosa
#import librosa.display
# Pythonで学ぶ音声合成
#import ttslearn
#import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
import joblib
import pyopenjtalk
#from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForXVector
from .models.ttslearn.tacotron.frontend.openjtalk import text_to_sequence, pp_symbols
from .models.models import Tacotron2, ParallelWaveGANGenerator


#話者の潜在表現を獲得
"""
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sv")
model = Wav2Vec2ForXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv")
def wav2sp_emb(wav_file):
    sr = 16000
    _sr, x = wavfile.read(wav_file)
    if x.dtype in [np.int16, np.int32]:
        x = (x / np.iinfo(x.dtype).max).astype(np.float64)
    if _sr != sr:
        x = librosa.resample(x, _sr, sr)
    inputs = feature_extractor(
    [x], sampling_rate=sr, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        embeddings = model(**inputs).embeddings

    z_feats = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
    pca = joblib.load(f"ttsai/weights/pca.joblib")
    pca_feats = pca.transform(z_feats.reshape(1, -1))
    sc = joblib.load(f"ttsai/weights/pca_sp_emb_scaler.joblib")
    pca_feats = sc.transform(pca_feats)[0]
    pca_feats = torch.tensor(pca_feats)
    return pca_feats
"""
@torch.no_grad()
def gen_waveform(wavenet_model, out_feats):
    gen_wav = wavenet_model.inference(c=out_feats, x=None)
    gen_wav = gen_wav.max(1)[0].float().cpu().numpy().reshape(-1)
    return gen_wav

device = torch.device("cpu")

acoustic_model = Tacotron2().to(device)
wavenet_model = ParallelWaveGANGenerator().to(device)

with open("ttsai/weights/jsut_tacotron_std.joblib", mode="rb") as f:
    checkpoint = joblib.load(f)#.to(device)
    acoustic_model.load_state_dict(checkpoint)#.to(device)
    acoustic_model.eval()

#parallelwaveganの重みを読み込む
checkpoint = torch.load(f"ttsai/weights/jsut_parallelwavegan.pkl", map_location=device)
wavenet_model.load_state_dict(checkpoint['model']['generator'])
# weight normalization は推論時には不要なため除く
wavenet_model.remove_weight_norm()
wavenet_model = wavenet_model.eval()

def tts(pca_feats=torch.tensor(np.load('ttsai/weights/jsut-feats.npy')), text=None):
    if text==None:
        text = input()
    labels = pyopenjtalk.extract_fullcontext(text)
    in_feats = text_to_sequence(pp_symbols(labels))
    in_feats = torch.tensor(in_feats, dtype=torch.long).to(device)
    with torch.no_grad():
        out_feats, out_feats_fine, stop_flags, alignment = acoustic_model.inference(in_feats, pca_feats)
    out_feats = out_feats_fine
    gen_wav = gen_waveform(wavenet_model, out_feats)
    
    return gen_wav