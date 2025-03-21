import torch
from .decoder import WavTokenizer
from .encoder.utils import convert_audio
import torchaudio
from huggingface_hub import hf_hub_download

class CodecWavTokenizer(torch.nn.Module):
    def __init__(self, model=None, config_path=None, ckpt_path=None, device='cpu'):
        super().__init__()
        self.model_specs = CodecWavTokenizer.get_model_specs()
        if model is not None:
            ckpt_path, config_path = self.get_model(model)
        self.model = WavTokenizer.from_pretrained0802(config_path, ckpt_path)
        self.model.to(device)
        self.device = device
        

    @staticmethod
    def get_model_specs():
        spec = {'small_40_speech': ['https://huggingface.co/novateur/WavTokenizer/blob/main/WavTokenizer_small_600_24k_4096.ckpt',
                             'https://huggingface.co/novateur/WavTokenizer/blob/main/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml'],
         'small_75_speech': ['https://huggingface.co/novateur/WavTokenizer/blob/main/WavTokenizer_small_320_24k_4096.ckpt',
                             'https://huggingface.co/novateur/WavTokenizer/blob/main/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml'],
         'medium_75_speech': ['https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/blob/main/wavtokenizer_medium_speech_320_24k_v2.ckpt',
                              'https://huggingface.co/novateur/WavTokenizer-medium-speech-75token/blob/main/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml'],
         'medium_75_audio': ['https://huggingface.co/novateur/WavTokenizer-medium-music-audio-75token/blob/main/wavtokenizer_medium_music_audio_320_24k_v2.ckpt',
                             'https://huggingface.co/novateur/WavTokenizer-medium-music-audio-75token/blob/main/wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml'],
         'large_40_audio': ['https://huggingface.co/novateur/WavTokenizer-large-unify-40token/blob/main/wavtokenizer_large_unify_600_24k.ckpt',None],
         'large_75_speech': ['https://huggingface.co/novateur/WavTokenizer-large-speech-75token/blob/main/wavtokenizer_large_speech_320_v2.ckpt',None]
        }
        return spec

    @staticmethod 
    def list_available_models():
        return list(CodecWavTokenizer.get_model_specs().keys())

    def get_model(self, model):
        def download_hf_path(x):
            repo_id = '/'.join(x.split('/')[3:5])
            path_in_repo = '/'.join(x.split('/')[7:])
            return hf_hub_download(repo_id=repo_id, filename=path_in_repo)

        ckpt_path = download_hf_path(self.model_specs[model][0])
        config_path = download_hf_path(self.model_specs[model][1])
        with open(config_path, 'r') as f:
            cfg_txt = f.read()
        cfg_txt = cfg_txt.replace(': decoder.', 'audioken.tokenizers.WavTokenizer.decoder.')
        with open(config_path, 'w') as f:
            f.write(cfg_txt)
        return ckpt_path, config_path

    def tokenize_wav(self, filename):
        wav, sr = torchaudio.load(filename)
        wav = convert_audio(wav, sr, 24000, 1)
        wav = wav.to(self.device)
        return self.tokenize_array(wav)

    def tokenize_array(self, x):
        bandwidth_id = torch.tensor([0])
        features, discrete_code = self.model.encode_infer(x, bandwidth_id=bandwidth_id)
        return {'z': features,
                'q': discrete_code}

    def decode_tokens(self, tokens):
        bandwidth_id = torch.tensor([0])
        return self.model.decode(tokens['z'], bandwidth_id=bandwidth_id)


