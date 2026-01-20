import os
import re
import sys
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import io
import json
import torchaudio
import wave
from pathlib import Path
print('Starting...')
import shutil
import time


import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import subprocess
import soundfile as sf
import ChatTTS
import datetime
from dotenv import load_dotenv
load_dotenv()

import logging
from logging.handlers import RotatingFileHandler

from random import random
from modelscope import snapshot_download
import numpy as np
import threading
from uilib.cfg import WEB_ADDRESS, SPEAKER_DIR, LOGS_DIR, WAVS_DIR, MODEL_DIR, ROOT_DIR
from uilib import utils,VERSION
from ChatTTS.utils.gpu_utils import select_device
from uilib.utils import is_chinese_os,modelscope_status
merge_size=int(os.getenv('merge_size',10))
env_lang=os.getenv('lang','')
if env_lang=='zh':
    is_cn= True
elif env_lang=='en':
    is_cn=False
else:
    is_cn=is_chinese_os()
    
CHATTTS_DIR= MODEL_DIR+'/pzc163/chatTTS'


chat = ChatTTS.Chat()
device=os.getenv('device','default')
chat.load(source="custom",custom_path=CHATTTS_DIR, device=None if device=='default' else device,compile=True if os.getenv('compile','true').lower()!='false' else False)

for it in os.listdir('speaker'):
    if it.startswith('seed_') and not it.endswith('_emb-covert.pt'):
        

        rand_spk=torch.load(f'./speaker/{it}', map_location=select_device(4096) if device=='default' else torch.device(device))

        torch.save( chat._encode_spk_emb(rand_spk) ,f"{SPEAKER_DIR}/{it.replace('.pt','-covert.pt')}")