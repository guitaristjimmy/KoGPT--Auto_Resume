import os
import torch
from torch.utils.data import DataLoader     # 데이터로더
from gluonnlp.data import SentencepieceTokenizer

from kogpt2.utils import get_tokenizer
from kogpt2.utils import download, tokenizer
from model.torch_gpt2 import GPT2Config, GPT2LMHeadModel
from util.data import QA_Dataset, NovelDataset
import gluonnlp
import time
import sys
import sampling


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 1"

pytorch_kogpt2 = {
    'url':
    'https://kobert.blob.core.windows.net/models/kogpt2/pytorch/pytorch_kogpt2_676e9bcfa7.params',
    'fname': 'pytorch_kogpt2_676e9bcfa7.params',
    'chksum': '676e9bcfa7'
}

kogpt2_config = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-06,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "vocab_size": 50000,
    'pad_token': '<pad>'
}

# ctx='cpu'
ctx = 'cuda'
cachedir = '~/kogpt2/'
save_path = '/home/kskim/kogpt2/ckpts/'
epochs = 100

# download model
model_info = pytorch_kogpt2
model_path = download(model_info['url'],
                       model_info['fname'],
                       model_info['chksum'],
                       cachedir=cachedir)

# download vocab
vocab_info = tokenizer
vocab_path = download(vocab_info['url'],
                       vocab_info['fname'],
                       vocab_info['chksum'],
                       cachedir=cachedir)

##############################################################################
device = torch.device(ctx)

# KoGPT-2 언어 모델 학습을 위한 GPT2LMHeadModel 선언
kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
# model_path로부터 다운로드 받은 내용을 load_state_dict으로 업로드
load_ckpts_flag = False
if os.path.isfile(save_path + 'narrativeKoGPT2_checkpoint_best.tar'):
    load_ckpts_flag = True
    checkpoint = torch.load(save_path + 'narrativeKoGPT2_checkpoint_best.tar', map_location=device)
    model_state = dict()
    for old_key in checkpoint['model_state_dict'].keys():
        new_key = '.'.join(old_key.split('.')[1:])
        model_state[new_key] = checkpoint['model_state_dict'][old_key]
    kogpt2model.load_state_dict(model_state)
    s_epoch = checkpoint['epoch']
    print('load ckpts model')
else:
    kogpt2model.load_state_dict(torch.load(model_path))
    s_epoch = 0
    print('load open pre-trained model')

kogpt2model.to(device)

# kogpt2model.eval()
# 추가로 학습하기 위해
kogpt2model.train()
vocab_b_obj = gluonnlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
                                                     mask_token=None,
                                                     sep_token=None,
                                                     cls_token=None,
                                                     unknown_token='<unk>',
                                                     padding_token='<pad>',
                                                     bos_token='<s>',
                                                     eos_token='</s>')
print('load vocab')

##############################################################################

tok_path = get_tokenizer()
model, vocab = kogpt2model, vocab_b_obj
model = torch.nn.DataParallel(model, output_device=0)

sentencepieceTokenizer = SentencepieceTokenizer(tok_path, num_best=8, alpha=0.0)

# data_file_path = '/media/kskim/nvme_data/ai_hub_QA.txt'
data_file_path = '/home/kskim/Codes/narrativeKoGPT2/data/train.txt'
batch_size = 3
# novel_dataset = QA_Dataset(data_file_path, vocab, sentencepieceTokenizer)
novel_dataset = NovelDataset(data_file_path, vocab, sentencepieceTokenizer)
novel_data_loader = DataLoader(novel_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=8)

learning_rate = 1e-6
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


pre_loss = 1e9

if load_ckpts_flag:
    pre_loss = checkpoint['loss'].item()
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('load ckpts optimizer state dict')

print('start epoch')
early_stop_cnt = 0
save_cnt = 0

data_len = len(novel_data_loader.dataset)

for epoch in range(s_epoch, epochs):
    count = 0
    for data in novel_data_loader:
        start_time = time.time()
        count += 1
        optimizer.zero_grad()

        data = torch.stack(data).to(device)    # list of Tensor로 구성되어 있기 때문에 list를 stack을 통해 변환해준다.

        data = data.transpose(1, 0)
        # y_pred = model(data)
        # loss = criterion(y_pred, data)
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        outputs = model(data, labels=data)
        loss, logits = outputs[:2]
        loss = loss.mean()
        output_txt = 'epoch : {e}\ttrain no : {c}/{t}\tloss :: {l}\ttime :: {time}'.format(
            e=str(epoch), c=str(count), t=str(data_len//batch_size), l=str(loss.item()),
            time=round((time.time()-start_time)*(data_len//batch_size-count), 2))
        print(output_txt, end='\r')
        loss.backward()
        optimizer.step()

    if loss.item() > pre_loss:
        early_stop_cnt += 1
    else:
        print(loss.item(), pre_loss)
        print('save torch model :: loss : ', loss.item())
        early_stop_cnt = 0
        pre_loss = loss.item()

        torch.save({
            'epoch': epoch,
            'train_no': count,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, save_path + 'narrativeKoGPT2_checkpoint_best.tar')

    if epoch % 30 == 0:
        print('{epoch} text gen :: ')
        model.eval()
        test_sent = '자기소개를 작성해주세요.'
        toked = sentencepieceTokenizer(test_sent)
        g_count = 0
        output_size = 50
        sent = ''
        while 1:
            input_ids = torch.tensor([vocab[vocab.bos_token], ] + vocab[toked] + [vocab[vocab.eos_token], ]).unsqueeze(
                0)
            predicts = model(input_ids)
            pred = predicts[0]

            last_pred = pred.squeeze()[-1]
            # top_p 샘플링 방법
            # sampling.py를 통해 random, top-k, top-p 선택 가능.
            _gen = sampling.top_p(last_pred, vocab, 0.8)
            # _gen = sampling.top_k(last_pred, vocab, 100)

            if g_count > output_size or _gen == '</s>':
                sent += _gen.replace('▁', ' ')
                toked = sentencepieceTokenizer(sent)
                g_count = 0
                break

            sent += _gen.replace('▁', ' ')
            toked = sentencepieceTokenizer(sent)
            g_count += 1
        print(sent)

        model.train()

        print(loss.item(), pre_loss)
        print('save torch model :: loss : ', loss.item())
        early_stop_cnt = 0
        pre_loss = loss.item()

        torch.save({
            'epoch': epoch,
            'train_no': count,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, save_path + 'narrativeKoGPT2_checkpoint.tar')

    if early_stop_cnt > 20:
        print('early stop...')
        sys.exit(0)
