# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import qdarkstyle
from kogpt2.utils import get_tokenizer
import torch
import gluonnlp
from gluonnlp.data import SentencepieceTokenizer
from model.torch_gpt2 import GPT2Config, GPT2LMHeadModel
from functools import partial
import numpy as np
import kss

### 1. koGPT2 Config
ctx = 'cpu'#'cuda' #'cpu' #학습 Device CPU or GPU. colab의 경우 GPU 사용
cachedir = '~/kogpt2/'      # KoGPT-2 모델 다운로드 경로
model_path = './gen_m.tar'
load_path_moli_sim = 'C:\\Users\\K\\Desktop\\I_SW\\Python_Note\\gpt-2\\model\\narrativeKoGPT2_checkpoint_best.tar'
vocab_path = './vocab.spiece'

#use_cuda = True # Colab내 GPU 사용을 위한 값

pytorch_kogpt2 = {
    'url':
    'https://kobert.blob.core.windows.net/models/kogpt2/pytorch/pytorch_kogpt2_676e9bcfa7.params',
    'fname': 'pytorch_kogpt2_676e9bcfa7.params',
    'chksum': '676e9bcfa7'
}

kogpt2_config = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "vocab_size": 50000
}

class Ui_MainWindow(object):
    def __init__(self):
        self.input_ids = []

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(940, 535)
        self.load_model()
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.central_layout = QGridLayout()
        self.centralwidget.setLayout(self.central_layout)

        self.text_group = QtWidgets.QGroupBox(self.centralwidget)
        self.text_group.setGeometry(QtCore.QRect(10, 5, 920, 435))
        self.text_group.setObjectName("text_group")
        self.text_layout = QGridLayout()
        self.text_group.setLayout(self.text_layout)

        self.gen_btn = QtWidgets.QPushButton(self.text_group)
        self.gen_btn.setObjectName("gen_btn")
        self.gen_btn.clicked.connect(self.gen_text)
        self.gen_btn.setDisabled(True)
        self.text_layout.addWidget(self.gen_btn, 0, 0, 1, 1)

        self.edit_btn = QtWidgets.QPushButton(self.text_group)
        self.edit_btn.setObjectName("edit_btn")
        self.edit_btn.clicked.connect(self.edit)
        self.text_layout.addWidget(self.edit_btn, 0, 1, 1, 1)

        self.text_edit = QtWidgets.QTextEdit(self.text_group)
        self.text_edit.setObjectName("text_edit")
        self.text_edit.setDisabled(True)
        self.text_layout.addWidget(self.text_edit, 1, 0, 10, 4)

        self.central_layout.addWidget(self.text_group, 0, 0, 10, 10)

        self.btn_list = []
        for i in range(10):
            self.btn_list.append(QtWidgets.QPushButton(self.centralwidget))
            self.btn_list[-1].setGeometry(QtCore.QRect(20+i*90, 450, 75, 50))
            self.btn_list[-1].setObjectName("btn_{}".format(i))
            self.btn_list[-1].clicked.connect(partial(self.next_gen, i))
            self.btn_list[-1].setDisabled(True)
            self.btn_list[-1].setText(' \n ')
            self.central_layout.addWidget(self.btn_list[-1], 11, i, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        # menu bar -----------------------------------------------------------------------------------------------------
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 922, 20))
        self.menubar.setObjectName("menubar")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.actionHelp = QtWidgets.QAction(MainWindow)
        self.actionHelp.setShortcutVisibleInContextMenu(True)
        self.actionHelp.setObjectName("actionHelp")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setShortcutVisibleInContextMenu(True)
        self.actionSave.setObjectName("actionSave")
        self.menuHelp.addAction(self.actionHelp)
        self.menuFile.addAction(self.actionSave)
        self.menubar.addAction(self.menuHelp.menuAction())
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.text_group.setTitle(_translate("MainWindow", "Text"))
        self.gen_btn.setText(_translate("MainWindow", "Start Generation"))
        self.edit_btn.setText(_translate("MainWindow", "Edit Text"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionHelp.setText(_translate("MainWindow", "Help"))
        self.actionHelp.setShortcut(_translate("MainWindow", "F1"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionSave.setShortcut(_translate("MainWindow", "Ctrl+S"))

    def edit(self):
        self.edit_btn.setDisabled(True)
        self.text_edit.setDisabled(False)
        self.gen_btn.setDisabled(False)

    def gen_text(self):
        self.gen_btn.setDisabled(True)
        self.text_edit.setDisabled(True)
        self.edit_btn.setDisabled(False)

        sentences = self.text_edit.toPlainText()
        for sent in kss.split_sentences(sentences):
            toked = self.tok(sent)
            self.input_ids += [self.vocab[self.vocab.bos_token], ] + \
                              self.vocab[toked] + \
                              [self.vocab[self.vocab.eos_token], ]

        self.run()

    def load_model(self):

        ### 3. 체크포인트 및 디바이스 설정
        # Device 설정
        self.device = torch.device(ctx)
        # 저장한 Checkpoint 불러오기
        checkpoint = torch.load(model_path, map_location=self.device)

        # KoGPT-2 언어 모델 학습을 위한 GPT2LMHeadModel 선언
        kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
        model_state_dict = {'.'.join(key.split('.')[1:]): checkpoint['model_state_dict'][key] for key in checkpoint['model_state_dict'].keys()}
        kogpt2model.load_state_dict(model_state_dict)

        kogpt2model.eval()

        vocab_b_obj = gluonnlp.vocab.BERTVocab.from_sentencepiece(vocab_path,
                                                             mask_token=None,
                                                             sep_token=None,
                                                             cls_token=None,
                                                             unknown_token='<unk>',
                                                             padding_token='<pad>',
                                                             bos_token='<s>',
                                                             eos_token='</s>')
        ### 4. Tokenizer
        tok_path = get_tokenizer()
        self.tok = SentencepieceTokenizer(tok_path, alpha=0.0, num_best=0)
        self.model, self.vocab = kogpt2model, vocab_b_obj

    def top_k(self, predict, k=10):
        # topk 중 랜덤으로 선택된 값을 반환.
        gen = []
        print(np.shape(predict))
        probs, indexs = torch.topk(predict, k=k, dim=-1)
        # probs = probs.squeeze().tolist()[-1]
        # indexs = indexs.squeeze().tolist()[-1]
        probs = probs.tolist()
        indexs = indexs.tolist()
        print('indexs :: ', indexs)

        for i in range(len(indexs)):
            gen.append((self.vocab.to_tokens(indexs[i]), probs[i]))

        return gen

    def run(self):
        # sent = self.text_edit.toPlainText()
        # toked = self.tok(sent)
        # input_ids = torch.tensor([self.vocab[self.vocab.bos_token], ] +
        #                          self.vocab[toked] +
        #                          [self.vocab[self.vocab.eos_token], ]).unsqueeze(0).to(self.device)
        while True:
            if len(self.input_ids) >= 1024:
                print('del')
                del self.input_ids[0]
            else:
                break
        print('input ids :: ', np.shape(self.input_ids))
        predicts = self.model(torch.tensor(self.input_ids).unsqueeze(0).to(self.device))
        pred = predicts[0].squeeze()[-1]

        k_list = self.top_k(pred)
        # for idx in  range(len(self.btn_list)):
        #     self.btn_list[idx].setDisabled(True)

        for idx, k in enumerate(k_list):
            self.btn_list[idx].setText(f'{k[0]}\n({round(k[1], 2)})')
            self.btn_list[idx].setDisabled(False)

    def next_gen(self, btn_idx):
        selected_word = self.btn_list[btn_idx].text().split('\n')[0]
        self.text_edit.setText(self.text_edit.toPlainText() + selected_word)
        if selected_word == '<s>':
            self.input_ids += [self.vocab[self.vocab.bos_token], ]
        elif selected_word == '</s>':
            self.input_ids += [self.vocab[self.vocab.eos_token], ]
        else:
            self.input_ids += self.vocab[self.tok(selected_word)]

        self.run()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    dark_stylesheet = qdarkstyle.load_stylesheet_pyqt5()
    app.setStyleSheet(dark_stylesheet)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
