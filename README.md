# KoGPT2 자기소개서 도우미

  
## 1. 소개
[SKT-AI의 KoGPT2](https://github.com/SKT-AI/KoGPT2)와 pytorch를 이용해 소설을 생성하는 
[NarrativeKoGPT2](https://github.com/shbictai/narrativeKoGPT2)를 참고하여 작성하였습니다.

KoGPT2모델에, 자기소개서 10,000편 이상을 이용해 Fine-Tuning 하였습니다.

간단한 GUI도 구현해 놨으며,

[EXE Program Download](https://drive.google.com/file/d/1sMS5v-ZR8JDWID6XbRWPQDt-u9izQXke/view?usp=sharing)
링크에서 zip파일을 다운로드 및 압축해제하고, 내부 exe파일을 실행하시면 됩니다.

(다만 일부 파이썬이 설치되지 않은 환경에서 정상 실행이 불가능한 버그가 존재합니다.
시간이 되는대로 고쳐두겠습니다.)

기타 프로젝트는 완성해놓고 이제야 깃허브에 업로드 중입니다.
시간이 나는대로 코드도 좀 더 보기 쉽게 정리하고, 설명도 달아놓도록하겠습니다.

### 1. 데이터
인터넷에 공개된 자기소개서 10,000여편

(크롤링 관련 코드는 공개하지 않겠습니다.; 다만 selenium과 BeautifulSoup을 이용하였습니다.)

<!--
#### 1.1 데이터 샘플
1024 토큰에서 시작과 끝 토큰 갯수 2개를 제외한 1022개의 토큰으로 구성된 학습 데이터 사용. 아래 데이터를 NovelDataSet을 통해 학습시 처리 
```
하지만 그는 나이가 많아 보였고 무엇보다 새하얗고 기다란 턱수염을 하고 있는 자였다.  “고 녀석 참,”라며 노인(유령)이 큰소리로 말했다. “넌 이제 등골이 오싹해질 게다, 왜냐면 넌 곧 죽게 될 테니까.” “놀고 있네,”라며 청년(주인공)이 말했다. “그런 소리는 내가 죽고 나서나 해야 하는 거 아냐.” “네 놈을 뭉개주지.”라며 그 친구(노인 유령)가 말했다. “어이, 어이, 그렇게 큰 소리 치지 말라고. 힘은 내가 당신보다 더 있을 테니 말이야.” “어디 한 번 겨뤄볼까,”라며 그 노인(유령)이 말했다. “만약 네가 나보다 쌔면, 놓아주지… 어디, 겨뤄보자고.” 그런 다음 노인(유령)은 청년(주인공)을 데리고 껌껌한 통로들을 지나 대장간으로 데리고 갔다.  노인(유령)이 도끼 한 자루를 집어 들더니 ‘모루’(=받침으로 사용하는 쇳덩이. 사진링크 ▶ https://goo.gl/vTy5cD )를 한 방에 내려쳐 땅에다 받아버렸다.  “놀고 있네, 난 그보다 더 잘할 자신 있다고.”라며 청년이 말하더니 다른 ‘모루’(받침으로 사용하는 쇳덩이)로 갔다.  노인은 청년 옆에서 직접 보고 싶었다.  그러다 노인의 긴 턱수염이 모루 위에 놓이고 말았다.  그때 청년이 도끼를 집어 들더니 한 방에 모루를 쪼갰는데, 그 바람에 노인의 턱수염이 쪼개진(찢어진) 모루에 끼고 말았다.  “내가 도리어 당신을 잡았군,”라며 청년이 말했다. “자 누가 죽나 보자고.” 그런 다음 청년은 철봉(철로 된 막대기)을 부여잡더니 그걸로 노인(유령)을 노인이 신음 소리를 내며 “막대한 재물(=돈)을 줄 테니” 제발 그만하라고 애걸복걸 할 때까지 때렸다.  그러자 청년은 철봉을 던져버리고 노인(유령)을 놓아주었다.  노인(유령)은 청년을 데리고 다시 ‘성’으로 돌아와 성 지하실에 있던 세 상자의 궤짝들을 보여주었다. 거기엔 금은보화가 가득했다.  “이 중,”라며 노인(유령)이 말했다. “이 중 하나는 가난한 사람들에게 나누어줄 것이며, 다른 하나는 왕깨 드릴 것이며, 세 번째 궤짝에 든 게 자네의 것이네.” 그러는 사이에 밤 12시를 알리는 종이 울리자, 유령(노인)이 흔적도 없이 사라져다.  그리하여 청년은 다시 혼자 어둠 속에 남게 되었다.  “다행히 출구를 찾을 순 있겠어,”라며 청년이 말했다. 어둠이 눈에 익자 곧 방으로 가는 출구를 찾아 청년은 다시 불 옆에 와서 잠이 들었다.  다음날 아침 왕이 와 말했다.  “그래 이젠 좀 등골이 오걸 배웠는감?” “아뇨,”라고 청년(주인공)이 말했다. “그게 도대체 뭔데요? 제 죽은 사촌동생도 여기 왔다 갔고, 턱수염을 기른 노인네도 다녀갔지만 지하창고에 가득 든 금은보화만 보여주고 만 걸요. 아주도 제게 등골이 오싹한 게 뭔지 알려주지 않았어요.” “자자,”라면 왕이 말했다. “성으로 가서 내 딸과 결혼식을 올리게.” “성은이 망극하나이다.”라며 청년이 말했다. “하지만 등골이 오싹한 걸 어째 좀 배웠음 하는 마음만은 여전해요.” 그런 다음 노인(유령)이 준 금은보화가 운반되었고, 그리고 결혼식도 성대하게 잘 열렸다.  하지만 젊은 왕(주인공)이 자신의 왕비를 무척 사랑했고 결혼생활도 행복 그 자체였지만, 젊은 왕은 여전히 시간만 나면 중얼거렸다.  “등골이 오싹한 걸 배워야 하는데… 등골이 오싹한 걸 배워야 하는데.” 그래서 마침내 왕비도 짜증만땅이 되고 말았다.  그녀의 시녀(몸종) 하나가 말했다.  “제게 좋은 방법이 있어요. 등골이 오싹한 게 뭔지 왕께서도 곧 배우시게 되실 거예요.” 시녀(몸종. 왕비의 시녀)는 정원에 졸졸 흐르고 있는 개울가로 가 ‘모샘치’(잉어과에 속하는 작은 물고기. 낚싯밥으로 사용됨. 사진링크 ? https://goo.gl/vNQKwC )를 양동이에 한 가득 담아와 왕비께 건넸다.  그날 밤 젊은 왕(주인공)이 잠을 자고 있는데, 왕비(아내)가 왕의 옷을 벗기곤 시녀에게 건네받았던 ‘양동이에 한 가득 든 모샘치(작은 물고기)와 물’을 왕에게 쏟아 부었다.  그 바람에 작은 물고기들이 왕의 몸 위 여기저기에서 버둥버둥 거렸다.
```
#### 1.2 학습 데이터
위 데이터 샘플들을 한줄 씩 읽으면서 시작과 끝에 `BOS`, `EOS` 토큰 추가.
```python
class NovelDataset(Dataset):
  ...생략...
    while True:
      line = file.readline()
      if not line:
        break
      toeknized_line = tokenizer(line[:-1])
      index_of_words = [vocab[vocab.bos_token],] + vocab[toeknized_line]+ [vocab[vocab.eos_token]]

      self.data.append(index_of_words)

    file.close()
  ...생략...
```
### 2. 텍스트 생성
NovelGenerator 이용시 sampling.py에서 각 샘플링 방법 이용 가능. 잦은 중복으로 인해 Top-P 샘플링이 가장 적절하다고 판단. 다른 샘플링 방법을 추가하여 사용 가능.

1. Random Sampling: 모든 가능성에 대해 랜덤으로 샘플링
2. Top-K Sampling: 확률이 높은 순서에 따라 k개의 토큰을 뽑아 샘플링
3. Top-P Sampling: 누적분포를 계산하고, 누적 분포 함수가 $p$ 값을 초과하면 샘플링

### 3. 문장 분리
[likejazz/korean-sentence-splitter](https://github.com/likejazz/korean-sentence-splitter)의 문장 분리기를 사용.
```python
import kss

s = "회사 동료 분들과 다녀왔는데 분위기도 좋고 음식도 맛있었어요 다만, 강남 토끼정이 강남 쉑쉑버거 골목길로 쭉 올라가야 하는데 다들 쉑쉑버거의 유혹에 넘어갈 뻔 했답니다 강남역 맛집 토끼정의 외부 모습."
for sent in kss.split_sentences(s):
    print(sent)
```
### 언어 및 라이브러리
python, pytorch
### 학습
- Colab GPU: Tesla P100
- 학습 소요기간: 약 12시간.


## 2. 사용 방법
1) 모델학습: NarriveKoGPT2.ipynb
2) 문장생성: NovelGenerator.ipynb

###  colab 학습 방법 
1. Colab 디렉토리에 narrativeKoGPT2 복사
2. 사용방법에 명시된 파일을 실행. 
3. 경로 오류 발생하는 경우, `import` 경로 맞춰줄것 
4. 사용 목적에 따라 NarriveKoGPT2.ipynb 또는 NovelGenerator.ipynb 실행
 
## 3. 문장 생성(top-p 샘플링 이용, p=0.85)
  
**1. 문장 생성**
```
문장입력: 사람들이 하늘을 보니 저 멀리서
사람들이 하늘을 보니 저멀리서  굴뚝새가 고요히 바라보는 게 마치 할머니  집 입구 쪽이나 동굴 밖까지 날아가는 것만 같아요.
그것도 사람들이지만 달빛 한 점 눈딱 뜨고 볼 수 있었지요.
작은 동산엔 빛나는 달 한 점 없는 아름다운 별 하나가 서 있었는데, 이게 어디 여행자인지 큰 배낭을 메고서도 매일 하늘 아래 모여 있었어요.
어쩌다 태양 한 점만 남겨도 산딸기를 구별해내 건널 수 있었던 여인숙에 여행객이 묵고 싶어서 안달이 난 거지 뭐예요.
배롱나무 수염도 짜고 베개도 벨 수도 없었던 거지 뭐예요.
배낭엔 달콤한 연주가 연주가 따로 있었고요,
부엌용 파이프들도 제각각 들어 있었어요.
소녀는 손가락을 들어 빙빙 돌더니 별들의 탱고 리듬에 맞춰 노래했지요.
“사실은 마법이나 이런 건 어쩔 수 없어, 적어도 영험하진 못한 달엔 보석도 많으니까 말이야, 그치?” 창문 없는 뾰족지붕집 같은 공간이 애처로웠지요.
```
**2. 문장 생성**
```
문장입력: 지금보다 더 어리고 쉽게 상처받던 시절
지금보다 더 어리고 쉽게 상처받던 시절, 노인 네 분이 좀 더 단정하고 숭고하게 옷을 차려입고 노루들이 잠에서 깨어나 우셨어요.
그럼 각자가 휴식을 취하는 사이, 촛불 하나 하나가 수북이 위 아래로 늘어서 ‘불이 있는 힘껏!’을 내질렀어요.
‘오호!’라고 불기 시작한 촛불은 세상으로 뻗어졌고, 천만다행하게도 자비는 다했지요.
세 오빠들이 포도주를 드시고 푹  잔 다음, 정말 사악한 사냥꾼들이 신부를 때려죽였어요.
왜냐면 황야에서 그녀(소년과 소녀)가 살아 있었거든요.
나무 아래에 앉아서 남루한 말들을 돌본 사냥꾼이 말했어요.
“하나님을 저버리시오, 사랑하는 소년을 죽이다니요.” 왕이 옆에 앉으며 여쭈었어요.
“하느님이 세 마녀를 쓰시어 왕국을 차 버리셨소. 거기 들어가셔서 지금 뭐하고 계신지 잘 보시구려.” 그녀가 다가가 머리를  숙여 보여드리니 왕은 놀라며 소리쳤어요.
```

**3. 문장생성**
```
문장 입력: 와 너무 졸린걸?
와 너무 졸린걸?
음흠  자기는 긴장하며 첫 글자로 뭘 표현해야 하는 거야? 남자라서 할 말은 많잖아
휴- 흠 글쎄 자기 어떻게 그 옛날의 기억이... 미안하다 하지 않았었잖아...
아무래도 생각났다  - 눈 이 이 피곤하구... 약간이나 ... 다음... 피눈물날 때 일어나야 해... 막 넘어지고 발은 허공으로 갈아버리고, 나의 목뼈, 창, 등에 발라지며 뒤뚱뒤뚱 움직이는 목뼈.... 다음.... 자꾸 그게 싫어지는 것이 서럽다...
사람이 정신 나간 놈인지 한심한놈인지 정말 나 스스로 멍청하다...
말도 없이 날아가 버려.... 혼돈에서 나온 것이다. ...
자꾸 떠나간 자신의 모습 그리고 남자로서의 생각을 하지만 아무리 당신이 정 대못 때려 놓고 있으라고는 생각하지만 그래도 사랑하구 있어...
사랑한다면 애지중지 하 는 것 이 바로 가장 무서운 사랑이 아닐지...
저런 감정을 무슨, 이유없이 빼앗아 갔을까?
```

**3. 문장생성**
```
문장 입력: 야 사과가 배 좋아하는것 같아
야 사과가 배 좋아하는것 같아도 솔직히 할말은 없을 거야 그래서 무서워 막 가서 색포청 입을 먹고는 싶다고 한다.
알았으면 제대로 오렴 부탁하마; 
그려 그럼 좀 그만 마실래 미안. 너 내가 실수로 잊었잖아 그리워서 사과 씹었어?
" 술마셨을까...사과 뜯고 이짜<unk> 하며 씹어먹는다<und> 조용히 있어, 부드럽게 안 먹게!!. 지워어  떡이고 빨면 좋겠지  그래 그러지마 지칠려면 마마지 박아서..." 이라고 나와 잇는다. !
안돼요?
좀 비켜서 있으세요.
야 그러니까 내가 대체 뭐라고 그랬냐구요,
마마.
" 테리우스가 재빨리 쳐다본다. 짜증스러워서 치마를 들어다 부들부들 떨린다. 아무렴 내게 미움 받지 말고 집에 와봐요."태준이이 쪽! 그러지 마.
아무렴 또 실수를 저질러버릴지 모르니까 그냥 있다가 들어가!
```
-->
## 2. 라이센스
`modified MIT`라이센스
## 3. 참조
- [SKT-AI/KoGPT2](https://github.com/SKT-AI/KoGPT2)    
- [likejazz/korean-sentence-splitter](https://github.com/likejazz/korean-sentence-splitter)