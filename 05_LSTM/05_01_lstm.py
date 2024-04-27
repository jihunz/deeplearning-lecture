import numpy as np
import json
import re
import string

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, losses

VOCAB_SIZE = 10000
MAX_LEN = 200
EMBEDDING_DIM = 100
N_UNITS = 128
VALIDATION_SPLIT = 0.2
SEED = 42
LOAD_MODEL = False
BATCH_SIZE = 32
EPOCHS = 25

# 전체 데이터셋 로드
with open("kaggle/hugodarwood/epirecipes/full_format_recipes.json") as json_data:
    recipe_data = json.load(json_data)

# 데이터셋 필터링
filtered_data = [
    "Recipe for " + x["title"] + " | " + " ".join(x["directions"])
    for x in recipe_data
    if "title" in x
    and x["title"] is not None
    and "directions" in x
    and x["directions"] is not None
]

# 레시피 개수 확인
n_recipes = len(filtered_data)
print(f"{n_recipes}개 레시피 로드")

example = filtered_data[9]
print(example)

# 구두점을 분리하여 별도의 '단어'로 취급합니다.
def pad_punctuation(s):
    s = re.sub(f"([{string.punctuation}])", r" \1 ", s)
    s = re.sub(" +", " ", s)
    return s


text_data = [pad_punctuation(x) for x in filtered_data]

# 레시피 샘플 출력
example_data = text_data[9]
example_data

# 텐서플로 데이터셋으로 변환하기
text_ds = (tf.data.Dataset.from_tensor_slices(text_data).batch(BATCH_SIZE).shuffle(1000))

# 벡터화 층 만들기
vectorize_layer = layers.TextVectorization(
    standardize="lower",
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_LEN + 1,
)

# 훈련 세트에 층 적용
vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()

# 토큰:단어 매핑 샘플 출력하기
for i, word in enumerate(vocab[:10]):
    print(f"{i}: {word}")

# 동일 샘플을 정수로 변환하여 출력하기
example_tokenised = vectorize_layer(example_data)
print(example_tokenised.numpy())

# 레시피와 한 단어 이동한 동일 텍스트로 훈련 세트를 만듭니다.
def prepare_inputs(text):
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y


train_ds = text_ds.map(prepare_inputs)

inputs = layers.Input(shape=(None,), dtype="int32")
x = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM)(inputs)
x = layers.LSTM(N_UNITS, return_sequences=True)(x)
outputs = layers.Dense(VOCAB_SIZE, activation="softmax")(x)
lstm = models.Model(inputs, outputs)
lstm.summary()

if LOAD_MODEL:
    # model.load_weights('./models/model')
    lstm = models.load_model("./models/lstm", compile=False)

loss_fn = losses.SparseCategoricalCrossentropy()
lstm.compile("adam", loss_fn)

# TextGenerator 체크포인트 만들기
class TextGenerator(callbacks.Callback):
    def __init__(self, index_to_word, top_k=10):
        self.index_to_word = index_to_word
        self.word_to_index = {
            word: index for index, word in enumerate(index_to_word)
        }

    def sample_from(self, probs, temperature):
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs), probs

    def generate(self, start_prompt, max_tokens, temperature):
        start_tokens = [
            self.word_to_index.get(x, 1) for x in start_prompt.split()
        ]
        sample_token = None
        info = []
        while len(start_tokens) < max_tokens and sample_token != 0:
            x = np.array([start_tokens])
            y = self.model.predict(x, verbose=0)
            sample_token, probs = self.sample_from(y[0][-1], temperature)
            info.append({"prompt": start_prompt, "word_probs": probs})
            start_tokens.append(sample_token)
            start_prompt = start_prompt + " " + self.index_to_word[sample_token]
        print(f"\n생성된 텍스트:\n{start_prompt}\n")
        return info

    def on_epoch_end(self, epoch, logs=None):
        self.generate("recipe for", max_tokens=100, temperature=1.0)

# 모델 저장 체크포인트 만들기
model_checkpoint_callback = callbacks.ModelCheckpoint(
    # filepath="./checkpoint/checkpoint.ckpt",
    filepath="./checkpoint/checkpoint.weights.h5",
    save_weights_only=True,
    save_freq="epoch",
    verbose=0,
)

tensorboard_callback = callbacks.TensorBoard(log_dir="./logs")

# 시작 프롬프트 토큰화
text_generator = TextGenerator(vocab)

lstm.fit(train_ds, epochs=EPOCHS, callbacks=[model_checkpoint_callback, tensorboard_callback, text_generator],
)

# 최종 모델 저장
lstm.save("./models/lstm")

def print_probs(info, vocab, top_k=5):
    for i in info:
        print(f"\n프롬프트: {i['prompt']}")
        word_probs = i["word_probs"]
        p_sorted = np.sort(word_probs)[::-1][:top_k]
        i_sorted = np.argsort(word_probs)[::-1][:top_k]
        for p, i in zip(p_sorted, i_sorted):
            print(f"{vocab[i]}:   \t{np.round(100*p,2)}%")
        print("--------\n")

info = text_generator.generate("recipe for roasted vegetables | chop 1 /", max_tokens=10, temperature=1.0)
print_probs(info, vocab)

info = text_generator.generate("recipe for roasted vegetables | chop 1 /", max_tokens=10, temperature=0.2)
print_probs(info, vocab)

info = text_generator.generate("recipe for chocolate ice cream |", max_tokens=7, temperature=1.0)
print_probs(info, vocab)

info = text_generator.generate("recipe for chocolate ice cream |", max_tokens=7, temperature=0.2)
print_probs(info, vocab)

