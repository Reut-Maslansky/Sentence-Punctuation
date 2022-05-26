import pandas as pd
import numpy as np
from tqdm import trange
import torch

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from seqeval.metrics import accuracy_score

with open('heb_wiki.raw', "r") as file:
    dataset = file.read().replace('\n', ' ')
    file.close()

dataset = dataset.split(' ')

i = 0
data = []
while i + 30 < len(dataset):
    if (i > 1000000):
        break
    split_sent = []
    for j in range(i, (i + 1) + 30):
        split_sent.append(dataset[j])
    i += 10
    data.append(split_sent)
print(data[0])

punc = {",": "comma", ".": "period", "?": "question_mark", ":": "colon", ";": "semicolon", "_": "dash", "-": "hyphen",
        ")": "right_round_bracket", "(": "left_round_bracket", "]": "right_square_bracket", "[": "left_square_bracket",
        "\\": "slash", "'": "apostrophe", "\"": "speech_mark"}

sent_taged = []
for count, sent in enumerate(data, start=1):
    for word in sent:
        tag = 'O'
        w = word
        if len(word) > 0 and word[0] in punc:
            if word[-1] in punc:
                tag = punc[str(word[-1])] + " " + punc[str(word[0])]
                w = word[1:-1]
            else:
                tag = punc[str(word[0])]
                w = word[1:]
        elif len(word) > 0 and word[-1] in punc:
            tag = punc[str(word[-1])]
            w = word[:-1]
        sent_taged.append(list([count, w, tag]))
print(sent_taged[0])
print('convert data 1')
my_df = pd.DataFrame(sent_taged)
my_df.to_csv('my_csv.csv', index=False, header=False)
data = my_df.fillna(method="ffill")
data.tail(100)


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s[1].values.tolist(),
                                                     s[2].values.tolist())]
        self.grouped = self.data.groupby(0).apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


getter = SentenceGetter(data)
sentences = [[word[0] for word in sentence] for sentence in getter.sentences]

labels = [[s[1] for s in sentence] for sentence in getter.sentences]

tag_values = list(set(data[2].values))
tag_values.append("PAD")
tag_index = {t: i for i, t in enumerate(tag_values)}

print('convert data 2')
MAX_LEN = 75
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

from transformers import BertTokenizerFast
from transformers import AdamW, BertForTokenClassification

tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')


def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        tokenized_sentence.extend(tokenized_word)

        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


tokenized_texts_and_labels = [
    tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(sentences, labels)
]
print('tokenized_text')
tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")

tags = pad_sequences([[tag_index.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag_index["PAD"], padding="post",
                     dtype="long", truncating="post")

attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

alephbert = BertForTokenClassification.from_pretrained(
    'onlplab/alephbert-base',
    num_labels=len(tag_index),
    output_attentions=False,
    output_hidden_states=False
)

alephbert.cuda()

FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(alephbert.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(alephbert.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = AdamW(alephbert.parameters(), lr=1e-5)

from transformers import get_linear_schedule_with_warmup

epochs = 3
max_grad_norm = 1.0

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

print('start training')
# Store the average loss after each epoch so we can plot them.
loss_values, validation_loss_values = [], []
for _ in trange(epochs, desc="Epoch"):
    # Put the model into training mode.
    alephbert.train()
    # Reset the total loss for this epoch.
    total_loss = 0

    # Training loop
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        outputs = alephbert(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        loss.backward()
        total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(parameters=alephbert.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    # After the completion of each training epoch, measure our performance on
    # our validation set.

    # Put the model into evaluation mode
    alephbert.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions, true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = alephbert(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
        # Move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

    eval_loss = eval_loss / len(valid_dataloader)
    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels
                  for l_i in l if tag_values[l_i] != "PAD"]
    print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
    print()

test_sentence = "בית ספר הוא מוסד חינוכי בבעלות פרטית או ממשלתית האחראי על פרק הלימוד הטרום אקדמי על פי רוב בית ספר הוא מבנה בו משתתפים תלמידים וסגל הוראה בשיעורים יומיים מנהל בית ספר אחראי בדרך כלל גם על ניהול פיזי וכלכלי של בית הספר וגם על הנהגתו החינוכית עם זאת ניתן למצוא לעיתים לצידו של המנהל החינוכי יועצים חינוכיים המסייעים לתלמידים לפתור בעיות שונות העלולות לפגוע בתפקוד האקדמי שלהם"
test_sentence_punc = ".בית ספר הוא מוסד חינוכי בבעלות פרטית או ממשלתית, האחראי על פרק הלימוד הטרום־אקדמי. על פי רוב, בית ספר הוא מבנה בו משתתפים תלמידים וסגל הוראה בשיעורים יומיים. מנהל בית ספר אחראי בדרך כלל גם על ניהול פיזי וכלכלי של בית הספר וגם על הנהגתו החינוכית. עם זאת, ניתן למצוא לעיתים לצידו של המנהל החינוכי יועצים חינוכיים, המסייעים לתלמידים לפתור בעיות שונות העלולות לפגוע בתפקוד האקדמי שלהם"

tokenized_sentence = tokenizer.encode(test_sentence)
input_ids = torch.tensor([tokenized_sentence]).cuda()

with torch.no_grad():
    output = alephbert(input_ids)
label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

# join bpe split tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
new_tokens, new_labels = [], []
for token, label_idx in zip(tokens, label_indices[0]):
    if token.startswith("##"):
        new_tokens[-1] = new_tokens[-1] + token[2:]
    else:
        new_labels.append(tag_values[label_idx])
        new_tokens.append(token)

for token, label in zip(new_tokens, new_labels):
    print("{}\t{}".format(label, token))

result = ""
val_list = list(punc.values())
key_list = list(punc.keys())
for token, label in zip(new_tokens, new_labels):
    result = result + token
    if label != 'O':
        result = result + key_list[val_list.index(label)]
    result = result + " "
print(result)

torch.save(alephbert, "punctuation_model.pth")
torch.save(tokenizer, "tokenizer.pth")
torch.save(tag_values, "tag_values.pth")
