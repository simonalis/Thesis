import pandas as pd
import numpy as np

from src import fragment_creation
from src.LoadData import load_dataset, train_base_path
from sklearn.metrics import classification_report
from transformers import BertModel, BertTokenizerFast, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
# optimizer from hugging face transformers
from torch.optim import Adam
import os
from torch import nn

from matplotlib import pyplot as plt
print('Loading model to GPU...')
device = torch.device('cuda')
print('GPU:', torch.cuda.get_device_name(0))
print('DONE.')

model_directory = train_base_path + 'history' # directory to save model history after every epoch
model_history_file_name = '/model_history.csv'
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
file_name = './saved_weights_512_1_full_chunked_bert.pt'

n_categories = 5
from collections import defaultdict
read_from_path = train_base_path + "/transformers_evaluation_data_small/"
size = 100
#tokenizer = AutoTokenizer.from_pretrained("roberta-base") #Tokenizer

# specify GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(str(torch.cuda.memory_allocated(device)/1000000 ) + 'M')

alephbert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")#('onlplab/alephbert-base')#('avichr/heBERT')
alephbert = BertModel.from_pretrained("bert-base-uncased", return_dict=False)

# if not finetuning - disable dropout
alephbert.eval()

def load_data_from_file(what_data_str, size):
  inputs = ""
  count = 0
  texts = []
  f = open(str(read_from_path) + 'model_' + str(size) + what_data_str , "r")
  while True :
    count += 1
    line = f.readline()
    if not line:# or count > 1000000:
         break
    texts.append(line)
  f.close()
  print(count)
  return texts
#https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html


class RobertoDataLoader:
    def __init__(self, tokenizer, training_file, labels, batch_size, max_length):
        self.tokenizer = tokenizer
        self.training_file = training_file
        self.batch_size = batch_size
        self.max_length = max_length
        self.texts = load_data_from_file(training_file, size)
        self.labels = labels
        self.len = len(labels)

    def data_len(self):
        return self.len

    def __iter__(self):
        # Open the training file
       # with open(self.training_file, 'r') as f:
            # Read the lines in the file
          #  lines = f.readlines()
          #print("111")

            a = len(self.texts)
            #a = 1000
            # Calculate the number of batches
            num_batches = a // self.batch_size
            #print("num_batches", num_batches)
            # Iterate over the number of batches
            for i in range(num_batches):
                # Get the batch start and end indices

                if i % 500 == 0 and not i == 0:
                    print('  Batch {:>5,}  of  {:>5,}.'.format(i, num_batches))
                batch_start = i * self.batch_size
                batch_end = (i+1) * self.batch_size
                # Get the batch lines
                batch_lines = self.texts[batch_start:batch_end]
                batch_labels = self.labels[batch_start:batch_end]
                # Tokenize the batch lines
                tokenized_batch = self.tokenizer.batch_encode_plus(
                    list(batch_lines), max_length=self.max_length, pad_to_max_length=True,
                    truncation=True)
                # Convert the tokenized batch to tensors
                input_ids = torch.tensor(tokenized_batch['input_ids'])
                attention_mask = torch.tensor(tokenized_batch['attention_mask'])
                # Yield the input_ids, attention_mask tensors as a tuple
                torch_labels = torch.tensor(batch_labels.tolist())
                targets = torch_labels.type(torch.LongTensor)
                yield input_ids, attention_mask, targets


class BertBinaryClassifier(nn.Module):
    def __init__(self, bert):
        super(BertBinaryClassifier, self).__init__()

        self.bert = bert
        # print(bert)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, 5)
        #self.sigmoid = nn.Sigmoid()
       # self.softmax = nn.Softmax()

    def forward(self, tokens, masks=None):
        _, pooled_output = self.bert(tokens, attention_mask=masks, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
       # proba = self.softmax(linear_output)
        #proba = self.sigmoid(linear_output)
        return linear_output

def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets)

def create_model():#tokens_train):
    model = BertBinaryClassifier(alephbert)
    model = model.to(device)
    print(str(torch.cuda.memory_allocated(device)/1000000 ) + 'M')
    # tt = tokens_train['input_ids']
    # x = torch.tensor(tt[:3]).to(device)
    # y, pooled = model.bert(x)
    # x.shape, y.shape, pooled.shape
    #
    # y = model(x)
    # y.cpu().detach().numpy()

   # param_optimizer = list(model.softmax.named_parameters())
    #optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = Adam(model.parameters(), lr=3e-6)

    torch.cuda.empty_cache()
    #nn.L1Loss()
    #cross_entropy = nn.BCEWithLogitsLoss()# nn.BCELoss()
    epocs = 2
    return model, optimizer, epocs


# function to train the model
def train(model, train_dataloader):
    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(train_dataloader):

        # progress update after every 50 batches.
        # if step % 5 == 0 and not step == 0:
        #     print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu

        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        #labels = labels.view(-1, 1)
        #print('labels ', labels, '\nsize =', labels.size())
        # clear previously calculated gradients

        model.zero_grad()

        # get model predictions for the current batch -> here
        preds = model(sent_id, mask)
        #print('preds ', preds, '\nshape =', preds.size())
        # # compute the loss between actual and predicted values
       # labels = labels.to(torch.float32)
        loss = loss_fn(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()
        # backward pass to calculate the gradients
        model.zero_grad()
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / train_dataloader.data_len()

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    return avg_loss, total_preds


# function for evaluating the model
def evaluate(model, val_dataloader):
    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step, batch in enumerate(val_dataloader):

        # Progress update every 10 batches.
        #if step % 100 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            # elapsed = format_time(time.time() - t0)

            # Report progress.
            #print('  Batch {:>100,}  of  {:>100,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]  # to(device)

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)
           # labels = labels.view(-1, 1)
            # compute the validation loss between actual and predicted values
            #labels = labels.to(torch.float32)
            loss = loss_fn(preds, labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / val_dataloader.data_len()

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds


def train_epocs(epocs, model, train_dataloader, val_dataloader):
    # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []

    # for each epoch
    for epoch in range(epocs):

        print('\n Epoch {:} / {:}'.format(epoch + 1, epocs))

        print("Start training")
        # train model
        train_loss, _ = train(model, train_dataloader)#train()
        print("End training, start evaluation")
        # evaluate model
        valid_loss, _ = evaluate(model, val_dataloader)#evaluate()
        print("End evaluation")
        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), train_base_path + file_name)

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')

        # np.mean(valid_losses)

        # print(classification_report(val_y, valid_losses))

        model.eval()
        bert_predicted = []
        all_logits = []
        with torch.no_grad():
            for step_num, batch_data in enumerate(val_dataloader):
                token_ids, masks, labels = tuple(t.to(device) for t in batch_data)

                logits = model(token_ids, masks)
                # loss_func = nn.BCELoss()

                # loss = loss_func(logits, labels)

               # labels = labels.view(-1, 1)
                # compute the validation loss between actual and predicted values
                #labels = labels.to(torch.float32)
                loss = loss_fn(logits, labels)

                numpy_logits = logits.cpu().detach().numpy()

                bert_predicted += list(numpy_logits[:, 0] > 0.5)
                all_logits += list(numpy_logits[:, 0])

        np.mean(bert_predicted)

        print(classification_report(val_y, bert_predicted))
        plot_train_logs(train_losses, valid_losses)

def plot_train_logs(t_losses, v_losses):
    # plot training progress
    plt.plot(t_losses)
    plt.plot(v_losses)
    plt.title('loss=loss_fn')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    # plt.ylim(bottom=0, top=3)
    plt.show()
    plt.savefig('training_log.png')
    plt.close()


class Dataset:
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 32

    def __getitem__(self, idx):
        return texts, indices

def indicize_labels(labels):
    categories = sorted(list(set(y_train)))
    n_categories = len(categories)
    """Transforms string labels into indices"""
    indices=[]
    for j in range(len(labels)):
        for i in range(n_categories):
            if labels[j]==categories[i]:
                indices.append(i)
    return indices

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(train_base_path)
    #train_dataloader, val_dataloader = create_data_loader(y_train, y_val)#convert_lists_to_tensors(y_train, y_val)
    texts = load_data_from_file("_train", size)
    indices = indicize_labels(y_train)  # Integer label indices
    file_dataset = Dataset()
    batch_size = 8
    #tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    train_dataloader = RobertoDataLoader(alephbert_tokenizer, "_train", y_train, batch_size, 512)
    val_dataloader = RobertoDataLoader(alephbert_tokenizer, "_val", y_val, batch_size, 512)
    # a = train_dataloader.__iter__()
    # print(a)
    print("1")
    model, optimizer, epocs = create_model()#tokens_train)
    print("2")

    train_epocs(epocs, model, train_dataloader, val_dataloader, y_val)
    #train_epocs(epocs, model, val_dataloader, y_val)

# def create_data_loader( y_train, y_val):
#     df_train = load_data_from_file("_train",
#                                    size)  # pd.read_csv("/content/drive/MyDrive/Colab Notebooks/train_dataset.csv", index_col=0)
#
#     # X = df_train.loc[:, df_train.columns != 'label']
#     # y = df_train.loc[:, df_train.columns == 'label']
#     df_val = load_data_from_file("_val",
#                                  size)  # pd.read_csv("/content/drive/MyDrive/Colab Notebooks/test_dataset.csv", index_col=0)
#
#     """# Fine-tune BERT"""
#     # define a batch size
#     batch_size = 1  # 4 #2 0.63
#
#     # sampler for sampling the data during training
#     # train_sampler = RandomSampler(train_data)
#
#     # dataLoader for train set
#     train_dataloader = DataLoader(df_train, batch_size=batch_size)
#
#     # wrap tensors
#    # val_data = TensorDataset(val_seq, val_mask, val_y)
#
#     # sampler for sampling the data during training
#     # val_sampler = SequentialSampler(val_data)
#
#     # dataLoader for validation set
#     val_dataloader = DataLoader(df_val, batch_size=batch_size)
#
#     return train_dataloader, val_dataloader
