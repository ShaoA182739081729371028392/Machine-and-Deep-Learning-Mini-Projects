import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import albumentations as A
import numpy as np
import cv2
from albumentations.pytorch import ToTensorV2
import timm
import nltk
nltk.download("punkt")
# Unwieldy amount of Code: Because it's not in a Jupyter Notebook.
device = torch.device("cpu")
IMAGE_SIZE = 224
to_tensor = torchvision.transforms.ToTensor() 

test_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(),
    ToTensorV2()
])
GLOVE_PATH = None
VOCAB_PATH = "./vocab.pth"
MODEL_PATH = "./BestVal.pth"
print("Loading Vocab Path")
vocabulary = torch.load(VOCAB_PATH)
print("Loading State Dict")
state_dict = torch.load(MODEL_PATH, map_location = device)
class ConvBlock(nn.Module):
  def __init__(self, in_features, out_features, kernel_size, padding, groups):
    super().__init__()
    self.conv = nn.Conv2d(in_features, out_features, kernel_size = kernel_size, padding = padding, groups = groups)
    self.bn = nn.BatchNorm2d(out_features)
    self.act1 = nn.SiLU(inplace = True)
  def forward(self, x):
    return self.bn(self.act1(self.conv(x)))

class ResNetBase(nn.Module):
    '''
    Tiny ResNet Pretrained as a baseline
    '''
    def freeze(self, layer):
        for parameter in layer.parameters():
            parameter.requires_grad = False
    def __init__(self, in_dim, device):
        super().__init__()
        self.in_dim = in_dim
        self.device = device
        self.model_name = 'resnet200d'
        self.model = timm.create_model(self.model_name, pretrained = False)
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.freeze(self.model)
        self.proj = ConvBlock(2048, self.in_dim, 1, 0, 1)
    def forward(self, x):
        features = self.proj(self.model(x))
        B, C, H, W = features.shape
        return features.view(B, C, H * W).transpose(1, 2)

class CustomTokens(nn.Module):
    def __init__(self, vocabulary, glove_path, max_length, device):
        super().__init__()
        self.device = device
        self.glove_path = glove_path
        self.vocab = vocabulary
        self.idx2word = {idx + 4: self.vocab[idx] for idx in range(len(self.vocab))}
        self.START = "<START>"
        self.END = "<END>"
        self.PAD = "<PAD>"
        self.UNK = "<UNK>"
        self.START_ID = 0
        self.END_ID = 1
        self.PAD_ID = 2
        self.UNK_ID = 3
    
        
        self.idx2word[self.START_ID]=self.START
        self.idx2word[self.END_ID] = self.END
        self.idx2word[self.PAD_ID] = self.PAD
        self.idx2word[self.UNK_ID] = self.UNK
        
        self.word2idx = {self.vocab[idx]: idx + 4 for idx in range(len(self.vocab))}
        self.word2idx[self.START] = self.START_ID
        self.word2idx[self.END] = self.END_ID
        self.word2idx[self.PAD] = self.PAD_ID
        self.word2idx[self.UNK] = self.UNK_ID
        self.dim = 200
        self.max_length = max_length
        if self.glove_path:
            self.embeddings = self.load_glove(self.glove_path)
        else:
            self.embeddings = nn.init.xavier_uniform(torch.zeros((len(self.word2idx), self.dim), device = self.device))
        self.embeddings = nn.Embedding(len(self.word2idx), self.dim, _weight = self.embeddings)
    def load_glove(self, glove_path):
        '''
        glove_path: path to the glove file.
        '''
        embeddings = nn.init.xavier_uniform(torch.zeros((len(self.word2idx), self.dim), device = self.device))
        with open(glove_path, 'r') as file:
            for line in tqdm.tqdm(file):
                vals = line.split()
                word = vals[0]
                if word in self.word2idx:
                    embedding = torch.tensor([float(val) for val in vals[1:]], device = self.device)
                    embeddings[self.word2idx[word], :] = embedding
        return embeddings
    def decode(self, x):
        return self.idx2word[x]
    def pad_sents(self, x):
        '''
        Pads and Tokenizes Inputs
        '''
        tokenized_sents = []
        for sent in x:
            tok_sent = [self.PAD_ID for i in range(self.max_length)]
            for word_idx in range(self.max_length):
                if word_idx >= len(sent):
                    break
                if sent[word_idx] in self.word2idx:
                    tok_sent[word_idx] = self.word2idx[sent[word_idx]]
                else:
                    tok_sent[word_idx] = self.word2idx[self.UNK]
            tokenized_sents += [tok_sent]
        return torch.tensor(tokenized_sents, device = self.device)
            
    def forward(self, x):
        '''
        Tokenizes a Given Input
        '''
        tokenized = [nltk.word_tokenize(sent) for sent in x]
        padded = self.pad_sents(tokenized)
        return padded
    def embed(self, x):
        return self.embeddings(x)
class LSTM(nn.Module):
    '''
    Uses LSTMS to Caption Images
    '''
    def __init__(self, in_dim, im_dim, max_length, device, drop_prob = 0.2):
        super().__init__()
        self.im_dim = im_dim
        self.in_dim = in_dim
        self.drop_prob = drop_prob
        self.proj_hidden = nn.Linear(self.im_dim, self.im_dim)
        self.proj_cell = nn.Linear(self.im_dim, self.im_dim)
        self.max_length = max_length
        self.device = device
        
        #self.Attention = Attention(self.im_dim, self.im_dim, self.im_dim)
        self.LSTMCell = nn.LSTMCell(self.in_dim, self.im_dim)
        
        #self.model_name = 'distilbert-base-uncased'
        self.tokenizer = CustomTokens(vocabulary, GLOVE_PATH, self.max_length, self.device)
        self.num_classes = len(self.tokenizer.word2idx)
        self.Dropout = nn.Dropout(self.drop_prob)
        self.Linear = nn.Linear(self.im_dim, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
    def forward_train(self, x, GT):
        '''
        Uses LSTMs to Generate Captions
        '''
        # Tokenize Ground Truth
        tokenized_GT = self.tokenizer(GT)
        B, L = tokenized_GT.shape
        # Project the Image down to One LSTM Cell
        hidden_state = self.proj_hidden(torch.mean(x, dim = 1))
        cell_state = self.proj_cell(torch.mean(x, dim = 1))
        total_loss = torch.zeros((1), device = self.device)
        # Begin Decoding
        for l in range(0, L - 1):
            input_id = tokenized_GT[:, l]
            GT_id = tokenized_GT[:, l + 1]
            
            embeddings = self.tokenizer.embed(input_id)
            hidden_state, cell_state = self.LSTMCell(embeddings, (hidden_state, cell_state))
            copied_h = hidden_state.clone()
            #attended = self.Attention(copied_h, x)
            #concat = torch.cat([attended, copied_h], dim = -1)
            concat = self.Dropout(copied_h)
            pred = self.Linear(concat)
            # Mask Out PAD tokens
            keep = GT_id != self.tokenizer.PAD_ID
            pred = pred[keep]
            GT_id = GT_id[keep]
            if pred.shape[0] != 0:
                loss = self.criterion(pred, GT_id)
                total_loss = total_loss + loss
        return total_loss
    def forward(self, x):
        # Project Down the X to the hidden_state
        hidden_state = self.proj_hidden(torch.mean(x, dim = 1))
        cell_state = self.proj_cell(torch.mean(x, dim = 1) )
        B, L = cell_state.shape
        # Starter Sentences
        current_tokens = torch.tensor([self.tokenizer.START_ID for i in range(B)], device = self.device) 
        pred_sentences = [self.tokenizer.START for i in range(B)]
        finished = [False for i in range(B)]
        # Begin Decoding
        for i in range(self.max_length):
            embeddings = self.tokenizer.embed(current_tokens)
            hidden_state, cell_state = self.LSTMCell(embeddings, (hidden_state, cell_state))
            copied_h = hidden_state.clone()
            #attended = self.Attention(copied_h, x)
            #concat = torch.cat([attended, copied_h], dim = -1)
            pred = F.softmax(self.Linear(copied_h))
            _, indices = torch.max(pred, dim = -1)
            for b in range(B):
                if finished[b]:
                    continue # Finished Already
                elif indices[b].item() == self.tokenizer.END_ID:
                    finished[b] = True
                    pred_sentences[b] += f" {self.tokenizer.decode(indices[b].item())}"
                else:
                    pred_sentences[b] += f" {self.tokenizer.decode(indices[b].item())}"
                    current_tokens[b] = indices[b].item()
        return pred_sentences
class FullModel(nn.Module):
  '''
  Houses the Full Image Captioning Model
  '''
  def __init__(self, device):
    super().__init__()
    self.device = device
    self.in_dim = 200
    self.im_dim = 2048
    self.max_length = 20
    self.image_encoder = ResNetBase(self.im_dim, self.device)
    self.LSTM = LSTM(self.in_dim, self.im_dim, self.max_length, self.device)
  def forward_train(self, images, captioning):
    image_encoded = self.image_encoder(images)
    return self.LSTM.forward_train(image_encoded, list(captioning))
  def forward(self, images):
    image_encoded = self.image_encoder(images)
    return self.LSTM(image_encoded)
class ImageCaptioningSolver(nn.Module):
  def __init__(self, device):
    super().__init__()
    self.device = device
    self.model = FullModel(self.device)
  def forward(self, x):
    self.eval()
    with torch.no_grad():
      return self.model(x)

def load_captioner():
    '''
    Loads the Entire Pytorch Model, it's file path, etc. and Returns it.
    '''
    model = ImageCaptioningSolver(device)
    model.model.load_state_dict(state_dict)
    return model
def process_image(model, image):
    image = np.array(image.convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = test_transforms(image = image)['image'].unsqueeze(0)
    caption = model(image)
    return caption