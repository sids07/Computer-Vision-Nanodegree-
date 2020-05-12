import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        #print(features.shape)
        features = features.view(features.size(0), -1)
        #print(features.shape)
        features = self.embed(features)
        #print(features.shape)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN,self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size,embed_size)
        #suru ma ako op vocab_size ma hunxa teslai 
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,batch_first =True)
        self.final = nn.Linear(hidden_size,vocab_size)
        
    def forward(self, features, captions):
        
        captions = captions[:, :-1]
        #all caption except end token is used.
        
        captions = captions.type(torch.long)
        embeded = self.embed(captions)
        #caption = (batch_size,caption_size-1) --> embeded = (batch_size,caption_size-1,embed_size)
        
        inp = torch.cat((features.unsqueeze(dim=1),embeded),dim=1)
        #features = (batch_size,embed_size)
        #feature is 2D but embeded is 3D so to concat we add dimension to index 1 i.e. (batch_size,1,embed_size)
        #when concat inp = (batch_size,caption_size,embed_size)
        x, _ = self.lstm(inp)
        #x = (batch_size,caption_size, hidden_size)
        
        x = self.final(x)
        #since output must be equivalent to vocab . So, now x = (batch_size,caption_size,vocab_size)
            
        return x

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []
        for i in range(max_len):
            out,states = self.lstm(inputs,states)
            #at first states will be None then after each iteration states value will be updated
            #inputs = (1,1,emb_size) and output = (1,1,hid_size)
            output = self.final(out)
            #(1,1,vocab_size)
            
            predict = output.argmax(dim=2)
            #find max value on dim=1 i.e. vocab_size then output = (1,1)
            outputs.append(predict[0].item())
            #predict[0].item() will give a single number output
            inputs = self.embed(predict)
            #inputs is updated with new predicted word. Here i/p = (1,1) and o/p = (1,1,256) i.e. again ready for process
        return outputs