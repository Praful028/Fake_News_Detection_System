import torch
import pickle
from flask import Flask, render_template, request
from transformers import AutoModel, BertTokenizerFast
import numpy as np

app = Flask(__name__)

# Define the BERT model and tokenizer
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


# Define the BERT_Arch class
class BERT_Arch(torch.nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = torch.nn.Dropout(0.1)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(768, 512)
        self.fc2 = torch.nn.Linear(512, 2)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Instantiate your BERT model
model = BERT_Arch(bert)

# Save the model using torch.save instead of pickling
torch.save(model.state_dict(), 'fake_news_detection_model.pth')

print('Model Loaded')

def get_className(ClassNo):
    if ClassNo==0:
        return "Fake News"
    elif ClassNo==1:
        return "Not Fake"

def getResult(newspred):
    resultpred = model.predict(newspred)
    classes_result = np.argmax(resultpred, axis=1)
    return classes_result

@app.route('/')
def home():
    return render_template('index.html')

def fake_news_det(news):
    tokens = tokenizer.batch_encode_plus(
        news,
        max_length=15,
        pad_to_max_length=True,
        truncation=True
    )
    seq = torch.tensor(tokens['input_ids'])
    mask = torch.tensor(tokens['attention_mask'])
    with torch.no_grad():
        preds = model(seq, mask)
        preds = preds.detach().cpu().numpy()
    return preds

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        #preds = fake_news_det([message])
        value= getResult(message)
        result = get_className(value)
        return result

if __name__ == '__main__':
    app.run(debug=True)
