###########################################################################################################################################
import pandas as pd

INPUT_FOLDER = '/Users/simone/desktop/Homework2_NLP/CNN_task'
OUTPUT_FOLDER = '/Users/simone/desktop/Homework2_NLP/CNN_task'

PATH_TO_YELP_REVIEWS = INPUT_FOLDER + '/review.json'
OUTPUT_FILE = OUTPUT_FOLDER + '/output_reviews_top.csv'
chunk_size = 1000000

# create an empty DataFrame to store the results
results = pd.DataFrame()

# read the file in chunks and append the results to the DataFrame
#for chunk in pd.read_json(PATH_TO_YELP_REVIEWS, lines=True, chunksize=chunk_size):
for chunk in pd.read_json(PATH_TO_YELP_REVIEWS, lines=True, chunksize=chunk_size):
    results = results.append(chunk)
    if len(results) >= 1000000:
        break

# save the first 100000 rows to a CSV file
#results.head(100000).to_csv(OUTPUT_FILE, index=False)

top_data_df = pd.read_csv(INPUT_FOLDER + '/output_reviews_top.csv')
print("Columns in the original dataset:\n")
print(top_data_df.columns)



################################################################################################################################
#defining true sentiment from stars

import matplotlib.pyplot as plt 

#print count of rows per star rating
print("Number of rows per star rating:")
print(top_data_df['stars'].value_counts())

# Function to map stars to sentiment
def map_sentiment(stars_received):
    if stars_received <= 2:
        return -1
    elif stars_received == 3:
        return 0
    else:
        return 1

# Mapping stars to sentiment into three categories
top_data_df['sentiment'] = [ map_sentiment(x) for x in top_data_df['stars']]
# Plotting the sentiment distribution
plt.figure()
pd.value_counts(top_data_df['sentiment']).plot.bar(title="Sentiment distribution in df")
plt.xlabel("Sentiment")
plt.ylabel("No. of rows in df")
plt.show()


###################################################################################################################################################
##Creating equal distribution of each sentiment in df
# Function to retrieve top few number of each category
def get_top_data(top_n = 5000):
    top_data_df_positive = top_data_df[top_data_df['sentiment'] == 1].head(top_n)
    top_data_df_negative = top_data_df[top_data_df['sentiment'] == -1].head(top_n)
    top_data_df_neutral = top_data_df[top_data_df['sentiment'] == 0].head(top_n)
    top_data_df_small = pd.concat([top_data_df_positive, top_data_df_negative, top_data_df_neutral])
    return top_data_df_small

# Function call to get the top 10000 from each sentiment
top_data_df_small = get_top_data(top_n=10000)

# After selecting top few samples of each sentiment
print("After segregating and taking equal number of rows for each sentiment:")
print(top_data_df_small['sentiment'].value_counts())
top_data_df_small.head(10)

# Plotting the sentiment distribution
plt.figure()
pd.value_counts(top_data_df_small['sentiment']).plot.bar(title="Sentiment distribution in df")
plt.xlabel("Sentiment")
plt.ylabel("No. of rows in df")
plt.show()





########################################################################################################################
#PREPROCESS
from gensim.parsing.preprocessing import remove_stopwords

# Define a function to remove stopwords from a sentence
def preprocess_sentence(sentence):
    return remove_stopwords(sentence)

# Apply the preprocessing function to the Sentences column
top_data_df_small['ProcessedSentences'] = top_data_df_small['text'].apply(preprocess_sentence)


from gensim.utils import simple_preprocess

# Tokenize the text column to get the new column 'tokenized_text'
top_data_df_small['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in top_data_df_small['ProcessedSentences']] 
print(top_data_df_small['tokenized_text'].head(10))

from gensim.parsing.porter import PorterStemmer

porter_stemmer = PorterStemmer()
# Get the stemmed_tokens
top_data_df_small['stemmed_tokens'] = [[porter_stemmer.stem(word) for word in tokens] for tokens in top_data_df_small['tokenized_text'] ]
top_data_df_small['stemmed_tokens'].head(10)




###################################################################################################################################
##### TRAIN TEST SPLIT

from sklearn.model_selection import train_test_split

# Train Test Split Function
def split_train_test(top_data_df_small, test_size=0.2, shuffle_state=True):
    X_train, X_test, Y_train, Y_test = train_test_split(top_data_df_small[['business_id', 'cool', 'date', 'funny', 'review_id', 'stars', 'text', 'useful', 'user_id', 'stemmed_tokens']], 
                                                        top_data_df_small['sentiment'], 
                                                        shuffle=shuffle_state,
                                                        test_size=test_size, 
                                                        random_state=15)
    print("Value counts for Train sentiments")
    print(Y_train.value_counts())
    print("Value counts for Test sentiments")
    print(Y_test.value_counts())
    print(type(X_train))
    print(type(Y_train))
    X_train = X_train.reset_index()
    X_test = X_test.reset_index()
    Y_train = Y_train.to_frame()
    Y_train = Y_train.reset_index()
    Y_test = Y_test.to_frame()
    Y_test = Y_test.reset_index()
    print(X_train.head())
    return X_train, X_test, Y_train, Y_test

# Call the train_test_split
X_train, X_test, Y_train, Y_test = split_train_test(top_data_df_small)




########################################################################################################################
#Packages for CNN
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################################################################################################################
#WORD2VEC
from gensim.models import Word2Vec
size = 500
window = 3
min_count = 1
workers = 3
sg = 1

# Function to train word2vec model
def make_word2vec_model(top_data_df_small, padding=True, sg=1, min_count=1, vector_size=500, workers=3, window=3):
    if padding:
        print(len(top_data_df_small))
        temp_df = pd.Series(top_data_df_small['stemmed_tokens']).values
        temp_df = list(temp_df)
        temp_df.append(['pad'])
        word2vec_file = OUTPUT_FOLDER + '/models/' + 'word2vec_' + str(vector_size) + '_PAD.model'
    else:
        temp_df = top_data_df_small['stemmed_tokens']
        word2vec_file = OUTPUT_FOLDER + '/models/' + 'word2vec_' + str(vector_size) + '.model'
    w2v_model = Word2Vec(temp_df, min_count=min_count, vector_size=vector_size, workers=workers, window=window, sg=sg)

    w2v_model.save(word2vec_file)
    return w2v_model, word2vec_file

# Train Word2vec model
w2vmodel, word2vec_file = make_word2vec_model(top_data_df_small, padding=True, sg=sg, min_count=min_count, vector_size=500, workers=workers, window=window)

max_sen_len = top_data_df_small.stemmed_tokens.map(len).max()
padding_idx = w2vmodel.wv.key_to_index['pad']
def make_word2vec_vector_cnn(sentence):
    padded_X = [padding_idx for i in range(max_sen_len)]
    i = 0
    for word in sentence:
        if word not in w2vmodel.wv.key_to_index:
            padded_X[i] = 0
            print(word)
        else:
            padded_X[i] = w2vmodel.wv.key_to_index[word]
        i += 1
    return torch.tensor(padded_X, dtype=torch.long, device=device).view(1, -1)

def make_target(label):
    if label == -1:
        return torch.tensor([0], dtype=torch.long, device=device)
    elif label == 0:
        return torch.tensor([1], dtype=torch.long, device=device)
    else:
        return torch.tensor([2], dtype=torch.long, device=device)




##########################################################################################################
#Check embeddings
import random

# Load the pre-trained word2vec model
model = Word2Vec.load(OUTPUT_FOLDER + '/models/' + 'word2vec_500_PAD.model')

# Get the vocabulary of the model
keys = list(model.wv.key_to_index.keys())

# Generate 10 random words from the keys
random_words = random.sample(keys, 40)

# Print the embeddings of the random words
for word in random_words:
    embedding = model.wv[word]
    print(f"Embedding for '{word}': {embedding}")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# Get the embeddings of the random words
embeddings = [model.wv[word] for word in random_words]
embeddings = np.array(embeddings)

# Apply t-SNE to reduce the dimensionality of the embeddings to 2D
tsne = TSNE(n_components=2, random_state=42, perplexity= 5)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot the embeddings in 2D space
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

# Label each point with its corresponding word
for i, word in enumerate(random_words):
    plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

# Show the plot
plt.show()




##############################################################################################################
##########Define CNN

EMBEDDING_SIZE = 500
NUM_FILTERS = 10
import gensim

class CnnTextClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, window_sizes=(1,2,3,5)):
        super(CnnTextClassifier, self).__init__()
        w2vmodel = gensim.models.Word2Vec.load(OUTPUT_FOLDER + '/models/' + 'word2vec_500_PAD.model')
        weights = w2vmodel.wv
        # With pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors), padding_idx=weights.key_to_index['pad'])
        # Without pretrained embeddings
        # self.embedding = nn.Embedding(vocab_size, EMBEDDING_SIZE)

        self.convs = nn.ModuleList([
                                   nn.Conv2d(1, NUM_FILTERS, [window_size, EMBEDDING_SIZE], padding=(window_size - 1, 0))
                                   for window_size in window_sizes
        ])

        self.fc = nn.Linear(NUM_FILTERS * len(window_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x) # [B, T, E]

        # Apply a convolution + max_pool layer for each window size
        x = torch.unsqueeze(x, 1)
        xs = []
        for conv in self.convs:
            x2 = torch.tanh(conv(x))
            x2 = torch.squeeze(x2, -1)
            x2 = F.max_pool1d(x2, x2.size(2))
            xs.append(x2)
        x = torch.cat(xs, 2)

        # FC
        x = x.view(x.size(0), -1)
        logits = self.fc(x)

        probs = F.softmax(logits, dim = 1)

        return probs


######################################################################################################
########TRAIN CNN
#set the parameters
NUM_CLASSES = 3
VOCAB_SIZE = len(w2vmodel.wv.index_to_key)

cnn_model = CnnTextClassifier(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)
cnn_model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
num_epochs = 3 
padding_idx = 0

# Open the file for writing loss
loss_file_name = OUTPUT_FOLDER +  '/plots/' + 'cnn_class_big_loss_with_padding.csv'
f = open(loss_file_name,'w')
f.write('iter, loss')
f.write('\n')
losses = []
cnn_model.train()



#Train Model
from tqdm import tqdm

# Loop over epochs
for epoch in range(num_epochs):
    print("Epoch" + str(epoch + 1))
    
    # Initialize progress bar
    progress_bar = tqdm(X_train.iterrows(), total=len(X_train))
    
    # Initialize training loss
    train_loss = 0
    
    # Loop over training examples
    for index, row in progress_bar:
        # Clearing the accumulated gradients
        cnn_model.zero_grad()

        # Make the bag of words vector for stemmed tokens 
        bow_vec = make_word2vec_vector_cnn(row['stemmed_tokens'])
       
        # Forward pass to get output
        probs = cnn_model(bow_vec)

        # Get the target label
        target = make_target(Y_train['sentiment'][index])

        # Calculate Loss: softmax --> cross entropy loss
        loss = loss_function(probs, target)
        train_loss += loss.item()

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()
        
        # Update progress bar
        progress_bar.set_description(f'Training loss: {train_loss / (index+1):.4f}')
        progress_bar.update(1)
    
    # Finish the progress bar for the epoch
    progress_bar.close()

    # Write results to file
    f.write(str((epoch+1)) + "," + str(train_loss / len(X_train)))
    f.write('\n')
    
    # Reset training loss
    train_loss = 0


torch.save(cnn_model, OUTPUT_FOLDER + '\cnn_big_model_500_with_padding.pth')

f.close()
print("Input vector")
print(bow_vec.cpu().numpy())
print("Probs")
print(probs)
print(torch.argmax(probs, dim=1).cpu().numpy()[0])



############################################################################################################################################################
#TEST MODEL

cnn_model_path = (OUTPUT_FOLDER + '\cnn_big_model_500_with_padding.pth') 
cnn_model = torch.load(cnn_model_path)

from sklearn.metrics import classification_report
bow_cnn_predictions = []
original_lables_cnn_bow = []
cnn_model.eval()
loss_df = pd.read_csv(OUTPUT_FOLDER + 'cnn_class_big_loss_with_padding.csv')
print(loss_df.columns)
# loss_df.plot('loss')

with torch.no_grad():
    for index, row in X_test.iterrows():
        bow_vec = make_word2vec_vector_cnn(row['stemmed_tokens'])
        probs = cnn_model(bow_vec)
        _, predicted = torch.max(probs.data, 1)
        bow_cnn_predictions.append(predicted.cpu().numpy()[0])
        original_lables_cnn_bow.append(make_target(Y_test['sentiment'][index]).cpu().numpy()[0])

print(classification_report(original_lables_cnn_bow,bow_cnn_predictions))

loss_file_name = OUTPUT_FOLDER +  '/plots/' + 'cnn_class_big_loss_with_padding.csv'
loss_df = pd.read_csv(loss_file_name)
print(loss_df.columns)
plt_500_padding_30_epochs = loss_df[' loss'].plot()
fig = plt_500_padding_30_epochs.get_figure()
fig.savefig(OUTPUT_FOLDER +'/plots/' + 'loss_plt_500_padding_30_epochs.pdf')



# Convert report to pandas dataframe
report = classification_report(original_lables_cnn_bow,bow_cnn_predictions, output_dict=True)
df = pd.DataFrame(report)

# Transpose the dataframe so that classes are columns and metrics are rows
df = df.transpose()

# Save dataframe as csv
df.to_csv(OUTPUT_FOLDER + '/classification_report.csv', index=True)




#number of parameters
num_params = sum(p.numel() for p in cnn_model.convs[3].parameters() if p.requires_grad)

#View the hidden layer
# Define model and set to evaluation mode
cnn_model.eval()

# Pass dummy input through the model
dummy_input = torch.zeros((1, 10), dtype=torch.long) # example input with batch size 1 and sequence length 10

#embedding layer
hidden_output = cnn_model.embedding(dummy_input)



# Print shape of the hidden layer output
print(hidden_output.shape)

