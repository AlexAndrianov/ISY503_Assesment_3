import pandas as pd

# code to: 
#Encode the words in the review
#Encode the labels for ‘positive’ and ‘negative’
#Conduct outlier removal to eliminate really short or wrong reviews.
#Pad/truncate remaining data
#Split the data into training, validation and test sets

#after that we will have: training.csv validation.csv and test.csv 

# Load split dataset
data = pd.read_csv('training.csv')

# Ensure dataset fields
assert 'text' in data.columns
assert 'label' in data.columns

texts = data['text'].tolist()
labels = data['label'].tolist()


# from transformers import BertTokenizer

# # Load pre-trained BERT tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# def tokenize_data(texts, tokenizer, max_length=500):
#     return tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='tf')

# encoded_data = tokenize_data(texts, tokenizer)
# input_ids = encoded_data['input_ids']
# attention_masks = encoded_data['attention_mask']

# import tensorflow as tf

# # Convert labels to tensor
# labels = tf.convert_to_tensor(labels)

# # Create a TensorFlow dataset
# dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids, 'attention_mask': attention_masks}, labels))

# # Shuffle dataset
# batch_size = 32
# dataset = dataset.shuffle(len(texts)).batch(batch_size)

# from transformers import TFBertForSequenceClassification

# # Load pre-trained BERT model for sequence classification
# model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# # Compile the model
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), 
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
#               metrics=['accuracy'])

# # Train the model
# model.fit(dataset, epochs=3)

class BERT_Model:
    pass

def compile_NLP_model():
    #save model to database 
    return None

def analyse_text(text):
    #load model from database

    return 0

#Obtain batches of training data (you may use DataLoaders or generator functions)
#Define the network architecture
#Define the model class
#Instantiate the network
#Train your model
#Test
#Develop a simple web page/create an executable of your solution that will take an input sentence and provide an output of whether the review
#sentiment was positive or negative.
#Run an inference on some test input data – both positive and negative and observe how often the model gets these right.
#Repeat training and rearchitect the model if required.

#saving and loading model from sqlite