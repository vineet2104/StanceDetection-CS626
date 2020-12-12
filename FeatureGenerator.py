
import pandas as pd
from skip_thoughts import configuration
from skip_thoughts import encoder_manager
from sklearn.feature_extraction.text import TfidfVectorizer

VOCAB_FILE = ".\\skip_thoughts_bi_2017_02_16\\vocab.txt"
EMBEDDING_MATRIX_FILE = ".\\skip_thoughts_bi_2017_02_16\\embeddings.npy"
CHECKPOINT_PATH = ".\\skip_thoughts_bi_2017_02_16\\model.ckpt-500008"

encoder = encoder_manager.EncoderManager()
encoder.load_model(configuration.model_config(bidirectional_encoder=True),vocabulary_file=VOCAB_FILE,embedding_matrix_file=EMBEDDING_MATRIX_FILE,checkpoint_path=CHECKPOINT_PATH)

def neural_features(dataset_loc):

	english_dataset = pd.read_csv(dataset_loc)
	headline = english_dataset['headline']
	body = english_dataset['content']
	labels = [int(x) for x in english_dataset['label']]

	labels_done = []
    flag = True
    body_encodings = np.zeros((len(body),2400))
    j = 0
    for i in range(len(body)):
        flag=True
        try:
            current_body_encoding = encoder.encode(body[i:i+1])
        except:
            flag=False
        
        if(flag==True):
            
            labels_done.append(i)
            print("Labels Done "+str(len(labels_done)))
            body_encodings[len(labels_done)-1] = current_body_encoding

    body_encodings = body_encodings[0:len(labels_done)]

    headline_encodings = np.zeros((len(labels_done),2400))
    for count,i in enumerate(labels_done):
        current_headline_encode = encoder.encode(headline[i:i+1])
        headline_encodings[count] = current_headline_encode

    feat1 = np.zeros((len(labels_done),2400))
    feat2 = np.zeros((len(labels_done),2400))
    i = 0
    for headline_vector,body_vector in zip(headline_encodings,body_encodings):
        print(i)
        feat1[i] = np.dot(headline_vector,body_vector)
        feat2[i] = np.absolute(headline_vector-body_vector)
        i+=1

    final_neural_features = np.concatenate((feat1,feat2),axis = 1)

    return final_neural_features,labels_done


def statistical_features(dataset_loc,labels_done_):

	english_dataset = pd.read_csv(dataset_loc)
	headline = english_dataset['headline']
	body = english_dataset['content']
	labels = [int(x) for x in english_dataset['label']]

    headline_ = []
    body_ = []
    for i in labels_done_:
        headline_.append(headline[i])
        body_.append(body[i])

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(headline_)

    vectorizer = TfidfVectorizer(max_features = (10000-X.shape[1]))
    Y = vectorizer.fit_transform(body_)

    final_statistical_features = np.concatenate((np.array(X.toarray()),np.array(Y.toarray())),axis = 1)

    return final_statistical_features

def external_features(dataset_loc,labels_done_):
    
	english_dataset = pd.read_csv(dataset_loc)
	headline = english_dataset['headline']
	body = english_dataset['content']
	labels = [int(x) for x in english_dataset['label']]

    headline_ = []
    body_ = []
    for i in labels_done_:
        headline_.append(headline[i])
        body_.append(body[i])

    eng_ext = []
    i = 0
    for sent1,sent2 in zip(headline_,body_):
        print(i)
        i+=1
        vec = []
        #character ngrams
        for n in range(2,17):
        n_grams_1 = ngrams(sent1.lower(), n)
        n_grams_2 = ngrams(sent2.lower(),n)
        vec.append(len(list(set(n_grams_1).intersection(n_grams_2))))

        #word ngrams
        for n in range(2,7):
            n_grams_1 = ngrams(sent1.lower().split(), n)
            n_grams_2 = ngrams(sent2.lower().split(),n)
            vec.append(len(list(set(n_grams_1).intersection(n_grams_2))))

        #Sentence polarity
        flag=False
        text1 = Text(sent1)
        text2 = Text(sent2)
        pol1 = 0
        pol2 = 0
        for word in text1.words:
            try:
                pol1+=word.polarity
            except:
                flag=True
                vec.append(0)
                break
         if (flag==True):
            eng_ext.append(vec)
            continue
        flag=False
        for word in text2.words:
            try:
                pol2+=word.polarity
            except:
                flag=True
                vec.append(0)
                break
        if (flag==True):
            eng_ext.append(vec)
            continue
  
        pol1 = pol1/(len(sent1.split())*1.0)
        pol2 = pol2/(len(sent2.split())*1.0)
        vec.append(pol1-pol2)

        eng_ext.append(vec)

    eng_ext = np.array(eng_ext)

    return eng_ext






    



