import warnings
warnings.filterwarnings("ignore")

# glove_embedding_path = '/media/kyle/Data/Data/embedding/glove.840B.300d.txt'
# crawl_embedding_path = '/media/kyle/Data/Data/embedding/crawl-300d-2M.vec'
# train_csv_path = 'train.csv'
# test_csv_path = 'test.csv'

glove_embedding_path = '/data2/jianglibin/toxicity/glove.840B.300d.txt'
crawl_embedding_path = '/data2/jianglibin/toxicity/crawl-300d-2M.vec'
glove_embedding_pkl_path = '/data2/jianglibin/toxicity/glove_embedding.pkl'
crawl_embedding_pkl_path = '/data2/jianglibin/toxicity/crawl_embedding.pkl'

train_csv_path = '/data2/jianglibin/toxicity/train.csv'
test_csv_path = '/data2/jianglibin/toxicity/test.csv'

toxicity_embadding_path = '/data2/jianglibin/toxicity/embedding5.pkl'
processed_data_path = '/data2/jianglibin/toxicity/processed_data5.pkl'
tokenizer_path = '/data2/jianglibin/toxicity/tokenizer5.pkl'

statistics_features_path = '/data2/jianglibin/toxicity/statistics_features.pkl'
normalize_path = '/data2/jianglibin/toxicity/normalize_data.pkl'
toxicity_word_path = '/data2/jianglibin/toxicity/toxicity_word.pkl'
word_value_path = '/data2/jianglibin/toxicity/word_value.pkl'

kfold_path = '/data2/jianglibin/toxicity/kfold_3.pkl'


cv = 3

MAX_LEN = 220

identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
                    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

aux_columns = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
aux_columns_new = ['rating', 'funny', 'wow', 'sad', 'likes', 'disagree', 'sexual_explicit', 'identity_annotator_count', 'toxicity_annotator_count']
# aux_columns = aux_columns + aux_columns_new

train_num = 1804874
train_csv_shape = (1804874, 45)
train_csv_columns = ['id', 'target', 'comment_text', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'asian', 'atheist', 'bisexual',
                     'black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu', 'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability',
                     'jewish', 'latino', 'male', 'muslim', 'other_disability', 'other_gender', 'other_race_or_ethnicity', 'other_religion', 'other_sexual_orientation',
                     'physical_disability', 'psychiatric_or_mental_illness', 'transgender', 'white', 'created_date', 'publication_id', 'parent_id', 'article_id',
                     'rating', 'funny', 'wow', 'sad', 'likes', 'disagree', 'sexual_explicit', 'identity_annotator_count', 'toxicity_annotator_count']

test_num = 97320
test_csv_shape = (97320, 2)
test_csv_columns = ['id', 'comment_text']

# n unknown words (glove):  130075