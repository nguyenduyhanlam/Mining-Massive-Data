import pandas as pd 
import numpy as np
#Reading user file:
u_cols =  ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1')
#users_attack = pd.read_csv('u_attack.user', sep='|', names=u_cols,
# encoding='latin-1')

n_users = users.shape[0]
print('Number of users:', n_users)
# users.head() #uncomment this to see some few examples

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

rate_train = ratings_base.as_matrix()
rate_test = ratings_test.as_matrix()

print('Number of traing rates:', rate_train.shape[0])
print('Number of test rates:', rate_test.shape[0])

#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols,
 encoding='latin-1')

n_items = items.shape[0]
print('Number of items:', n_items)

def get_items_rated_by_user(rate_matrix, user_id):
    """
    in each line of rate_matrix, we have infor: user_id, item_id, rating (scores), time_stamp
    we care about the first three values
    return (item_ids, scores) rated by user user_id
    """
    y = rate_matrix[:,0] # all users
    # item indices rated by user_id
    # we need to +1 to user_id since in the rate_matrix, id starts from 1 
    # while index in python starts from 0
    ids = np.where(y == user_id + 1)[0] 
    item_ids = rate_matrix[ids, 1] - 1 # index starts from 0 
    scores = rate_matrix[ids, 2]
    return (item_ids, scores)
  
attack_profile = pd.read_csv('ua_attack.base', sep='\t', names=r_cols, encoding='latin-1')
attack_profile = attack_profile.as_matrix()
    
def WDMA(rate_matrix, user_id):
    """
    in each line of rate_matrix, we have infor: user_id, item_id, rating (scores), time_stamp
    we care about the first three values
    return (item_ids, scores) rated by user user_id
    """
    y = rate_matrix[:,0] # all users
    # item indices rated by user_id
    # we need to +1 to user_id since in the rate_matrix, id starts from 1 
    # while index in python starts from 0
    ids = np.where(y == user_id + 1)[0] 
    item_ids = rate_matrix[ids, 1] - 1 # index starts from 0 
    scores = rate_matrix[ids, 2]
    
    sum_WDMA = 0
    
    for f in range(len(item_ids)):
        x = rate_matrix[:,1]
        f_ids = np.where(x == item_ids[f] + 1)[0]
        user_rate_f = rate_matrix[f_ids, 0] - 1
        s = rate_matrix[f_ids, 2]
        avg = np.mean(s)
        minus_result = abs(scores[f] - avg)
        sum_WDMA = sum_WDMA + (minus_result / len(user_rate_f)**2)   
        
    WDMA_value = sum_WDMA / len(item_ids)
    return WDMA_value

n = 1
axis_x = []
np.set_printoptions(precision=2) # 2 digits after . 
ids, scores = get_items_rated_by_user(rate_test, n)
print('Rated movies ids :', ids + 1)
print('True ratings     :', scores)
print('Mean rating value:',np.mean(scores))
arrWDMA = []
for i in range(len(users)):
    normal_WDMA = WDMA(rate_train, i)
    print('WDMA normal profile',str(i+1),":",normal_WDMA)
    arrWDMA.append(normal_WDMA)
avg_WDMA = np.mean(normal_WDMA)

attack_arrWDMA = []
a_arrWDMA = []
count_fake = 0
for i in range(len(users)):
    attack_WDMA = WDMA(attack_profile, i)
    print('WDMA profile',str(i+1),"when have attacked:",attack_WDMA)
    if(i + 1 <= 893):
        attack_arrWDMA.append(['real',attack_WDMA])
    else:
        attack_arrWDMA.append(['fake',attack_WDMA])
    a_arrWDMA.append(attack_WDMA)
    
attack_arrWDMA = np.array(attack_arrWDMA)
#attack_arrWDMA = pd.DataFrame(attack_arrWDMA, columns=['type','WDMA'])
avg_att_WDMA = np.mean(a_arrWDMA)

tp = 0
fp = 0
tn = 0
fn = 0
for i in range(len(a_arrWDMA)):
    if i < 893:
        if(a_arrWDMA[i] <= avg_att_WDMA):
            tp = tp + 1
        else:
            fp = fp + 1
    else:
        if a_arrWDMA[i] > avg_att_WDMA:
            tp = tp + 1
        else:
            fn = fn + 1
            
print("true positive:",tp)
print("false positive:",fp)
print("false negative:",fn)

precision = tp / (tp + fp)
recall = tp / (tp + fn)
print("precision (tp/(tp+fp)):", precision)
print("recall (tp/(tp+fn)):", recall)