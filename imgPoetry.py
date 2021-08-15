# encoding:utf-8
#项目的说明可参见：查图配诗--给好图配个好诗 https://aistudio.baidu.com/aistudio/projectdetail/2283565?channelType=0&channel=0

import numpy as np
import json
import base64
import requests
import paddlehub as hub
def get_img_base64str(image):
    with open(image, 'rb') as fp:
        imgbase64 = base64.b64encode(fp.read())
        return imgbase64.decode()
def get_access_token(APP_ID, API_KEY, SECRET_KEY):
    params = {
        "grant_type": "client_credentials",
        'client_id': API_KEY,
        'client_secret': SECRET_KEY, }
    token_url = 'https://aip.baidubce.com/oauth/2.0/token'
    res = requests.get(token_url, params=params)
    try:
        data = res.json()
        return data['access_token']
    except:
        return ''
def get_allchars(dataset):
    allchars = []
    for i in range(len(dataset)):
        allchars += [dataset[i][4].replace('\n', '')]   
    return ''.join(allchars)
def get_dict(allchars):
    char_freq_dict = dict()  
    for char in allchars:
        if char not in char_freq_dict:
            char_freq_dict[char] = 0
        char_freq_dict[char] += 1
    char_freq_dict = sorted(char_freq_dict.items(), key = lambda x:x[1], reverse = True)
    char2id_dict = dict()
    id2char_dict = dict()
    n = 0
    for i in range(len(char_freq_dict)):
        char2id_dict[char_freq_dict[i][0]] = i
        id2char_dict[i] = char_freq_dict[i][0]
    return char2id_dict, id2char_dict
def is_chinese(string):
    """
    检查整个字符串是否为中文
    Args:
        string (str): 需要检查的字符串,包含空格也是False
    Return
        bool
    """
    if (len(string) <= 1): 
        return False
    for chart in string: 
        if (chart < u'\u4e00' or chart > u'\u9fff'):
            return False
    return True
def get_data(chars, char2id_dict, dataset):
    data_list = [None]*len(dataset)
    for i in range(len(dataset)):
        adata = dataset[i][4].replace('\n', '')
        data_list[i] = [char2id_dict[char] for char in adata]
        data_list[i] += [char2id_dict['<END>']]
        data_list[i] = data_list[i][-(L-1):]
        data_list[i] = [char2id_dict['<START>']] + data_list[i]
        data_list[i] = [char2id_dict['</s>']]*(L-len(data_list[i])) + data_list[i]
    return np.array(data_list)
dataset = json.load(open('唐诗.json',encoding="utf-8"))   
allchars = get_allchars(dataset)
char2id_dict, id2char_dict = get_dict(allchars)
char2id_dict['</s>'] = 7394
char2id_dict['<START>'] = 7395
char2id_dict['<END>'] = 7396
id2char_dict[7394] = '</s>'
id2char_dict[7395] = '<START>'
id2char_dict[7396] = '<END>'
L = 125 
data = get_data(allchars, char2id_dict, dataset)
'''
通用物体和场景识别
'''
request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general"
image = './test/yh.jpg'
img=get_img_base64str(image)
params = {"image":img}
#需要申请百度api，获取相关以下信息，目前个人也可免费申请。
APP_ID = '******'
API_KEY = '*******'
SECRET_KEY = '*******'
access_token = get_access_token(APP_ID, API_KEY, SECRET_KEY)
request_url = request_url + "?access_token=" + access_token
headers = {'content-type': 'application/x-www-form-urlencoded'}
response = requests.post(request_url, data=params, headers=headers)
module_image = hub.Module(name="xception71_imagenet") 
test_img_path = image
input_dict = {"image": [test_img_path]}
results_image = module_image.classification(data=input_dict)
PictureClassification = list(results_image[0][0].keys())[0]
from translate import Translator
translator = Translator(to_lang="chinese")
PictureClassification_ch = translator.translate("{}".format(PictureClassification))
if response:
    result_num=response.json()['result_num']
    work_num=int(result_num)+1
else:
    work_num=1
#work_num=1 #若不使用百度api，可以将百度api相关代码注释掉，启用此句，应该也是可以跑通的，相当于只使用xception71_imagenet模型分类。
total_max_icount=[0]
total_result=[(0,0)]    # j记录诗歌id。#icount记录字匹配量。
for  j in range(work_num):     
    if j==0:
        PictureClassification_ch=PictureClassification_ch
    else:
        PictureClassification_ch=response.json()['result'][j-1]['keyword']
    module_similar = hub.Module(name="ernie_gen_couplet") 
    texts = ["{}".format(PictureClassification_ch)]
    results = module_similar.generate(texts=texts, use_gpu=True, beam_width=20)
    words = [] 
    for item in range(20):
        if (is_chinese(results[0][item])):
            words.append(results[0][item])
    strwords="".join(words)
    words_dict=get_dict(strwords)
    setwords=set(strwords)
    words_set=get_dict(setwords)
    result=[(0,0)]
    max_icount=[0]  
    for j in range(len(dataset)):
        txt = []
        for i in range(L):
            txt += [id2char_dict[data[j,i]]]
        icount=0
        for item in setwords:
            icount=icount+txt.count(item)
        if max_icount[len(max_icount)-1]<icount:   
            max_icount[0]=icount       
            max_icount.sort()
            result[0] = (j, icount) 
            result=sorted(result, key=lambda x: (x[1], x[0])) 
            total_max_icount[0] = icount
            total_result[0]  = (j, icount)
            total_max_icount.sort()
            total_result = sorted(total_result, key=lambda x: (x[1], x[0]))
        else:
            if max_icount[len(max_icount)-1]== icount and icount!=0:
                if max_icount[0] < icount:
                    max_icount[0] = icount
                    max_icount.sort()
                    result[0] = (j, icount) 
                    result = sorted(result, key=lambda x: (x[1], x[0])) 
                    total_max_icount[0] = icount
                    total_result[0] = (j, icount)
                    total_max_icount.sort()
                    total_result = sorted(total_result, key=lambda x: (x[1], x[0]))
                else:
                    max_icount.append(icount)
                    total_max_icount.append(icount)
                    if result[0][0] == 0 and result[0][1] == 0:
                        result[0] = (j, icount) 
                        total_result[0] = (j, icount)
                    else:
                        result.append((j, icount))
                        total_result.append((j, icount))
max_positive=0
max_total_result=[0,0,"",0,0] 
for k in range(len(total_result)):
    senta = hub.Module(name="senta_bow")
    test_text =["{}".format(dataset[total_result[k][0]][4])]       #新手处理方式，估计还有更好的，期待能被指正吧。
    emotion_results = senta.sentiment_classify(texts=test_text, use_gpu=False, batch_size=1)
    for emotion in emotion_results:
        if max_positive<=emotion['positive_probs']: 
              max_positive=emotion['positive_probs']
              max_total_result=[total_result[k][0],total_result[k][1],emotion['sentiment_key'],emotion['positive_probs'],emotion['negative_probs']]

#最后输出最佳匹配的诗：
print("诗全部信息dataset[max_total_result[0]]=", dataset[max_total_result[0]])
print("匹配数量total_result[k]][1]=", max_total_result[1])
print("诗内容dataset[max_total_result[0]][4]=", dataset[max_total_result[0]][4])
print("该诗的情感倾向：", max_total_result[2])
print("积极的情感倾向概率：", max_total_result[3])
print("消极的情感倾向概率：", max_total_result[4])
