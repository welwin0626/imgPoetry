## [AI训练营]基于飞桨实现：好图要配上好诗<br>

## 一、前言
当我们看到美景或有了一张精美好图后，是不是会有配上一首好诗的冲动？<br>
AI可以帮你搞定！就像下面这样配首诗。<br>
![](https://ai-studio-static-online.cdn.bcebos.com/09e09d0da18048af967749375ed78ecea8360bafdbcf4baa902621e23ea8256d)<br>


AI根据这张图片生成的古诗节选是： <br>
- 莫言短枝条，中有长相思。
- 朱颜与绿杨，并在别离期。
- 楼上春风过，风前杨柳歌。
- 枝疏缘别苦，曲怨为年多。
- 花惊燕地云，叶映楚池波。
<br>
<br>怎么样？<br>
![](https://ai-studio-static-online.cdn.bcebos.com/104144c2c470495cb1661934479fdb21979372755b8346c598a6e5594d62561e)

<br>配诗感觉还不错吧！是不是觉得匹配得实在粗糙了点？<br>
不用着急，这是最粗糙的版本了，有着很多的优化空间，相信通过优化、训练，一定会达到我们满意的配诗效果。<br>
- 一起来看看还有哪些可以优化的吧！

## 二、实现思路

要想让机器根据一张图片匹配到古诗，可以分为以下四步：<br>
1. 最简单的就是图像分类，将图像分到某一个类别里，比如：上面那张图的类别可以是“樱花”。
2. 拿到图片的类别之后，也就有了这张图的关键字，但关键字往往很短，不足以写出一首完整的古诗，因此我们需要找到更多的关键词。
3. 当关键词足够多时就去找匹配度高的诗歌。
- 这里用了最简单的办法：关键词形成的字去匹配每首诗的每个字，匹配量大选中，有一个冗余，就可以得到一组匹配量较大的诗歌。
- 实测下来发现可能更好的办法应该是找到最核心的关键词必须能匹配上吧，有待进一步改善。
4. 最后对每首诗歌，分析一下情感倾向，选择情感较积极、匹配度较大的诗歌作为最后的推荐配诗。

## 三、调用三到四个模型就能快速匹配诗！

### 1.首先将paddlehub更新到最新的版本：


```python
!pip install --upgrade paddlehub -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.安装所需要的模型：
- 图像分类：xception71_imagenet
- 文本生成：ernie_gen_couplet
- 情感分析：senta_bow


```python
!hub install xception71_imagenet==1.0.0 #图像分类
!hub install ernie_gen_couplet==1.0.0 #文本生成（对联），获取近似词
!hub install senta_bow==1.2.0 #情感倾向分析
```

### 3.准备诗歌数据

- 唐诗.json,通过参考项目获取，已经上传。
- 数据集包含41560首唐诗。从json文件中读取41560x10的二维列表，列表中包含卷、作者、诗名、简体诗、繁体诗等内容，我们只需要其中的简体诗。
- 后续还可以继续扩展诗歌库。


```python
import json
dataset = json.load(open('/home/aistudio/唐诗.json'))
print(type(dataset))
print(len(dataset))
print(dataset[998])
print(len(dataset[998]))
print(dataset[998][4])
```

> 对语料进行统计，为每个字符构造ID。  
构造词典，统计每个字符(包括，。？！等)的频率，并根据频率将每个字符转换为一个整数ID，出现次数越高，排序越靠前。  
添加'< /s >','< START >','< END >'三个字符，分别代表填充、开始和结束。去掉换行符\n。


```python
def get_allchars(dataset):
    allchars = []
    for i in range(len(dataset)):
        allchars += [dataset[i][4].replace('\n', '')]
    return ''.join(allchars)

allchars = get_allchars(dataset)


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

char2id_dict, id2char_dict = get_dict(allchars)
print(len(char2id_dict))
char2id_dict['</s>'] = 7394
char2id_dict['<START>'] = 7395
char2id_dict['<END>'] = 7396
id2char_dict[7394] = '</s>'
id2char_dict[7395] = '<START>'
id2char_dict[7396] = '<END>'
```

> 使用一个循环，将语料中的每个词替换成对应的id，以便于神经网络进行处理。  
每首诗结尾加上'< END >'，然后截取倒数L-1个字符，再在前面加上'< START >'。如果此时不满L个字符，则在前面用'< /s >'补齐。  
最终得到 [多少首, 每首截取&补充长度] 的数组。


```python
L = 125

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

data = get_data(allchars, char2id_dict, dataset)
print("data=\n",data)
```

### 4.获取图片的分类

这里其实使用了两个图像分类模型,只用一个也行，两个一起用也行,第2个模型主要是验证一下目前图像分类模型大致上限：
- 一个是训练不太足的xception71_imagenet，例如：分类样图的樱花图片，得到的结果翻译成中文是“蠕虫围栏”；
- 另一个是直接调用了百度通用图像识别的api，效果就好很多，例如：上图分类结果为：樱花、晚樱、樱花盛开等。



```python
import paddlehub as hub

module_image = hub.Module(name="xception71_imagenet") # 调用图像分类的模型

test_img_path = "yh.jpg" # 选择图像，即要根据哪张图片写诗

# set input dict
input_dict = {"image": [test_img_path]}

# execute predict and print the result
results_image = module_image.classification(data=input_dict)

PictureClassification = list(results_image[0][0].keys())[0]
print("该图片的分类是：",PictureClassification) # 得到分类结果

```

因为这里得到的标签都是英文，所以这里我们先把英文翻译成中文,用到的库是translate：


```python
!pip install translate
```


```python
from translate import Translator

translator = Translator(to_lang="chinese")
PictureClassification_ch = translator.translate("{}".format(PictureClassification))

print("该图片的分类是：",PictureClassification_ch)
```


```python
import base64
import requests

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
#需要申请百度api，获取相关以下信息，目前个人也可免费申请。
APP_ID = '******'
API_KEY = '*******'
SECRET_KEY = '*******'
request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general"
image = 'yh.jpg'
# 二进制方式打开图片文件
img=get_img_base64str(image)
params = {"image":img}
access_token = get_access_token(APP_ID, API_KEY, SECRET_KEY)
request_url = request_url + "?access_token=" + access_token
headers = {'content-type': 'application/x-www-form-urlencoded'}
response = requests.post(request_url, data=params, headers=headers)
if response:
    print (response.json())

```

### 5.根据图片的分类获取近义词

使用对联生成的模型来生成近义词，使用对联生成的可行性分析：<br>
- 对联对仗工整，平仄协调，字数相同，结构相同
- 生成的近义词能更加丰富，诗词是丰富多彩的，需要丰富而充满想象力的词汇
- 大胆抄袭参考项目后，发现效果不错


```python
module_similar = hub.Module(name="ernie_gen_couplet") # 调用对联生成的模型
# PictureClassification_ch="樱花"  #可以用来单cell测试
texts = ["{}".format(PictureClassification_ch)]
results = module_similar.generate(texts=texts, use_gpu=True, beam_width=20)

def is_chinese(string):
    """
    检查整个字符串是否为中文
    Args:
        string (str): 需要检查的字符串,包含空格也是False
    Return
        bool
    """
    if (len(string) <= 1): # 去除只有单个字或者为空的字符串
        return False

    for chart in string: # 把除了中文的所有字母、数字、符号去除
        if (chart < u'\u4e00' or chart > u'\u9fff'):
            return False

    return True

words = [] # 将符合标准的近义词保存在这里（标准：字符串为中文且长度大于1）

for item in range(20):
    # print(results[0][item])
    if (is_chinese(results[0][item])):
        words.append(results[0][item])

print(words)
strwords="".join(words)
words_dict=get_dict(strwords)
setwords=set(strwords)  #去重不必，以上结果已去重，只是顺序不同。
words_set=get_dict(setwords)
print("words_set=",words_set)
print("type(words_set)=",type(words_set))
print("len(words_set)=",len(words_set))

```

### 6.匹配诗词
- 匹配诗歌字数处理。


```python
#setwords
result=[(0,0)]
max_icount=[0]   
for j in range(len(dataset)):
	txt = []
	for i in range(L):
		txt += [id2char_dict[data[j,i]]]        
	icount=0
	# print("max_icount=",max_icount)
	for item in setwords:
		icount=icount+txt.count(item)
	# print("the total has found %d" %(icount))
	if max_icount[len(max_icount)-1]<icount:   
		max_icount.sort()
		result[0] = (j, icount)  # j,记录诗歌id。#icount记录字匹配量。
		result=sorted(result, key=lambda x: (x[1], x[0]))  #第2列优先第1列排序。
		# total_max_icount[0] = icount  #total是对图像每个分类词联想的汇总比较，如果只用1个分类词联想就不需要了，逻辑相同，自己补全也很方便。
		# total_result[0]  = (j, icount)
		# total_max_icount.sort()
		# total_result = sorted(total_result, key=lambda x: (x[1], x[0]))
	else:
		if max_icount[len(max_icount)-1]== icount and icount!=0: #与最后最大的那个相等的话，添加到最后。
			if max_icount[0] < icount:
				max_icount[0] = icount
				max_icount.sort()
				result[0] = (j, icount)  
				result = sorted(result, key=lambda x: (x[1], x[0]))  
				# total_max_icount[0] = icount
				# total_result[0] = (j, icount)
				# total_max_icount.sort()
				# total_result = sorted(total_result, key=lambda x: (x[1], x[0]))
			else:
				max_icount.append(icount)
				# total_max_icount.append(icount)
				if result[0][0] == 0 and result[0][1] == 0:
					result[0] = (j, icount)  # j,记录诗歌id。#icount记录字匹配量。
					# total_result[0] = (j, icount)
				else:
					result.append((j, icount))
					# total_result.append((j, icount))
	#print("total_max_icount=", total_max_icount)
print("max_icount=", max_icount)   
print("result=", result)


```

### 7.分析匹配诗的情感倾向
- 选取最积极且尽量匹配量大的诗词。

匹配诗以后，我们可以分析一下这首诗的情感，看一下是 积极的 还是 消极的 ，另一方面也可以体现出这张图片的情感倾向


```python
# 情感倾向分析
import paddlehub as hub

#如何前面获得total_result的话，可以把result改成total_result
senta = hub.Module(name="senta_bow")
    test_text =["{}".format(dataset[result[k][0]][4])]     #新手处理方式，估计还有更好的。
    print("test_text=", test_text)
    print("type(test_text)=", type(test_text))
    emotion_results = senta.sentiment_classify(texts=test_text, use_gpu=False, batch_size=1)
```


```python
#汇总的处理：即对total_result的处理。
for k in range(len(total_result)):
    #加上情感分析：
    senta = hub.Module(name="senta_bow")
    test_text =["{}".format(dataset[total_result[k][0]][4])]   
    emotion_results = senta.sentiment_classify(texts=test_text, use_gpu=False, batch_size=1)

    for emotion in emotion_results:
    # 选择一首情感最积极的吧。
        if max_positive<=emotion['positive_probs']:  #越后面，匹配度高，相等也替换了。
              max_positive=emotion['positive_probs']
              max_total_result=[total_result[k][0],total_result[k][1],emotion['sentiment_key'],emotion['positive_probs'],emotion['negative_probs']]

#最后输出最佳匹配的诗：
print("诗全部信息dataset[max_total_result[0]]=", dataset[max_total_result[0]])
print("匹配数量total_result[k]][1]=", max_total_result[1])
print("诗内容dataset[max_total_result[0]][4]=", dataset[max_total_result[0]][4])
print("该诗的情感倾向：", max_total_result[2])
print("积极的情感倾向概率：", max_total_result[3])
print("消极的情感倾向概率：", max_total_result[4])

```

## 四、总结与展望

本文用的方法适合初学者快速体验查图配诗的功能，有很多优化的空间、以及实用场景。步骤和简单的原理：<br>
* 提取图像里的关键信息
* 根据关键信息匹配诗词

该项目从paddlehub直接调用了4个模型，顺利地完成了查图配诗的功能，体现了paddlehub的易用性即可以简单快速地上手

最后，欢迎大家更换自己的图片，来试试匹配到哪首诗词，并不断优化自己配诗效果吧！

### 完整源代码链接：

- [Github项目链接](https://github.com/welwin0626/imgPoetry.git)：[https://github.com/welwin0626/imgPoetry.git](https://github.com/welwin0626/imgPoetry.git)
- [Gitee项目链接](https://gitee.com/wellwin/imgPoetry.git)：[https://gitee.com/wellwin/imgPoetry.git](https://gitee.com/wellwin/imgPoetry.git)

### 参考文章

> 两篇：向大佬致敬，纯属抄袭，不对，也可以说是活学活用吧。
- [还在担心发朋友圈没文案？快来试试看图写诗吧！](https://aistudio.baidu.com/aistudio/projectdetail/738634?channelType=0&channel=0)
- [浪迹天涯去写诗](https://aistudio.baidu.com/aistudio/projectdetail/1488249?channelType=0&channel=0)
