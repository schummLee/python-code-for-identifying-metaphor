import nltk
import numpy as np
from gensim.models import KeyedVectors
import pandas as pd

# 准备 Word2Vec 模型
# load Word2Vec model
model = KeyedVectors.load_word2vec_format(r"C:\Users\de'l'l\Desktop\test0\bnc_lower.bin", binary=True)

# save the Word2Vec model in binary format
model.save(r"C:\Users\de'l'l\Desktop\test0\bnc_lower.bin")


# 定义形容词和名词的词性标记
adj_tags = {'JJ', 'JJR', 'JJS'}
noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}

# 计算两个词向量的相似度
def similarity(w1, w2):
    try:
        return model.similarity(w1, w2)
    except KeyError:
        return 0

# 检测隐喻的主函数
def detect_metaphor(text, similarity_threshold=0.3):
    # 对文本进行预处理
    text = text.lower()
    sentences = nltk.sent_tokenize(text)
    results = []
    
    for sentence in sentences:
        # 分词并标记词性
        words = nltk.word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)

        # 提取形容词和名词的组合
        adj_n_pairs = [(tagged_words[i][0], tagged_words[i+1][0])
                       for i in range(len(tagged_words)-1)
                       if tagged_words[i][1] in adj_tags and tagged_words[i+1][1] in noun_tags]

        # 计算每个形容词和名词的组合的相似度
        valid_pairs = [(pair, similarity(pair[0], pair[1]))
                       for pair in adj_n_pairs]

        # 如果至少有一个合法的组合且相似度小于给定阈值，我们认为该句子包含隐喻
        if valid_pairs and max(valid_pairs, key=lambda x: x[1])[1] < similarity_threshold:
            metaphor_pair = max(valid_pairs, key=lambda x: x[1])[0]
            if metaphor_pair[0] != metaphor_pair[1] and metaphor_pair[1] != '.':
                results.append({'sentence': sentence, 'metaphor': metaphor_pair})

    return pd.DataFrame(results, columns=['sentence', 'metaphor'])

# 读取文件
filename = 'sentence-LloydList.txt'
with open(r"C:\Users\de'l'l\Desktop\test0\\" + filename, 'r', encoding='utf-8') as f:
    text = f.read()



# identify metaphors in text and add to DataFrame
results_df = detect_metaphor(text)

# Define the file path
file_path = 'finalmetaphors.xlsx'



# Write the DataFrame to Excel
with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
    # Write the DataFrame to a sheet named 'Sheet1'
    results_df.to_excel(writer, sheet_name='Sheet1', index=False)