"""
@auther: pkusp
@contact: pkusp@outlook.com
@description: kaggle toxic classification competition exercise
@ref: kaggle kernel
@time: 2018.03
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import re, string


class NbSvm(object):
	def __init__(self,train,test):
		self.train = train
		self.test = test
	def data_check(self):
		train = deepcopy(self.train)
		test = deepcopy(self.test)
		print(train.head())
		print(train['comment_text'][0])
		lens = train.comment_text.str.len()
		print(lens.mean(),lens.std(),lens.max())
		print(lens.hist()) # hist 
		label_cols = ['toxic', 'severe_toxic', 'obscene',
					'threat', 'insult', 'identity_hate']
		print(train['toxic'].head())
		print(train[label_cols].max(axis=1).head())
		print("lens train:",len(train))
		print("lens test:",len(test))
	def data_pre(self):
		COMMENT = 'comment_text'
		train_cmt = self.train[COMMENT].fillna("unknown",inplace=True)
		test_cmt = self.test[COMMENT].fillna("unknown",inplace=True)
		return train_cmt,test_cmt
	def tokenize(self,s):
		re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
		return re_token.sub(r'\1',s).split()
	def tf_idf(self):
		train_cmt,test_cmt = self.data_pre()
		vec = TfidfVectorizer(ngram_range=(1,2),tokenizer=tokenize,
			min_df=3,max_df=0.9,strip_accents='unicode',use_id=1,
			smooth_idf=1,sublinear=1)
		train_term_doc = vec.fit_transform(train_cmt)
		test_term_doc = vec.transform(test_cmt)
		return train_term_doc,test_term_doc

	# def pr(self,y_i,y):
	# 	p = x[y==y_i].sum(0)
	# 	return (p+1)/((y==y_i).sun()+1)
	# def get_mdl(self,y):
	# 	y = y.values
	# 	r = np.log(pr(1,y)/pr(0,y))
	# 	m = LogisticRegression(C=4,dual=True)
	# 	x_nb = x.mutiply(r)
	# 	return m.fit(x_nb,y),r

	# def pred(self):
	# 	preds = np.zeros((len(test), len(label_cols)))
	# 	for i, j in enumerate(label_cols):
	# 	    print('fit', j)
	# 	    m,r = get_mdl(train[j])
	# 	    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
	# def submit(self):
	# 	submid = pd.DataFrame({'id': subm["id"]})
	# 	submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
	# 	submission.to_csv('submission.csv', index=False)



if __name__ == '__main__':
	train = pd.read_csv('../../data_raw/train.csv')
	test = pd.read_csv('../../data_raw/test.csv')
	subm = pd.read_csv('../../data_raw/sample_submission.csv')
	ns = NbSvm(train,test)
	ns.data_check()
