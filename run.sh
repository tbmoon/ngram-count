#!/bin/bash

# To save log file: 
# ./run.sh &> log.txt &

datasets=("ted" "news")
data_dir="../nlp_preprocessing/data"
out_dir="."

# 1) Two language models will be separately generated with N-gram using each data, ted and news.
#    One can evaluate and compare results from perplexity (ppl).
for dataset in ${datasets[@]}
do
	# Language Model Generation.
	time ngram-count -order 3 -kndiscount -text ${data_dir}/${dataset}.aligned.en.refined.tok.bpe.txt -lm ${out_dir}/${dataset}.aligned.en.refined.tok.bpe.lm -write-vocab ${out_dir}/${dataset}.aligned.en.refined.tok.bpe.vocab.txt -debug 2

	# Language Model Evaluation.
	ngram -ppl ${data_dir}/test.en.tok.bpe.txt -lm ${out_dir}/${dataset}.aligned.en.refined.tok.bpe.lm -order 3 -debug 2

	# Sentence Generation.
	ngram -lm ${out_dir}/${dataset}.aligned.en.refined.tok.bpe.lm -gen 10 | python
done

# 2) Try combined models from ted and news.
ngram -lm ${out_dir}/news.aligned.en.refined.tok.bpe.lm -mix-lm ${out_dir}/ted.aligned.en.refined.tok.bpe.lm -lambda .5 -write-lm ${out_dir}/news_ted.aligned.en.refined.tok.bpe.lm -debug 2
ngram -ppl ${data_dir}/test.en.tok.bpe.txt -lm ${out_dir}/news_ted.aligned.en.refined.tok.bpe.lm -order 3 -debug 2

# 3) Try different hyper-parameter, labmda .3.
ngram -lm ${out_dir}/news.aligned.en.refined.tok.bpe.lm -mix-lm ${out_dir}/ted.aligned.en.refined.tok.bpe.lm -lambda .3 -write-lm ${out_dir}/news_ted.aligned.en.refined.tok.bpe.lm -debug 2
ngram -ppl ${data_dir}/test.en.tok.bpe.txt -lm ${out_dir}/news_ted.aligned.en.refined.tok.bpe.lm -order 3 -debug 2
