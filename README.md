# Kaggle Toxicity Classification

Competitions:[Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)  
Rank:  
Note: This is a code backup, it's not runable due to the difference file path  



## Architectural
<img src="./toxicity_architectural.png">


## File Discribe
```
----- on-line-inference
  |            |
  |             ----- toxicity-final-inference.ipynb: Kaggle kernel final submission file
  |            |
  |             ----- toxicity_final_inference.py: convert .ipynb to .py
  |
   -- off-line-train
               |
                ----- features: use to eatract DL features
               |            |
               |             ----- dl_train_cv.py: train LSTM models
               |            |
               |             ----- dl_inference.py: extact LSTM features & predict
               |            |
               |             ----- dl_bert_train.py: train BERT models
               |            |
               |             ----- dl_bert_inferecne.py: extact BERT features & predict
               | 
                ----- emsemble: 1.train and inference on different model 2.ensemble predictions
               |            | 
               |             ----- nn_train_inference.py: NN train & predict
               |            | 
               |             ----- lgb_train_inference.py: LGB train & predict
               |            | 
               |             ----- random_forest_train_inference.py: RF train & predict
               |            | 
               |             ----- xgb_train_inference.py: XGB train & predict
               |            | 
               |             ----- ensemble.py: custom ensemble predictions 
               |
                ----- text_process5.py: text preprocess
               |
                ----- models.py: Pytorch DL models 
               |
                ----- dataset_helper.py: custom DataLoader
               |
                ----- global_variable.py: global_variable
               |
                ----- utils.py
```

## External Data  

#### Embedding:  
__glove_embedding:__ https://www.kaggle.com/takuok/glove840b300dtxt
__crawl_embedding:__ https://www.kaggle.com/yekenot/fasttext-crawl-300d-2m

#### BERT model:
__tf BERT:__ https://www.kaggle.com/maxjeblick/bert-pretrained-models
__convert2Pytorch:__ https://github.com/huggingface/pytorch-pretrained-BERT

#### Others
__apex(use for speed up training):__ https://www.kaggle.com/gabrichy/nvidiaapex