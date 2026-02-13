---
title: Uniclass Predictor Web
emoji: 🚀
colorFrom: pink
colorTo: yellow
sdk: docker
pinned: false
license: apache-2.0
short_description: Website  deploys the ML to predict Uniclass
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

Purpose:
This is the trial webapp that I have done during my Intern. The main purpose of this webapp is to deploy Deep Learning (Sentence Transformer) fine-tuning with few-shot learning (SetFitModel) framework to assign SS Uniclass Code from NBS Uniclass for each element in the designed IFC file based on the provided dataset.

Reason to choose current model:
The challenge that I faced with the given dataset is the imbalance of attributes. Some of the attributes take up to 62% of the training dataset. I have apply some approaches like applying resampling methods:using Hybrid method Undersampling (Tomek links) for dominant attributes and oversampling (SMOTE: Synthetic minorityover-sampling technique) for rare attributes. Then for the Algorithm-level approaches I have apply Cost-Sensitive Learning. However, within the dataset theere are some attributes just appear once within the whole dataset and my supervisor still want to predict those calue so I tried to apply one-shot learning and few-shot learning framework to fine-tune the Deep Learning (Sentence Tranformer) to be more inclusive. 

Web's Methodology:
The Web accept .xlsx or .csv file with the extracted parameters from a designed .ifc file. Then it will search for the parameters that have been saved in the training model after I applied statistics to decide which parameters matter for the task, group them together into a jsonb column then assign the SS Uniclass Code. The model give back the top 1 probabilities that is closest the data point in the training dataset and another column for the confidence_score

Things that can be improved:
The web app is pretty sensitive for any changes of the parameters' names so it has to be exactly similar names among the given spreadsheet with the trained parameters. 
The model is not actually performing as well. In the training stage, I have suffer from overfitting because I have used a lot of complicated methods for the given classification tasks. The accuracy score in the evaluation matrix was around 0.99 in the validation set. The model actually remembers to whole training data instead of actually learn the patterns so it perform poorly in the real-life tasks.
