from transformers import BertForMaskedLM, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer, DistilBertForMaskedLM, DistilBertTokenizer, RobertaForMaskedLM, RobertaTokenizer
import torch
import pandas as pd
from tqdm import tqdm
import math
import operator
from heapq import nlargest
import argparse
import pickle
import time

parser = argparse.ArgumentParser()
parser.add_argument('--proc-id', type=int)
parser.add_argument('--model', type=str, choices=['bert', 'distilbert', 'roberta'])
parser.add_argument('--dataset', type=str, choices=['news', 'twitter', 'wiki'])
args = parser.parse_args()


def get_all_texts(file):
    df = pd.read_csv(file, header=None)
    texts = df[2].to_list()
    texts = [t.replace("\\", " ") for t in texts]
    print(texts[:10])

    return texts



def get_all_tweets():
    df = pd.read_csv("tweets.csv", header=None, on_bad_lines='skip', encoding='ISO-8859-1')
    texts = df[5].to_list()[:800000]
    texts = [t for t in texts if len(t) > 65][:300000]
    print("length", len(texts))
    print(texts[:10])

    return texts


def get_all_texts_wikitext(split = "train"):

    with open('wikitext.txt') as f:
        sentences = f.readlines()

    if split == "train":
        sentences = sentences[:100000]

    elif split == "test":
        sentences = sentences[100000:200000]
    
    elif split == "alt":
        sentences = sentences[200000:300000]

    sentences = [s for s in sentences if len(s) > 25][:5000]

    return sentences




attack_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
attack_tokenizer.pad_token = attack_tokenizer.eos_token

if args.dataset == 'twitter':
    attack_model = GPT2LMHeadModel.from_pretrained('<path_to_attack_model_twitter>')
elif args.dataset == 'news':
    attack_model = GPT2LMHeadModel.from_pretrained('<path_to_attack_model_news>')
elif args.dataset == 'wiki':
    attack_model = GPT2LMHeadModel.from_pretrained('<path_to_attack_model_wiki>')

attack_model = attack_model.to('cuda:0')

if args.model == 'bert':
    search_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    search_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

elif args.model == 'distilbert':
    search_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    search_model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

elif args.model == 'roberta':
    search_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    search_model = RobertaForMaskedLM.from_pretrained('roberta-base')

print(search_model)

search_model = search_model.to('cuda:1')

token_dropout = torch.nn.Dropout(p=0.7)

if args.dataset == 'twitter':
    texts = get_all_tweets()
elif args.dataset == 'news':
    texts = get_all_texts('news.csv')
elif args.dataset == 'wiki':
    texts = get_all_texts_wikitext("train")[1200:1800]+get_all_texts_wikitext("alt")[1200:1800]



def generate_neighbours_alt(tokenized, num_word_changes=1):
    text_tokenized = search_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to('cuda:1')
    original_text = search_tokenizer.batch_decode(text_tokenized)[0]

    candidate_scores = dict()
    replacements = dict()

    for target_token_index in list(range(len(text_tokenized[0,:])))[1:]:

        target_token = text_tokenized[0,target_token_index]
        if args.model == 'bert':
            embeds = search_model.bert.embeddings(text_tokenized)
        elif args.model == 'distilbert':
            embeds = search_model.distilbert.embeddings(text_tokenized)
        elif args.model == 'roberta':
            embeds = search_model.roberta.embeddings(text_tokenized)
            
        embeds = torch.cat((embeds[:,:target_token_index,:], token_dropout(embeds[:,target_token_index,:]).unsqueeze(dim=0), embeds[:,target_token_index+1:,:]), dim=1)
        
        token_probs = torch.softmax(search_model(inputs_embeds=embeds).logits, dim=2)

        original_prob = token_probs[0,target_token_index, target_token]

        top_probabilities, top_candidates = torch.topk(token_probs[:,target_token_index,:], 6, dim=1)

        for cand, prob in zip(top_candidates[0], top_probabilities[0]):
            if not cand == target_token:

                #alt = torch.cat((text_tokenized[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to('cuda:1'), text_tokenized[:,target_token_index+1:]), dim=1)
                #alt_text = search_tokenizer.batch_decode(alt)[0]
                if original_prob.item() == 1:
                    print("probability is one!")
                    replacements[(target_token_index, cand)] = prob.item()/(1-0.9)
                else:
                    replacements[(target_token_index, cand)] = prob.item()/(1-original_prob.item())

    
    #highest_scored_texts = max(candidate_scores.iteritems(), key=operator.itemgetter(1))[:100]
    highest_scored_texts = nlargest(100, candidate_scores, key = candidate_scores.get)

    replacement_keys = nlargest(50, replacements, key=replacements.get)
    replacements_new = dict()
    for rk in replacement_keys:
        replacements_new[rk] = replacements[rk]
    
    replacements = replacements_new
    print("got highest scored single texts, will now collect doubles")

    highest_scored = nlargest(100, replacements, key=replacements.get)


    texts = []
    for single in highest_scored:
        alt = text_tokenized
        target_token_index, cand = single
        alt = torch.cat((alt[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to('cuda:1'), alt[:,target_token_index+1:]), dim=1)
        alt_text = search_tokenizer.batch_decode(alt)[0]
        texts.append((alt_text, replacements[single]))


    return texts




def generate_neighbours(tokenized, num_word_changes=1):
    text_tokenized = search_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to('cuda:1')
    original_text = search_tokenizer.batch_decode(text_tokenized)[0]

    candidate_scores = dict()
    replacements = dict()

    for target_token_index in list(range(len(text_tokenized[0,:])))[1:]:

        target_token = text_tokenized[0,target_token_index]
        embeds = search_model.bert.embeddings(text_tokenized)
        embeds = torch.cat((embeds[:,:target_token_index,:], token_dropout(embeds[:,target_token_index,:]).unsqueeze(dim=0), embeds[:,target_token_index+1:,:]), dim=1)
        
        token_probs = torch.softmax(search_model(inputs_embeds=embeds).logits, dim=2)

        original_prob = token_probs[0,target_token_index, target_token]

        top_probabilities, top_candidates = torch.topk(token_probs[:,target_token_index,:], 10, dim=1)

        for cand, prob in zip(top_candidates[0], top_probabilities[0]):
            if not cand == target_token:

                alt = torch.cat((text_tokenized[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to('cuda:1'), text_tokenized[:,target_token_index+1:]), dim=1)
                alt_text = search_tokenizer.batch_decode(alt)[0]
                candidate_scores[alt_text] = prob/(1-original_prob)
                replacements[(target_token_index, cand)] = prob/(1-original_prob)

    
    highest_scored_texts = nlargest(100, candidate_scores, key = candidate_scores.get)



    return highest_scored_texts



def get_logprob(text):
    text_tokenized = attack_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to('cuda:0')
    logprob = - attack_model(text_tokenized, labels=text_tokenized).loss.item()

    return logprob




def get_logprob_batch(text):
    text_tokenized = attack_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to('cuda:0')

    ce_loss = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=attack_tokenizer.pad_token_id)
    logits = attack_model(text_tokenized, labels=text_tokenized).logits[:,:-1,:].transpose(1,2)
    manual_logprob = - ce_loss(logits, text_tokenized[:,1:])
    mask = manual_logprob!=0
    manual_logprob_means = (manual_logprob*mask).sum(dim=1)/mask.sum(dim=1)


    return manual_logprob_means.tolist()


all_scores = []

if args.dataset == 'twitter':
    batch_size = 3000
elif args.dataset == 'news':
    batch_size = 1200
elif args.dataset == 'wiki':
    batch_size = 1200
for text in tqdm(texts[args.proc_id*batch_size:(args.proc_id+1)*batch_size]):
    attack_model.eval()
    search_model.eval()

    tok_orig = search_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to('cuda:1')
    orig_dec = search_tokenizer.batch_decode(tok_orig)[0].replace(" [SEP]", " ").replace("[CLS] ", " ")


    scores = dict()
    scores[f'<original_text>: {orig_dec}'] = get_logprob(orig_dec)

    with torch.no_grad():
        start = time.time()
        #neighbours = generate_neighbours(text)
        neighbours = generate_neighbours_alt(text)
        end = time.time()
        print("generating neighbours took seconds:", end-start)


        for i, neighbours in enumerate([one_word_neighbours]):
            neighbours_texts = []
            for n in neighbours:
                neighbours_texts.append((n[0].replace(" [SEP]", " ").replace("[CLS] ", " "), n[1]))
                score = get_logprob_batch([n[0].replace(" [SEP]", " ").replace("[CLS] ", " ")])
                scores[n] = score


            if i == 0:
                scores_temp = scores        
    
    all_scores.append(scores_temp)

with open(f'all_scores_{args.dataset}_{args.model}_{args.proc_id}.pkl', 'wb') as file:
    pickle.dump(all_scores, file)


all_scores = []

for text in tqdm(texts[args.proc_id*1200:(args.proc_id+1)*1200]):
    attack_model.eval()
    search_model.eval()

    tok_orig = search_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to('cuda:1')
    orig_dec = search_tokenizer.batch_decode(tok_orig)[0].replace(" [SEP]", " ").replace("[CLS] ", " ")


    scores = dict()
    scores[f'<original_text>: {orig_dec}'] = get_logprob(orig_dec)

    with torch.no_grad():
        neighbours = generate_neighbours(text)

        for n in neighbours:
            n = n.replace(" [SEP]", " ").replace("[CLS] ", " ")
            scores[n] = get_logprob(n)

    all_scores.append(scores)



with open(f'all_scores{args.proc_id}.pkl', 'wb') as file:
    pickle.dump(all_scores, file)
