# -*- coding: utf-8 -*-
#import ptvsd 

#ptvsd.enable_attach(address =('0.0.0.0',8848))
#ptvsd.wait_for_attach()

import pandas as pd
import os
import math
import torch
import pandas as pd
import numpy as np
import argparse
import torch.nn.functional as F
import torch.nn as nn
import pickle
from tqdm.auto import tqdm
import tqdm
from module_CL_F1 import PETER
from utils import rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens, unique_sentence_percent, \
    root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity


parser = argparse.ArgumentParser(description='PErsonalized Transformer for Explainable Recommendation (PETER)')
parser.add_argument('--data_path', type=str, default='./music',
                    help='path for loading the pickle data')
parser.add_argument('--index_dir', type=str, default='1',
                    help='load indexes')
parser.add_argument('--emsize', type=int, default=384,
                    help='size of embeddings')
parser.add_argument('--nhead', type=int, default=8,
                    help='the number of heads in the transformer')
parser.add_argument('--nhid', type=int, default=5096,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=24,
                    help='number of layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--lr', type=float, default=1.0,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch size')
parser.add_argument('--seed', type=int, default=2022,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',default="True",
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--checkpoint', type=str, default='./peter/',
                    help='directory to save the final model')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--vocab_size', type=int, default=20000,
                    help='keep the most frequent words in the dict')
parser.add_argument('--endure_times', type=int, default=5,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--rating_reg', type=float, default=0.2,
                    help='regularization on recommendation task')
parser.add_argument('--cl_reg', type=float, default=0.2,
                    help='contrast loss reg')
parser.add_argument('--context_reg', type=float, default=1.0,
                    help='regularization on context prediction task')
parser.add_argument('--text_reg', type=float, default=1.0,
                    help='regularization on text generation task')
parser.add_argument('--peter_mask', action='store_true',default=True,
                    help='True to use peter mask; Otherwise left-to-right mask')
parser.add_argument('--use_feature', action='store_true',default=True,
                    help='False: no feature; True: use the feature')
parser.add_argument('--output', type=str, default='lfm',
                    help='预测用的模块 mlp、fm、lfm、nfm')
parser.add_argument('--words', type=int, default=15,
                    help='number of words to generate for each sample')
parser.add_argument('--stage', type=str, default='test',
                    help='number of words to generate for each sample')
parser.add_argument('--model_path', type=str, default='/home/zjw/demo01/peter/model.pt',
                    help='number of words to generate for each sample')
args = parser.parse_args()

# if args.data_path is None:
#     parser.error('--data_path should be provided for loading data')
# if args.index_dir is None:
#     parser.error('--index_dir should be provided for loading data splits')

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda:0' if args.cuda else 'cpu')
sss=now_time()
model_dir=args.checkpoint
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, 'model.pt')
prediction_path = os.path.join(model_dir, args.outf)

###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading data')
# with open('./user_reviews_emb.pkl','rb') as f:
#     user_emb=pickle.load(f)
# with open('./item_reviews_emb.pkl','rb') as f:
#     item_emb=pickle.load(f)
# temp_u_dict={}
# temp_u=0
# for i in range(len(user_emb)):
#     for j in range(len(user_emb[i])):
#         temp_u=user_emb[i][j]+temp_u
#     temp_u=temp_u/len(user_emb[i])
#     temp_u_dict[i]=temp_u
# temp_i_dict={}
# temp_i=0
# for i in range(len(item_emb)):
#     for j in range(len(item_emb[i])):
#         temp_i=item_emb[i][j]+temp_i
#     temp_i=temp_i/len(item_emb[i])
#     temp_i_dict[i]=temp_i
# pd.to_pickle(temp_u_dict,'temp_u_dict.pkl')
# pd.to_pickle(temp_i_dict,'temp_i_dict.pkl')
# corpus = DataLoader(args.data_path, args.index_dir, args.vocab_size)
# with open('cells_corpus.pkl', "wb") as fOut:
#     pickle.dump(corpus, fOut)
with open('/home/zjw/demo01/cells_corpus_e.pkl','rb') as f:
    corpus=pickle.load(f)
word2idx = corpus.word_dict.word2idx
idx2word = corpus.word_dict.idx2word
# feature_set = corpus.feature_set

user_aspect_top2=torch.load('/home/zjw/demo01/cell_aspect_top2.pt')
# user_aspect_top2=pd.read_pickle("/home/wangshuo/dataset/data/TripAdvisor/tirp_user_aspects_top2.pkl")
print(now_time() + 'Loading done')
# user_aspect_top2=pd.read_pickle('yelp_aspect_retrive_top2.pkl')
train_data = Batchify(corpus.train, word2idx, args.words, args.batch_size, shuffle=True,user_aspect=user_aspect_top2)
val_data = Batchify(corpus.test, word2idx, args.words, args.batch_size,user_aspect=user_aspect_top2)
test_data = Batchify(corpus.test, word2idx, args.words, args.batch_size,user_aspect=user_aspect_top2)
print(now_time() + 'Splitt done')
###############################################################################
# Build the model
###############################################################################

if args.use_feature:
    src_len = 2+2   # [u, i, f]
else:
    src_len = 2  # [u, i]
tgt_len = args.words + 1  # added <bos> or <eos>
ntokens = len(corpus.word_dict)
nuser = len(corpus.user_dict.idx2entity)
nitem = len(corpus.item_dict.idx2entity)
pad_idx = word2idx['<pad>']
# with open('./videos/video_emb.pkl', "rb") as fIn:
#     corpus_emb = pickle.load(fIn)
model = PETER(args.peter_mask, src_len, tgt_len, pad_idx, nuser, nitem, ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, word2idx, idx2word, device, args.output, args.dropout).to(device)
text_criterion = nn.NLLLoss(ignore_index=pad_idx)  # ignore the padding when computing loss
rating_criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.25)

###############################################################################
# Training code
###############################################################################


def predict(log_context_dis, topk):
    word_prob = log_context_dis.exp()  # (batch_size, ntoken)
    if topk == 1:
        context = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1)
    else:
        context = torch.topk(word_prob, topk, 1)[1]  # (batch_size, topk)
    return context  # (batch_size, topk)


def train(data,user_retrive_global,item_retrive_global):
    # Turn on training mode which enables dropout.
    model.train()
    context_loss = 0.
    text_loss = 0.
    rating_loss = 0.
    contrast_cl_loss =0.
    sigel_loss = 0.0

    total_sample = 0


    while True:
        user, item, rating, seq,aspect= data.next_batch()  # (batch_size, seq_len), data.step += 1
        batch_size = user.size(0)
        user = user.to(device)  # (batch_size,)
        item = item.to(device)
        rating = rating.to(device)
        seq = seq.t().to(device)  # (tgt_len + 1, batch_size)
        aspect = aspect.t().to(device)  # (1, batch_size)
        if args.use_feature:
            text = torch.cat([aspect, seq[:-1]], 0)  # (src_len + tgt_len - 2, batch_size)
        else:
            text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        log_word_prob, \
        log_context_dis, rating_p, atten, rating_vec, embeddings = model(user, item, text,user_retrive_global=user_retrive_global,item_retrive_global=item_retrive_global)  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
        context_dis = log_context_dis.unsqueeze(0).repeat((tgt_len - 1, 1, 1))  # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)
        c_loss = text_criterion(context_dis.view(-1, ntokens), seq[1:-1].reshape((-1,)))
        r_loss = rating_criterion(rating_p, rating)
        t_loss = text_criterion(log_word_prob.view(-1, ntokens), seq[1:].reshape((-1,)))
        cl_loss = model.contrast_loss(rating_vec,log_word_prob)
        # sig_loss = model.sigel_loss(embeddings[0],embeddings[1])
        # loss = args.text_reg * t_loss + args.context_reg * c_loss + args.rating_reg * r_loss + args.cl_reg * cl_loss +0.001 * sig_loss
        loss = args.text_reg * t_loss + args.context_reg * c_loss + args.rating_reg * r_loss + args.cl_reg * cl_loss 
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        context_loss += batch_size * c_loss.item()
        text_loss += batch_size * t_loss.item()
        rating_loss += batch_size * r_loss.item()
        contrast_cl_loss += batch_size * cl_loss.item()
        # sigel_loss += batch_size * sig_loss.item()

        total_sample += batch_size

        if data.step % args.log_interval == 0 or data.step == data.total_step:
            cur_c_loss = context_loss / total_sample
            cur_t_loss = text_loss / total_sample
            cur_r_loss = rating_loss / total_sample
            cur_cl_loss = contrast_cl_loss / total_sample
            cur_sig_loss = sigel_loss / total_sample
            print(now_time() + 'context ppl {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} | contrast loss {:4.4f} |sig-contrast loss {:4.4f} | {:5d}/{:5d} batches'.format(
                math.exp(cur_c_loss), math.exp(cur_t_loss), cur_r_loss, cur_cl_loss, cur_sig_loss, data.step, data.total_step))
            context_loss = 0.
            text_loss = 0.
            rating_loss = 0.
            contrast_cl_loss = 0.
            sigel_loss = 0.
            total_sample = 0
        if data.step == data.total_step:
            break


def evaluate(data,user_retrive_global,item_retrive_global):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    context_loss = 0.
    text_loss = 0.
    rating_loss = 0.
    contrast_cl_loss = 0.
    sigel_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            user, item, rating, seq,aspect = data.next_batch()  # (batch_size, seq_len), data.step += 1
            batch_size = user.size(0)
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            rating = rating.to(device)
            seq = seq.t().to(device)  # (tgt_len + 1, batch_size)
  # (1, batch_size)
            aspect = aspect.t().to(device)  # (1, batch_size)
            if args.use_feature:
                text = torch.cat([aspect, seq[:-1]], 0)  # (src_len + tgt_len - 2, batch_size)
            else:
                text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)
            log_word_prob, log_context_dis, rating_p, atten, rating_vec, embeddings = model(user, item, text,user_retrive_global=user_retrive_global,item_retrive_global=item_retrive_global)  # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
            context_dis = log_context_dis.unsqueeze(0).repeat((tgt_len - 1, 1, 1))  # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)
            c_loss = text_criterion(context_dis.view(-1, ntokens), seq[1:-1].reshape((-1,)))
            r_loss = rating_criterion(rating_p, rating)
            t_loss = text_criterion(log_word_prob.view(-1, ntokens), seq[1:].reshape((-1,)))
            cl_loss = model.contrast_loss(rating_vec,log_word_prob)
            # sig_loss = model.sigel_loss(embeddings[0],embeddings[1])

            context_loss += batch_size * c_loss.item()
            text_loss += batch_size * t_loss.item()
            rating_loss += batch_size * r_loss.item()
            contrast_cl_loss += batch_size * cl_loss.item()
            # sigel_loss += batch_size *  sig_loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break
    return context_loss / total_sample, text_loss / total_sample, rating_loss / total_sample, contrast_cl_loss/total_sample, sigel_loss / total_sample


def beam_search(model, user, item, text, beam_width, user_retrive_global=None, item_retrive_global=None):
    context_predict = []
    rating_predict = []
    # 初始步骤
    batch_now = len(text[0])
    _, log_context_dis, rating_p, _, _, _ = model(user, item, text, False,
                                                                user_retrive_global=user_retrive_global,
                                                                item_retrive_global=item_retrive_global)
    rating_predict.extend(rating_p.tolist())
    context = predict(log_context_dis, topk=args.words)
    context_predict.extend(context.tolist())
    text_result = text.t().repeat_interleave(beam_width, dim=0)   # (beam_width*batch_size, 3)
    score = torch.zeros([beam_width*batch_now,1], dtype=torch.float, device=device)  # (beam_width*batch_size,1), all element is zero
    for idx in range(args.words):
        if idx == 0:
            # ATTENTION:put into the model is 'text', dim:(3, batch_size). log_word_prob:(batch_size, vocab_size)
            log_word_prob, _, _, _, _, _ = model(user, item, text, False,user_retrive_global=user_retrive_global,item_retrive_global=item_retrive_global)
            top_candidates = torch.topk(log_word_prob, beam_width, dim=1)   # indices:(beam_width*batch_size, beam_width) values: (beam_width*batch_size, beam_width)
            text_result = torch.cat([text_result, top_candidates.indices.view(-1,1)], dim=1)   # each item select the best 'beam_width'，form into text_result, dim:(batch_size*beam_width, 4)
            score = score + top_candidates.values.view(-1,1)   # (batch_size*beam_width, 1)
        else:
            text_chunks = torch.split(text_result, batch_now, dim=0)     # model only accept the size is ( , 256), so, split the text_result by row, 256 row is a chunk
            log_word_prob = torch.tensor([])
            for chunk in text_chunks:
                log_word_prob_split, _, _, _, _, _ = model(user, item, chunk.t(), False, False, False, user_retrive_global=user_retrive_global, item_retrive_global=item_retrive_global)   # (batch_size, vocab_size)
                if log_word_prob.numel() == 0:
                    log_word_prob = log_word_prob_split
                else:
                    log_word_prob = torch.cat([log_word_prob, log_word_prob_split], dim=0)   # combine all the probablity, dim:(batch_size*beam_width, vocab_size)
            log_word_prob = score.expand(-1, len(log_word_prob[0,:])) + log_word_prob
            log_beam_prob = log_word_prob.view(batch_now, len(log_word_prob[0,:])*beam_width)   # Sequence Extension, dim:(batch_size, beam_width*vocab_size)
            top_candidates = torch.topk(log_beam_prob, beam_width, dim=1)  # for each item, select the best 'beam_width'， of course, each element has contained the best result for previou step, dim:(batch_size, beam_width)
            current_token_select = top_candidates.indices // len(log_word_prob[0,:])  # rounding for index, we can get the best 'beam_width' for previous step, which has best score for vocabulary_list now, dim:(batch_size, beam_width)
            next_token_select = top_candidates.indices % len(log_word_prob[0,:])  # remainder for index, we can get the next token, dim:(batch_size, beam_width)
            
            index = 0
            text_result1 = []
            for beam_idx in range(len(current_token_select)):
                for beam_idy in range(len(current_token_select[0, :])):
                    text_select = torch.cat([text_result[current_token_select[beam_idx,beam_idy] + index * beam_width], next_token_select[beam_idx, beam_idy].unsqueeze(0)], dim=0)  # combine token, according to current_token_select and next_token_select， each element, dim:(seq_len, 1)
                    text_result1.append(text_select)  # combine all the seqence
                index += 1
            text_result = torch.stack(text_result1, dim=0)   # (batch_size*beam_width, seq_len)
            score = top_candidates.values.view(-1, 1)   # Logarithmic scores are summed, dim:(batch_size*beam_width, 1)
    text_final_result = []
    # Find the highest scoring per beam_width
    for max_idx in range(batch_now):
        max_value = float('-inf')
        index_select = 0
        for max_idy in range(beam_width):
            if score[max_idx*beam_width + max_idy, 0] > max_value:
                index_select = max_idy
                max_value = score[max_idx*beam_width + max_idy, 0]
        text_final_result.append(text_result[max_idx*beam_width+index_select])  # combine the best one for each item
    text_final_result = torch.stack(text_final_result, dim=0).t()   # (18, batch_size)
    return text_final_result, rating_predict, context_predict


def generate(data,user_retrive_global,item_retrive_global,beam_width):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    idss_predict = []
    context_predict = []
    rating_predict = []
    with torch.no_grad():
        pbar = tqdm.tqdm(total=data.total_step, desc="Processing")
        while True:
            user, item, rating, seq, aspect = data.next_batch()
            user = user.to(device)
            item = item.to(device)
            bos = seq[:, 0].unsqueeze(0).to(device)
            aspect = aspect.t().to(device)
            if args.use_feature:
                text = torch.cat([aspect, bos], 0)
            else:
                text = bos
            start_idx = text.size(0)
            
            generated_text, rating_predict1, context_predict1 = beam_search(model, user, item, text, beam_width, 
                                        user_retrive_global=user_retrive_global, item_retrive_global=item_retrive_global)
            rating_predict.extend(rating_predict1)
            context_predict.extend(context_predict1)
            ids = generated_text[start_idx:].t().tolist()
            idss_predict.extend(ids)
            pbar.update(1)
            #print(data.step)
            #if data.step == 10:
            if data.step == data.total_step:
                data.step = 0
                break
        pbar.close()

    # rating
    predicted_rating = [(r, p) for (r, p) in zip(data.rating.tolist(), rating_predict)]
    RMSE = root_mean_square_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    print(now_time() + 'RMSE {:7.4f}'.format(RMSE))
    MAE = mean_absolute_error(predicted_rating, corpus.max_rating, corpus.min_rating)
    print(now_time() + 'MAE {:7.4f}'.format(MAE))
    # text
    tokens_test = [ids2tokens(ids[1:], word2idx, idx2word) for ids in data.seq.tolist()]
    tokens_predict = [ids2tokens(ids, word2idx, idx2word) for ids in idss_predict]
    BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
    print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
    BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
    print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
    USR, USN = unique_sentence_percent(tokens_predict)
    print(now_time() + 'USR {:7.4f} | USN {:7}'.format(USR, USN))
    text_test = [' '.join(tokens) for tokens in tokens_test]
    text_predict = [' '.join(tokens) for tokens in tokens_predict]
    tokens_context = [' '.join([idx2word[i] for i in ids]) for ids in context_predict]
    ROUGE = rouge_score(text_test, text_predict)  # a dictionary
    for (k, v) in ROUGE.items():
        print(now_time() + '{} {:7.4f}'.format(k, v))
    text_out = ''
    with open('./peter/results.txt', 'a', encoding='utf-8') as file:
        file.write(f'{now_time()} num_beam={beam_width}\n')
        file.write('{:7.4f}\t'.format(BLEU1))
        file.write('{:7.4f}\t'.format(BLEU4))
        file.write('{:7.4F}\t'.format(USR))
        file.write('{:7}\t'.format(USN))
        for (k,v) in ROUGE.items():
            file.write('{:7.4f}\t'.format(v))
        file.write('\n')

    for (real, ctx, fake) in zip(text_test, tokens_context, text_predict):
        text_out += '{}\n{}\n{}\n\n'.format(real, ctx, fake)
    return text_out


# Loop over epochs.
best_val_loss = float('inf')
endure_count = 0
# user_retrive=torch.load('../data/TripAdvisor/retrive_result/user_revemb_By_item_retrive.pt')
# item_retrive=torch.load('../data/TripAdvisor/retrive_result/item_revemb_By_user_retrive.pt')
user_retrive_global=torch.load('/home/zjw/demo01/dataset/data/cells/retrive_result/user_glabos_retrive.pt')
item_retrive_global=torch.load('/home/zjw/demo01/dataset/data/cells/retrive_result/item_glabos_retrive.pt')
# user_retrive_global=torch.load('/home/wangshuo/dataset/data/TripAdvisor/retrive_result/user_glabos_retrive.pt')
# item_retrive_global=torch.load('/home/wangshuo/dataset/data/TripAdvisor/retrive_result/item_glabos_retrive.pt')
user_retrive_global = user_retrive_global.to(device)
item_retrive_global = item_retrive_global.to(device)

if args.stage == 'test':
    print('Enter the stage of test\n')
    with open(args.model_path, 'rb') as f:
        model = torch.load(f).to(device)
    for beam_width in range(6,21):
        text_o = generate(test_data,user_retrive_global,item_retrive_global, beam_width)
        save_result_name = "./peter/generate_{}.txt".format(beam_width)
        with open(save_result_name, 'w', encoding='utf-8') as file:
            file.write(text_o)
        print(f'{now_time()}Generated text saved, num_beam={beam_width}')
    exit(0)

for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train(train_data,user_retrive_global,item_retrive_global)
    val_c_loss, val_t_loss, val_r_loss, val_cl_loss, val_sig_loss = evaluate(val_data,user_retrive_global,item_retrive_global)
    # generate(test_data, user_retrive, item_retrive, user_retrive_global, item_retrive_global)
    if args.rating_reg == 0:
        val_loss = val_t_loss + val_cl_loss + val_sig_loss
    else:
        val_loss = val_t_loss + val_r_loss + val_cl_loss +val_sig_loss
    print(now_time() + 'context ppl {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} | contrast loss {:4.4f} | sig-contrast loss {:4.4f} | valid loss {:4.4f} on validation'.format(
        math.exp(val_c_loss), math.exp(val_t_loss), val_r_loss, val_cl_loss, val_sig_loss, val_loss))
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            # torch.save(model.user_embeddings,'userid_emb.pt')
            # torch.save(model.item_embeddings,'itemid_emb.pt')
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        scheduler.step()
        print(now_time() + 'Learning rate set to {:2.8f}'.format(scheduler.get_last_lr()[0]))

#Load the best saved model
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)
#
# # Run on test data.
# test_c_loss, test_t_loss, test_r_loss = evaluate(test_data,user_retrive_global,item_retrive_global)
# print('=' * 89)
# print(now_time() + 'context ppl {:4.4f} | text ppl {:4.4f} | rating loss {:4.4f} on test | End of training'.format(
#     math.exp(test_c_loss), math.exp(test_t_loss), test_r_loss))

print(now_time() + 'Generating text')
text_o = generate(test_data,user_retrive_global,item_retrive_global)
with open(prediction_path, 'w', encoding='utf-8') as f:
    f.write(text_o)
print(now_time() + 'Generated text saved to ({})'.format(prediction_path))
