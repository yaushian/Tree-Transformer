import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import subprocess
from models import *
from utils import *
from parse import *
import random
from bert_optimizer import BertAdam

class Solver():
    def __init__(self, args):
        self.args = args

        self.model_dir = make_save_dir(args.model_dir)
        self.no_cuda = args.no_cuda
        if not os.path.exists(os.path.join(self.model_dir,'code')):
            os.makedirs(os.path.join(self.model_dir,'code'))
        
        self.data_utils = data_utils(args)
        self.model = self._make_model(self.data_utils.vocab_size, 10)

        self.test_vecs = None
        self.test_masked_lm_input = []


    def _make_model(self, vocab_size, N=10, 
            d_model=512, d_ff=2048, h=8, dropout=0.1):
            
            "Helper: Construct a model from hyperparameters."
            c = copy.deepcopy
            attn = MultiHeadedAttention(h, d_model, no_cuda=self.no_cuda)
            group_attn = GroupAttention(d_model, no_cuda=self.no_cuda)
            ff = PositionwiseFeedForward(d_model, d_ff, dropout)
            position = PositionalEncoding(d_model, dropout)
            word_embed = nn.Sequential(Embeddings(d_model, vocab_size), c(position))
            model = Encoder(EncoderLayer(d_model, c(attn), c(ff), group_attn, dropout), 
                    N, d_model, vocab_size, c(word_embed))
            
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform(p)
            if self.no_cuda:
                return model
            else:
                return model.cuda()


    def train(self):
        if self.args.load:
            path = os.path.join(self.model_dir, 'model.pth')
            self.model.load_state_dict(torch.load(path)['state_dict'])
        tt = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                #print(name)
                ttt = 1
                for  s in param.data.size():
                    ttt *= s
                tt += ttt
        print('total_param_num:',tt)

        data_yielder = self.data_utils.train_data_yielder()
        optim = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
        #optim = BertAdam(self.model.parameters(), lr=1e-4)
        
        total_loss = []
        start = time.time()
        total_step_time = 0.
        total_masked = 0.
        total_token = 0.

        for step in range(self.args.num_step):
            self.model.train()
            batch = data_yielder.__next__()
            
            step_start = time.time()
            out,break_probs = self.model.forward(batch['input'].long(), batch['input_mask'])
            
            loss = self.model.masked_lm_loss(out, batch['target_vec'].long())
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.5)
            optim.step()

            total_loss.append(loss.detach().cpu().numpy())

            total_step_time += time.time() - step_start
            
            if step % 200 == 1:
                elapsed = time.time() - start
                print("Epoch Step: %d Loss: %f Total Time: %f Step Time: %f" %
                        (step, np.mean(total_loss), elapsed, total_step_time))
                self.model.train()
                print()
                start = time.time()
                total_loss = []
                total_step_time = 0.


            if step % 1000 == 0:
                print('saving!!!!')
                
                model_name = 'model.pth'
                state = {'step': step, 'state_dict': self.model.state_dict()}
                torch.save(state, os.path.join(self.model_dir, model_name))


    def test(self, threshold=0.8):
        path = os.path.join(self.model_dir, 'model.pth')
        self.model.load_state_dict(torch.load(path)['state_dict'])
        self.model.eval()
        txts = get_test(self.args.test_path)

        vecs = [self.data_utils.text2id(txt, 60) for txt in txts]
        masks = [np.expand_dims(v != 0, -2).astype(np.int32) for v in vecs]
        self.test_vecs = cc(vecs, no_cuda=self.no_cuda).long()
        self.test_masks = cc(masks, no_cuda=self.no_cuda)
        self.test_txts = txts

        self.write_parse_tree()


    def write_parse_tree(self, threshold=0.8):
        batch_size = self.args.batch_size

        result_dir = os.path.join(self.model_dir, 'result/')
        make_save_dir(result_dir)
        f_b = open(os.path.join(result_dir,'brackets.json'),'w')
        f_t = open(os.path.join(result_dir,'tree.txt'),'w')
        for b_id in range(int(len(self.test_txts)/batch_size)+1):
            out,break_probs = self.model.forward(self.test_vecs[b_id*batch_size:(b_id+1)*batch_size], 
                                                 self.test_masks[b_id*batch_size:(b_id+1)*batch_size])
            for i in range(len(self.test_txts[b_id*batch_size:(b_id+1)*batch_size])):
                length = len(self.test_txts[b_id*batch_size+i].strip().split())

                bp = get_break_prob(break_probs[i])[:,1:length]
                model_out = build_tree(bp, 9, 0, length-1, threshold)
                if (0, length) in model_out:
                    model_out.remove((0, length))
                if length < 2:
                    model_out = set()
                f_b.write(json.dumps(list(model_out))+'\n')

                """
                overlap = model_out.intersection(std_out)
                prec = float(len(overlap)) / (len(model_out) + 1e-8)
                reca = float(len(overlap)) / (len(std_out) + 1e-8)
                if len(std_out) == 0:
                    reca = 1.
                    if len(model_out) == 0:
                        prec = 1.
                f1 = 2 * prec * reca / (prec + reca + 1e-8)
                """

                nltk_tree = dump_tree(bp, 9, 0, length-1, self.test_txts[b_id*batch_size+i].strip().split(), threshold)
                f_t.write(str(nltk_tree).replace('\n','').replace(' ','') + '\n')