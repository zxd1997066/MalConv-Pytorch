# coding: utf-8
from codecs import strict_errors
import os
import time
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
from src.util import ExeDataset,write_pred
from src.model import MalConv
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch Inference.')
parser.add_argument('--config_path', default='config/example.yaml', type=str, help='config file')
parser.add_argument('--seed', default=123, type=int, help='random seed')
parser.add_argument('--device', default="cpu", type=str, help='cpu, cuda or xpu')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--precision', default="float32", type=str, help='precision')
parser.add_argument('--channels_last', default=1, type=int, help='Use NHWC or not')
parser.add_argument('--profile', action='store_true', default=False, help='collect timeline')
parser.add_argument('--num_iter', default=1, type=int, help='test iterations')
parser.add_argument('--num_warmup', default=0, type=int, help='test warmup')
args = parser.parse_args()
print(args)


# Load config file for experiment
config_path = args.config_path
seed = int(args.seed)
conf = yaml.safe_load(open(config_path,'r'))

exp_name = conf['exp_name']+'_sd_'+str(seed)
print('Experiment:')
print('\t',exp_name)

np.random.seed(seed)
torch.manual_seed(seed)

train_data_path = conf['train_data_path']
train_label_path = conf['train_label_path']

valid_data_path = conf['valid_data_path']
valid_label_path = conf['valid_label_path']

log_dir = conf['log_dir']
pred_dir = conf['pred_dir']
checkpoint_dir = conf['checkpoint_dir']


log_file_path = log_dir+exp_name+'.log'
chkpt_acc_path = checkpoint_dir+exp_name+'.model'
pred_path = pred_dir+exp_name+'.pred'

# Parameters
# use_gpu = conf['use_gpu']
use_gpu = True if args.device == "cuda" and torch.cuda.is_available() else False
use_cpu = conf['use_cpu']
learning_rate = conf['learning_rate']
max_step = conf['max_step']
test_step = conf['test_step']
# batch_size = conf['batch_size']
batch_size = args.batch_size
first_n_byte = conf['first_n_byte']
window_size = conf['window_size']
display_step = conf['display_step']

sample_cnt = conf['sample_cnt']


# Load Ground Truth.
tr_label_table = pd.read_csv(train_label_path,header=None,index_col=0)
tr_label_table.index=tr_label_table.index.str.upper()
tr_label_table = tr_label_table.rename(columns={1:'ground_truth'})
val_label_table = pd.read_csv(valid_label_path,header=None,index_col=0)
val_label_table.index=val_label_table.index.str.upper()
val_label_table = val_label_table.rename(columns={1:'ground_truth'})


# Merge Tables and remove duplicate
tr_table = tr_label_table.groupby(level=0).last()
del tr_label_table
val_table = val_label_table.groupby(level=0).last()
del val_label_table
tr_table = tr_table.drop(val_table.index.join(tr_table.index, how='inner'))

print('Training Set:')
print('\tTotal',len(tr_table),'files')
print('\tMalware Count :',tr_table['ground_truth'].value_counts()[1])
print('\tGoodware Count:',tr_table['ground_truth'].value_counts()[0])


print('Validation Set:')
print('\tTotal',len(val_table),'files')
print('\tMalware Count :',val_table['ground_truth'].value_counts()[1])
print('\tGoodware Count:',val_table['ground_truth'].value_counts()[0])

if sample_cnt != 1:
    tr_table = tr_table.sample(n=sample_cnt,random_state=seed)


dataloader = DataLoader(ExeDataset(list(tr_table.index), train_data_path, list(tr_table.ground_truth),first_n_byte),
                            batch_size=batch_size, shuffle=True, num_workers=use_cpu)
validloader = DataLoader(ExeDataset(list(val_table.index), valid_data_path, list(val_table.ground_truth),first_n_byte),
                        batch_size=batch_size, shuffle=False, num_workers=use_cpu)

valid_idx = list(val_table.index)
del tr_table
del val_table


malconv = MalConv(input_length=first_n_byte,window_size=window_size)
bce_loss = nn.BCEWithLogitsLoss()
adam_optim = optim.Adam([{'params':malconv.parameters()}],lr=learning_rate)
sigmoid = nn.Sigmoid()

if args.channels_last:
    malconv = malconv.to(memory_format=torch.channels_last)
    print("---- Use NHWC model ")

if use_gpu:
    malconv = malconv.cuda()
    bce_loss = bce_loss.cuda()
    sigmoid = sigmoid.cuda()


step_msg = 'step-{}-loss-{:.6f}-acc-{:.4f}-time-{:.2f}'
valid_msg = 'step-{}-tr_loss-{:.6f}-tr_acc-{:.4f}-val_loss-{:.6f}-val_acc-{:.4f}'
log_msg = '{}, {:.6f}, {:.4f}, {:.6f}, {:.4f}, {:.2f}'
history = {}
history['tr_loss'] = []
history['tr_acc'] = []

# log = open(log_file_path,'w')
# log.write('step,tr_loss, tr_acc, val_loss, val_acc, time\n')

valid_best_acc = 0.0
total_step = 0
step_cost_time = 0


def train(dataloader, validloader, valid_best_acc, args):
    while total_step < max_step:
        
        # Training 
        for step,batch_data in enumerate(dataloader):
            start = time.time()
            
            adam_optim.zero_grad()
            
            cur_batch_size = batch_data[0].size(0)

            exe_input = batch_data[0].cuda() if use_gpu else batch_data[0]
            exe_input = Variable(exe_input.long(),requires_grad=False)
            
            label = batch_data[1].cuda() if use_gpu else batch_data[1]
            label = Variable(label.float(),requires_grad=False)
            
            pred = malconv(exe_input)
            loss = bce_loss(pred,label)
            loss.backward()
            adam_optim.step()
            
            history['tr_loss'].append(loss.cpu().data.numpy())
            history['tr_acc'].extend(list(label.cpu().data.numpy().astype(int)==(sigmoid(pred).cpu().data.numpy()+0.5).astype(int)))
            
            step_cost_time = time.time()-start
            
            if (step+1)%display_step == 0:
                print(step_msg.format(total_step,np.mean(history['tr_loss']),
                                    np.mean(history['tr_acc']),step_cost_time),end='\r',flush=True)
            total_step += 1

            # Interupt for validation
            if total_step%test_step ==0:
                break
            
            evaluate(validloader, valid_best_acc, args)


def evaluate(validloader, valid_best_acc, args):
    # Testing
    history['val_loss'] = []
    history['val_acc'] = []
    history['val_pred'] = []

    total_time = 0.0
    total_sample = 0
    for _,val_batch_data in enumerate(validloader):
        cur_batch_size = val_batch_data[0].size(0)

        exe_input = val_batch_data[0].cuda() if use_gpu else val_batch_data[0]
        if args.channels_last:
            try:
                exe_input = exe_input.contiguous(memory_format=torch.channels_last)
            except Exception as e:
                pass
        exe_input = Variable(exe_input.long(),requires_grad=False)

        label = val_batch_data[1].cuda() if use_gpu else val_batch_data[1]
        label = Variable(label.float(),requires_grad=True)

        if args.profile:
            prof_act = [torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]
            with torch.profiler.profile(
                activities=prof_act,
                record_shapes=True,
                schedule=torch.profiler.schedule(
                    wait=int(args.num_iter/2),
                    warmup=2,
                    active=1,
                ),
                on_trace_ready=trace_handler,
            ) as p:
                for i in range(args.num_iter):
                    elapsed = time.time()
                    pred = malconv(exe_input)
                    loss = bce_loss(pred,label)
                    elapsed = time.time() - elapsed
                    p.step()
                    print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                    if i >= args.num_warmup:
                        total_time += elapsed
                        total_sample += args.batch_size
        else:
            for i in range(args.num_iter):
                elapsed = time.time()
                pred = malconv(exe_input)
                loss = bce_loss(pred,label)
                elapsed = time.time() - elapsed
                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_time += elapsed
                    total_sample += args.batch_size

        history['val_loss'].append(loss.cpu().float().data.numpy())
        history['val_acc'].extend(list(label.cpu().float().data.numpy().astype(int)==(sigmoid(pred).cpu().float().data.numpy()+0.5).astype(int)))
        history['val_pred'].append(list(sigmoid(pred).cpu().float().data.numpy()))
        break

    throughput = total_sample / total_time
    print("inference Throughput:\t {:.2f} samples/s".format(throughput))
    return

    print(log_msg.format(total_step, np.mean(history['tr_loss']), np.mean(history['tr_acc']),
                    np.mean(history['val_loss']), np.mean(history['val_acc']),step_cost_time),
          file=log,flush=True)
    
    print(valid_msg.format(total_step,np.mean(history['tr_loss']),np.mean(history['tr_acc']),
                           np.mean(history['val_loss']),np.mean(history['val_acc'])))
    if valid_best_acc < np.mean(history['val_acc']):
        valid_best_acc = np.mean(history['val_acc'])
        torch.save(malconv,chkpt_acc_path)
        print('Checkpoint saved at',chkpt_acc_path)
        write_pred(history['val_pred'],valid_idx,pred_path)
        print('Prediction saved at', pred_path)

    history['tr_loss'] = []
    history['tr_acc'] = []

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                'MalConv-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)


if __name__ == "__main__":
    if args.precision == "bfloat16":
        print("---- Use cpu AMP bfloat16")
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            evaluate(validloader, valid_best_acc, args)
    elif args.precision == "float16":
        print("---- Use cuda AMP float16")
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            evaluate(validloader, valid_best_acc, args)
    else:
        evaluate(validloader, valid_best_acc, args)
