import os
import time
import torch
import argparse
import numpy as np

from model import STRec   
from data_preprocess import *
from utils_geo import data_partition, WarpSampler, evaluate, evaluate_valid

from tqdm import tqdm
os.environ['CUDA_MPS_PIPE_DIRECTORY'] = '/nonexistent'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="")
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=20, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda:3', type=str)
parser.add_argument('--inference_only', default=False, action='store_true')
parser.add_argument('--state_dict_path', default=None, type=str)

args = parser.parse_args()

if __name__ == '__main__':

    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, poinum, geonum] = dataset

    print('user num:', usernum, 'poi num:', poinum, 'geo num:', geonum)

    num_batch = len(user_train) // args.batch_size

    cc = 0.0
    for u in user_train:
        cc += len(user_train[u][0])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    sampler = WarpSampler(
        user_train,
        usernum,
        poinum,   
        geonum,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        n_workers=3
    )

    model = SASRec(usernum, poinum, geonum, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    model.train()

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            kwargs, checkpoint = torch.load(args.state_dict_path, map_location=torch.device(args.device))
            kwargs['args'].device = args.device
            model = STRec(**kwargs).to(args.device)
            model.load_state_dict(checkpoint)
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts:', args.state_dict_path)
            import pdb; pdb.set_trace()

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@1: %.4f, HR@1: %.4f)' % (t_test[0], t_test[1]))

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()

    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):

        if args.inference_only:
            break

        for step in range(num_batch):

            u, seq, geo_seq, time_seq, pos, neg = sampler.next_batch()

            u = np.array(u)
            seq = np.array(seq)
            geo_seq = np.array(geo_seq)
            time_seq = np.array(time_seq)
            pos = np.array(pos)
            neg = np.array(neg)

            pos_logits, neg_logits = model(u, seq, geo_seq, time_seq, pos, neg)

            pos_labels = torch.ones(pos_logits.shape, device=args.device)
            neg_labels = torch.zeros(neg_logits.shape, device=args.device)

            adam_optimizer.zero_grad()

            indices = np.where(pos != 0)

            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            if hasattr(model, "poi_emb"):
                for param in model.poi_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)
            else:
                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)

            loss.backward()
            adam_optimizer.step()

            if step % 100 == 0:
                print(f"[POI TRAIN] epoch {epoch} step {step} loss: {loss.item()}")

        if epoch % 20 == 0 or epoch == 1:
            model.eval()

            t1 = time.time() - t0
            T += t1

            print('Evaluating...')

            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)

            print(
                'epoch:%d, time: %f(s), valid (NDCG@1: %.4f, HR@1: %.4f), test (NDCG@1: %.4f, HR@1: %.4f)'
                % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1])
            )

            print(str(t_valid) + ' ' + str(t_test) + '\n')

            t0 = time.time()
            model.train()

        if epoch == args.num_epochs:
            folder = ''
            fname = f''

            if not os.path.exists(folder):
                os.makedirs(folder)

            torch.save([model.kwargs, model.state_dict()], os.path.join(folder, fname))

    sampler.close()
    print("Done")
