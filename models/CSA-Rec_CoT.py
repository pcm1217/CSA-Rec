import random
import pickle
import gzip

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.recsys_model import *
from models.llm4rec import *
from sentence_transformers import SentenceTransformer
from collections import defaultdict


class two_layer_mlp(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.fc1 = nn.Linear(dims, 128)
        self.fc2 = nn.Linear(128, dims)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x1 = self.fc2(x)
        return x, x1


class CSA-Rec_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        rec_pre_trained_data = args.rec_pre_trained_data
        self.args = args
        self.device = args.device
        
        with gzip.open(f'','rb') as ft:
            self.text_name_dict = pickle.load(ft)

        with open(f'', 'rb') as f:
            self.image_feat_dict = pickle.load(f)

        self.recsys = RecSys(args.recsys, rec_pre_trained_data, self.device)

        self.poi_num = self.recsys.item_num
        self.rec_sys_dim = self.recsys.hidden_units
        self.sbert_dim = 768
        self.image_feat_dim = 768
        
        self.mlp = two_layer_mlp(self.rec_sys_dim)

        if args.pretrain_stage1:
            self.sbert = SentenceTransformer("")
            self.mlp2 = two_layer_mlp(self.sbert_dim)
            self.mlp3 = two_layer_mlp(self.image_feat_dim)

        self.bce_criterion = torch.nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

        self.maxlen = args.maxlen


    def contrastive_loss(self, z1, z2, temperature=0.2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        logits = torch.matmul(z1, z2.T) / temperature
        labels = torch.arange(z1.size(0)).long().to(z1.device)

        loss1 = self.ce_loss(logits, labels)
        loss2 = self.ce_loss(logits.T, labels)

        return (loss1 + loss2) / 2


    def find_poi_text(self, poi_ids):
        t = 'name'
        d = 'category'
        return [
            f'"{self.text_name_dict[t].get(i,"No name")}, {self.text_name_dict[d].get(i,"No category")}"'
            for i in poi_ids
        ]

    def get_image_feat(self, batch_ids):
        feats = []
        for pid in batch_ids:
            pid = pid.item() if torch.is_tensor(pid) else pid
            feats.append(torch.tensor(self.image_feat_dict.get(pid, np.zeros(self.image_feat_dim))))
        return torch.stack(feats).to(self.device)


    def pre_train_phase1(self, data, optimizer, batch_iter):

        epoch, total_epoch, step, total_step = batch_iter
        
        self.sbert.train()
        optimizer.zero_grad()

        u, seq, geo_seq, time_seq, pos, neg = data

        indices = [self.maxlen*(i+1)-1 for i in range(u.shape[0])]
        
        with torch.no_grad():
            log_emb, pos_emb, neg_emb = self.recsys.model(
                u, seq, geo_seq, time_seq, pos, neg, mode='item'
            )
            
        log_emb = log_emb[indices]
        pos_emb = pos_emb[indices]

        pos_text = self.find_poi_text(pos.reshape(pos.size)[indices])

        pos_token = self.sbert.tokenize(pos_text)
        pos_text_emb = self.sbert({
            'input_ids': pos_token['input_ids'].to(self.device),
            'attention_mask': pos_token['attention_mask'].to(self.device)
        })['sentence_embedding']

        pos_img_emb = self.get_image_feat(pos.reshape(pos.size)[indices])

        _, poi_proj = self.mlp(pos_emb)
        _, text_proj = self.mlp2(pos_text_emb)
        _, img_proj = self.mlp3(pos_img_emb)

        pos_logits = (log_emb * poi_proj).mean(axis=1)
        pos_labels = torch.ones_like(pos_logits)

        rec_loss = self.bce_criterion(pos_logits, pos_labels)

        cl_poi_text = self.contrastive_loss(poi_proj, text_proj)
        cl_poi_img = self.contrastive_loss(poi_proj, img_proj)

        cl_loss = cl_poi_text + cl_poi_img

        total_loss = rec_loss + 0.2 * cl_loss

        total_loss.backward()
        optimizer.step()

        print(f"[Phase1] Epoch {epoch} Step {step} Loss: {total_loss.item():.4f}")
        
    def make_interact_text(self, interact_ids, interact_max_num):
        interact_item_titles_ = self.find_poi_text(interact_ids, title_flag=True, description_flag=False)
        interact_text = []
    
        if interact_max_num == 'all':
            for title in interact_item_titles_:
                interact_text.append(title + '[History]')
        else:
            for title in interact_item_titles_[-interact_max_num:]:
                interact_text.append(title + '[History]')
            interact_ids = interact_ids[-interact_max_num:]
    
        interact_text = ','.join(interact_text)
        return interact_text, interact_ids


    def make_candidate_text(self, interact_ids, candidate_num, target_poi_id, target_poi_title):
        neg_poi_id = set()
    
        last_id = interact_ids[-1]
        g4_prefix = self.poi_geohash4_dict.get(last_id, None)
    
        if g4_prefix and g4_prefix in self.prefix_to_poi:
            candidate_pool = list(self.prefix_to_poi[g4_prefix])
            random.shuffle(candidate_pool)
    
            for pid in candidate_pool:
                if pid not in interact_ids and pid != target_poi_id:
                    neg_poi_id.add(pid)
                    if len(neg_poi_id) >= 20:
                        break
    
        while len(neg_poi_id) < 20:
            t = np.random.randint(1, self.poi_num + 1)
            if t not in interact_ids and t != target_poi_id and t not in neg_poi_id:
                neg_poi_id.add(t)
    
        neg_poi_id = list(neg_poi_id)
        random.shuffle(neg_poi_id)
    
        candidate_ids = [target_poi_id]
        candidate_text = [target_poi_title + '[Candidate]']
    
        for neg_candidate in neg_poi_id[:candidate_num - 1]:
            candidate_text.append(
                self.find_poi_text_single(neg_candidate, title_flag=True, description_flag=False) + '[Candidate]'
            )
            candidate_ids.append(neg_candidate)
    
        random_ = np.random.permutation(len(candidate_text))
        candidate_text = np.array(candidate_text)[random_]
        candidate_ids = np.array(candidate_ids)[random_]
    
        return ','.join(candidate_text), candidate_ids
    def pre_train_phase2(self, data, optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter
    
        optimizer.zero_grad()
        u, seq, geo_seq, time_seq, pos, neg = data
        mean_loss = 0
    
        text_input = []
        text_output = []
        interact_embs = []
        candidate_embs = []
    
        self.llm.eval()
    
        with torch.no_grad():
            log_emb = self.recsys.model(u, seq, geo_seq, time_seq, pos, neg, mode='log_only')
    
        for i in range(len(u)):
    
            target_poi_id = pos[i][-1]
            target_poi_title = self.find_poi_text_single(
                target_poi_id,
                title_flag=True,
                description_flag=False
            )
    
            interact_text, interact_ids = self.make_interact_text(
                seq[i][seq[i] > 0], 20
            )
    
            candidate_num = 20
            candidate_text, candidate_ids = self.make_candidate_text(
                seq[i][seq[i] > 0],
                candidate_num,
                target_poi_id,
                target_poi_title
            )
    
            input_text = f"""
            [User Representation] is a user representation.
            The following is the historical interaction of this user:
            {interact_text}
    
            You are asked to recommend the next POI for this user.
            Consider the following candidate POIs:
            {candidate_text}
    
            Step-by-step reasoning:
            1. Analyze user's preferences.
            2. Analyze candidate POIs.
            3. Match compatibility.
            4. Rank candidates.
    
            Recommend one next POI:
            """
    
            text_input.append(input_text)
            text_output.append(target_poi_title)
    
            interact_embs.append(
                self.item_emb_proj(self.get_item_emb(interact_ids))
            )
            candidate_embs.append(
                self.item_emb_proj(self.get_item_emb(candidate_ids))
            )
    
        samples = {
            'text_input': text_input,
            'text_output': text_output,
            'interact': interact_embs,
            'candidate': candidate_embs
        }
    
        log_emb = self.log_emb_proj(log_emb)
    
        loss_rm = self.llm(log_emb, samples)
        loss_rm.backward()
        optimizer.step()
    
        mean_loss += loss_rm.item()
    
        print(
            f"CSA-Rec model loss in epoch {epoch}/{total_epoch} "
            f"iteration {step}/{total_step}: {mean_loss}"
        )
    def generate(self, data):
        u, seq, geo_seq, time_seq, pos, neg = data
    
        answer = []
        text_input = []
        interact_embs = []
        candidate_embs = []
    
        with torch.no_grad():
    
            log_emb = self.recsys.model(
                u, seq, geo_seq, time_seq, pos, neg,
                mode='log_only'
            )
    
            for i in range(len(u)):
    
                target_poi_id = pos[i]
                target_poi_title = self.find_poi_text_single(
                    target_poi_id,
                    title_flag=True,
                    description_flag=False
                )
    
                interact_text, interact_ids = self.make_interact_text(
                    seq[i][seq[i] > 0], 20
                )
    
                candidate_num = 20
                candidate_text, candidate_ids = self.make_candidate_text(
                    seq[i][seq[i] > 0],
                    candidate_num,
                    target_poi_id,
                    target_poi_title
                )
    
                if len(interact_ids) == 0 or len(candidate_ids) == 0:
                    continue
    
                input_text = f"""
                [User Representation] is a user representation.
                The following is the historical interaction of this user:
                {interact_text}
    
                You are asked to recommend the next POI for this user.
                Consider the following candidate POIs:
                {candidate_text}
    
                Step-by-step reasoning:
                1. Analyze preferences.
                2. Analyze POIs.
                3. Match compatibility.
                4. Rank candidates.
    
                Recommend one next POI:
                """
    
                text_input.append(input_text)
                answer.append(target_poi_title)
    
                interact_embs.append(
                    self.item_emb_proj(self.get_item_emb(interact_ids))
                )
                candidate_embs.append(
                    self.item_emb_proj(self.get_item_emb(candidate_ids))
                )
    
            if len(text_input) == 0:
                return []
    
            log_emb = self.log_emb_proj(log_emb)
    
            atts_llm = torch.ones(
                log_emb.size()[:-1],
                dtype=torch.long
            ).to(self.device).unsqueeze(1)
    
            log_emb = log_emb.unsqueeze(1)
    
            self.llm.llm_tokenizer.padding_side = "left"
    
            llm_tokens = self.llm.llm_tokenizer(
                text_input,
                padding="longest",
                return_tensors="pt"
            ).to(self.device)
    
            inputs_embeds = self.llm.llm_model.get_input_embeddings()(
                llm_tokens.input_ids
            )
    
            llm_tokens, inputs_embeds = self.llm.replace_hist_candi_token(
                llm_tokens,
                inputs_embeds,
                interact_embs,
                candidate_embs
            )
    
            attention_mask = llm_tokens.attention_mask
    
            inputs_embeds = torch.cat([log_emb, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, attention_mask], dim=1)
    
            outputs = self.llm.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                top_p=0.9,
                temperature=1,
                num_beams=1,
                max_new_tokens=512,
                pad_token_id=self.llm.llm_tokenizer.eos_token_id,
                repetition_penalty=1.5,
                length_penalty=1
            )
    
            output_text = self.llm.llm_tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )
    
            return [t.strip() for t in output_text]