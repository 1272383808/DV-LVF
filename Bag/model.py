import numpy as np
import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_dim, args):
        super().__init__()
        self.args = args
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.Dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_dim, 1)

        self.rnn_pool = nn.GRU(input_size=768,
                               hidden_size=768,
                               num_layers=1,
                               batch_first=True)
        self.func_dense = nn.Linear(hidden_dim, hidden_dim)
        self.func_out_proj = nn.Linear(hidden_dim, 2)

    def forward(self, hidden):
        # statement prediction
        x = self.Dropout(hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.Dropout(x)
        x = self.out_proj(x)
        # function prediction
        out, func_x = self.rnn_pool(hidden)
        func_x = func_x.squeeze(0)
        func_x = self.Dropout(func_x)
        func_x = self.func_dense(func_x)
        func_x = torch.tanh(func_x)
        func_x = self.Dropout(func_x)
        func_x = self.func_out_proj(func_x)
        return x.squeeze(-1), func_x


class myModel(nn.Module):
    def __init__(self, t5, tokenizer, args, hidden_dim=768, vul_hidden=192, num_clusters=150):
        super(myModel, self).__init__()
        self.vul_hidden = vul_hidden
        self.word_embedding = t5.shared  # (shared): Embedding(词表大小, 768)
        self.t5 = t5
        self.tokenizer = tokenizer
        self.args = args
        # CLS head
        self.classifier = ClassificationHead(hidden_dim=hidden_dim, args=args)
        self.rnn_statement_embedding = nn.GRU(input_size=hidden_dim,
                                              hidden_size=hidden_dim,
                                              num_layers=1,
                                              batch_first=True)
        self.rnn_vulI_pooling = nn.GRU(input_size=hidden_dim,
                                            hidden_size=hidden_dim,
                                            num_layers=1,
                                            batch_first=True)
        self.num_clusters = num_clusters

        self.low = nn.Linear(hidden_dim, vul_hidden)
        self.high = nn.Linear(vul_hidden, hidden_dim)
        self.layer_norm = nn.LayerNorm(vul_hidden)

    def forward(self, input_ids_with_vulI, statement_mask, stmt_labels=None, labels=None):
        cluster_centers_tensor = torch.load("vuldict.pth")
        if self.training:
            embed = self.word_embedding(input_ids_with_vulI)
            inputs_embeds = torch.randn(embed.shape[0], embed.shape[1], embed.shape[3]).to(self.args.device)
            for i in range(len(embed)):
                statement_of_tokens = embed[i]
                out, statement_embed = self.rnn_statement_embedding(statement_of_tokens)
                inputs_embeds[i, :, :] = statement_embed
            vulIs = inputs_embeds[:, self.args.max_num_statements:, :]
            out, vulIs = self.rnn_vulI_pooling(vulIs)
            vulIs = vulIs.permute(1, 0, 2)
            vulIs = self.low(vulIs)
            vulIs = self.layer_norm(vulIs)
            vulIs = vulIs.squeeze(1)
            closest_vulCenter = np.zeros((embed.shape[0], self.vul_hidden))
            for n in range(len(vulIs)):
                vulI = vulIs[n]
                distances = torch.norm(vulI - cluster_centers_tensor, dim=1)
                closest_index = torch.argmin(distances)  # 索引
                closest_vulCenter[n] = cluster_centers_tensor[closest_index]
            closest_vulCenter = self.high(closest_vulCenter).unsqueeze(1)
            vulIs = self.high(vulIs).unsqueeze(1)
            
            inputs_embeds = inputs_embeds[:, :self.args.max_num_statements, :]
            inputs_embeds = torch.cat((inputs_embeds, closest_vulCenter), dim=1)

            rep = self.t5(inputs_embeds=inputs_embeds, attention_mask=statement_mask).last_hidden_state

            h_function = rep[:, :self.args.max_num_statements, :]

            logits, func_logits = self.classifier(h_function)
            loss_fct = nn.BCEWithLogitsLoss()
            statement_loss = loss_fct(logits, stmt_labels)
            loss_fct_func = nn.CrossEntropyLoss()
            func_loss = loss_fct_func(func_logits, labels)
            return statement_loss, func_loss
        else:  # 推理
            embed = self.word_embedding(input_ids_with_vulI)
            inputs_embeds = torch.randn(embed.shape[0], embed.shape[1], embed.shape[3]).to(self.args.device)
            for i in range(len(embed)):
                statement_of_tokens = embed[i]
                out, statement_embed = self.rnn_statement_embedding(statement_of_tokens)
                inputs_embeds[i, :, :] = statement_embed
            all_prob = None
            all_func_prob = None

            for z in range(len(cluster_centers_tensor)):
                vulCenter = self.vp_codebook.weight[z]
                vulCenter = vulCenter.unsqueeze(0).unsqueeze(0).expand(inputs_embeds.shape[0], 1,
                                                                           vulCenter.shape[0]).to(
                    self.args.device)
                vulCenter = self.high(vulCenter)

                input_embeddings = torch.cat((inputs_embeds, vulCenter), dim=1)
                rep = self.t5(inputs_embeds=input_embeddings, attention_mask=statement_mask).last_hidden_state

                h_function = rep[:, :self.args.max_num_statements, :]

                logits, func_logits = self.classifier(h_function)
                prob = torch.sigmoid(logits).detach().cpu()
                prob = prob.unsqueeze(0)
                func_prob = torch.softmax(func_logits, dim=-1)
                func_prob = func_prob.unsqueeze(0)
                if all_prob is None:
                    all_prob = prob
                    all_func_prob = func_prob
                else:
                    all_prob = torch.cat((all_prob, prob), dim=0)
                    all_func_prob = torch.cat((all_func_prob, func_prob), dim=0)
            all_prob = torch.mean(all_prob, dim=0)
            all_func_prob = torch.amax(all_func_prob, dim=0)
            return all_prob, all_func_prob
