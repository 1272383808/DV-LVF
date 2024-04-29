from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import RobertaTokenizerFast, T5EncoderModel, get_linear_schedule_with_warmup
from tqdm import tqdm
from model import myModel
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pandas as pd


logger = logging.getLogger(__name__)

class InputFeatures(object):
    def __init__(self,
                 input_ids,
                 statement_mask,
                 stmt_labels,
                 labels,
                 num_statements):
        self.input_ids = input_ids
        self.statement_mask = statement_mask
        self.stmt_labels = stmt_labels
        self.labels = labels
        self.num_statements = num_statements


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_type="train"):
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "val":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        self.examples = []
        
        df_all = pd.read_csv(file_path)
        
        df_vul = df_all[df_all["function_label"]==1].reset_index(drop=True)
        
        # no balance
        df_non_vul = df_all[df_all["function_label"]==0].reset_index(drop=True)


        df = pd.concat((df_vul, df_non_vul))
        df = df.sample(frac=1).reset_index(drop=True)
        
        if file_type == "train":
            patterns = df["vul_patterns"].tolist()
            labels = df["statement_label"].tolist()
            source = df["func_before"].tolist()
        else:
            patterns = df["vul_patterns"].tolist()
            labels = df["statement_label"].tolist()
            source = df["func_before"].tolist()

        print("\n*******\n", f"total non-vul funcs in {file_type} data: {len(df_non_vul)}")
        print(f"total vul funcs in {file_type} data: {len(df_vul)}", "\n*******\n")

        for i in tqdm(range(len(source))):
            self.examples.append(convert_examples_to_features(source[i], patterns[i], labels[i], tokenizer, args, file_type))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].statement_mask), torch.tensor(self.examples[i].labels).float(), torch.tensor(self.examples[i].func_labels), torch.tensor(self.examples[i].num_statements)

def convert_examples_to_features(source, pattern, labels, tokenizer, args, data_split):
    labels = labels.strip("[").strip("]")
    labels = labels.split(",")
    labels = [int(l.strip()) for l in labels]
    assert len(labels) == args.max_num_statements

    source = source.split("\n")
    source = source[:args.max_num_statements]
    padding_statement = [tokenizer.pad_token_id for _ in range(25)]
    num_statements = len(source)
    input_ids = []
    for stat in source:
        ids_ = tokenizer.encode(str(stat),
                                truncation=True,
                                max_length=25,
                                padding='max_length',
                                add_special_tokens=False)
        input_ids.append(ids_)
    if len(input_ids) < args.max_num_statements:
        for _ in range(args.max_num_statements-len(input_ids)):
            input_ids.append(padding_statement)

    statement_mask = []   # [120]
    for statement in input_ids:
        if statement == padding_statement:
            statement_mask.append(0)
        else:
            statement_mask.append(1)

    if 1 in labels:
        func_labels = 1
    else:
        func_labels = 0

    statement_mask += [1]

    if data_split == "train":
        if 1 not in labels:
            ids_ = tokenizer.encode(str(pattern),
                                        truncation=True,
                                        max_length=25,
                                        padding='max_length',
                                        add_special_tokens=False)
            pattern_ids = [ids_]
            for _ in range(11):
                pattern_ids.append(padding_statement)
            input_ids = input_ids + pattern_ids
        else:
            pattern = pattern.split("<SPLIT>")
            pattern = [p for p in pattern if p != ""]
            pattern_ids = []
            for pat in pattern:
                ids_ = tokenizer.encode(str(pat),
                                        truncation=True,
                                        max_length=25,
                                        padding='max_length',
                                        add_special_tokens=False)
                pattern_ids.append(ids_)
            pattern_ids = pattern_ids[:10]
            # 10 patterns - 90% of Qt.
            if len(pattern_ids) < 10:
                for _ in range(10-len(pattern_ids)):
                    pattern_ids.append(padding_statement)
            input_ids = input_ids + pattern_ids
        assert len(input_ids) == args.max_num_statements + 10
    else:
        assert len(input_ids) == args.max_num_statements
    return InputFeatures(input_ids, statement_mask, labels, func_labels, num_statements)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer, eval_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)

    args.max_steps = args.epochs * len(train_dataloader)

    # evaluate model per ? epoch
    args.save_steps = len(train_dataloader) * 1
    eval_epo = [34]
    # eval_epo = [0, 20, 22, 23, 24]

    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #scheduler = get_constant_schedule(optimizer)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*args.epochs*0.1, num_training_steps=len(train_dataloader)*args.epochs)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0
    model.zero_grad()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (input_ids, statement_mask, stmt_labels, labels, num_statements) = [x.to(args.device) for x in batch]
            model.train()
            statement_loss, s_loss = model(input_ids_with_pattern=input_ids,
                                                        statement_mask=statement_mask,
                                                        stmt_labels=stmt_labels,
                                                        labels=labels)

            loss = 0.7 * statement_loss + 0.3 * s_loss     ###

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num,5)

            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                avg_loss = round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if global_step % args.save_steps == 0 and idx in eval_epo:   # [0, 17, 18, 19]
                    eval_f1 = evaluate(args, model, tokenizer, eval_dataset, eval_when_training=True)
                    print("enter the second if. best_f1:", best_f1, ",eval_f1:", eval_f1)
                    # Save model pth
                    if eval_f1 > best_f1:
                        best_f1 = eval_f1
                        logger.info("  "+"*"*20)
                        logger.info("  Best F1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)
                        checkpoint_prefix = 'best-model'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    #build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()

    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    y_preds = []
    y_trues = []
    for step, batch in enumerate(bar):
        with torch.no_grad():
            (input_ids, statement_mask, stmt_labels, labels, num_statements) = [x.to(args.device) for x in batch]
            probs, func_probs = model(input_ids_with_pattern=input_ids,
                                      statement_mask=statement_mask)
            
            preds = torch.where(probs>0.5, 1, 0).tolist()

            func_preds = torch.argmax(func_probs, dim=-1).tolist()

            for indx in range(len(preds)):
                sample = preds[indx]   # 120*1
                if func_preds[indx] == 1:
                    for s in range(num_statements[indx]):
                        p = sample[s]
                        y_preds.append(p)
                else:
                    for _ in range(num_statements[indx]):
                        y_preds.append(0)
            stmt_labels = stmt_labels.cpu().numpy().tolist()
            for indx in range(len(stmt_labels)):
                sample = stmt_labels[indx]
                for s in range(num_statements[indx]):
                    lab = sample[s]
                    y_trues.append(lab)

    model.train()
    f1 = f1_score(y_true=y_trues, y_pred=y_preds)
    logger.info("***** Eval results *****")
    logger.info(f"F1 Accuracy: {str(f1)}")
    return f1

def test(args, model, tokenizer, eval_dataset, eval_when_training=False):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()

    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    y_preds = []    # 针对所有函数所有代码行的标签
    y_trues = []
    top_10_acc = []   # 只记录存在漏洞并且正确预测的函数中--每个函数top10行 是否真正是存在漏洞的行
    top_3_acc = []
    top_2_acc = []
    top_1_acc = []
    func_level_preds = []
    func_level_trues = []
    for step, batch in enumerate(bar):
        with torch.no_grad():
            (input_ids, statement_mask, stmt_labels, labels, num_statements) = [x.to(args.device) for x in batch]
            probs, func_probs = model(input_ids_with_pattern=input_ids,
                                      statement_mask=statement_mask,
                                      phase_one_training=args.phase_one_training)

            preds = torch.where(probs>0.5, 1, 0).tolist()

            func_preds = torch.argmax(func_probs, dim=-1).tolist()

            for indx in range(len(preds)):
                sample = preds[indx]
                if func_preds[indx] == 1:
                    for s in range(num_statements[indx]):
                        p = sample[s]
                        y_preds.append(p)
                else:
                    for _ in range(num_statements[indx]):
                        y_preds.append(0)
            stmt_labels = stmt_labels.cpu().numpy().tolist()
            for indx in range(len(stmt_labels)):
                sample = stmt_labels[indx]
                for s in range(num_statements[indx]):
                    lab = sample[s]
                    y_trues.append(lab)

            ### function-level ###
            labels = labels.cpu().numpy().tolist()
            func_level_trues += labels
            func_level_preds += func_preds

            ### top-k acc ###
            for indx in range(len(preds)):
                sample = probs[indx]
                line_label = labels[indx]
                prediction = []
                if func_preds[indx] == 1 and labels[indx] == 1:
                    for s in range(num_statements[indx]):
                        p = sample[s]
                        prediction.append(p)
                    ranking3 = sorted(range(len(prediction)), key=lambda i: prediction[i], reverse=True)[:3]
                    ranking2 = sorted(range(len(prediction)), key=lambda i: prediction[i], reverse=True)[:2]
                    ranking1 = sorted(range(len(prediction)), key=lambda i: prediction[i], reverse=True)[:1]
                    top_3_pred = [0 for _ in range(120)]
                    for r in ranking3:
                        top_3_pred[r] = 1
                    correct3 = 0
                    for x in range(len(line_label)):
                        if line_label[x] == 1 and top_3_pred[x] == 1:
                            correct3 = 1
                    top_3_acc.append(correct3)

                    top_2_pred = [0 for _ in range(120)]
                    for r in ranking2:
                        top_2_pred[r] = 1
                    correct2 = 0
                    for x in range(len(line_label)):
                        if line_label[x] == 1 and top_2_pred[x] == 1:
                            correct2 = 1
                    top_2_acc.append(correct2)

                    top_1_pred = [0 for _ in range(120)]
                    for r in ranking1:
                        top_1_pred[r] = 1
                    correct = 0
                    for x in range(len(line_label)):
                        if line_label[x] == 1 and top_1_pred[x] == 1:
                            correct = 1
                    top_1_acc.append(correct)

    f1 = f1_score(y_true=func_level_trues, y_pred=func_level_preds)
    acc = accuracy_score(y_true=func_level_trues, y_pred=func_level_preds)
    recall = recall_score(y_true=func_level_trues, y_pred=func_level_preds)
    pre = precision_score(y_true=func_level_trues, y_pred=func_level_preds)

    logger.info("***** Function-level Test results *****")
    logger.info(f"F1 Score: {str(f1)}")
    logger.info(f"acc Score: {str(acc)}")
    logger.info(f"recall Score: {str(recall)}")
    logger.info(f"pre Score: {str(pre)}")

    f1 = f1_score(y_true=y_trues, y_pred=y_preds)
    acc = accuracy_score(y_true=y_trues, y_pred=y_preds)
    recall = recall_score(y_true=y_trues, y_pred=y_preds)
    pre = precision_score(y_true=y_trues, y_pred=y_preds)

    top_3_acc = round(sum(top_3_acc)/len(top_3_acc), 4)
    top_2_acc = round(sum(top_2_acc)/len(top_2_acc), 4)
    top_1_acc = round(sum(top_1_acc)/len(top_1_acc), 4)

    logger.info("***** Line-level Test results *****")
    logger.info(f"F1 Score: {str(f1)}")
    logger.info(f"acc Score: {str(acc)}")
    logger.info(f"recall Score: {str(recall)}")
    logger.info(f"pre Score: {str(pre)}")
    logger.info(f"Top-3 Accuracy: {str(top_3_acc)}")
    logger.info(f"Top-2 Accuracy: {str(top_2_acc)}")
    logger.info(f"Top-1 Accuracy: {str(top_1_acc)}")

    return f1

def main():
    ps = argparse.ArgumentParser()
    ps.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    ps.add_argument("--eval_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    ps.add_argument("--test_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    ps.add_argument("--pretrain_language", default="", type=str, required=False,
                        help="python, go, ruby, php, javascript, java, c_cpp")
    ps.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ps.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    ps.add_argument("--encoder_block_size", default=512, type=int,
                        help="")
    ps.add_argument("--max_line_length", default=64, type=int,
                        help="")
    ps.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    ps.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    ps.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    ps.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    ps.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    ps.add_argument("--do_test", action='store_true',
                        help="Whether to run training.")
    ps.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    ps.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    ps.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    ps.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    ps.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for AdamW.")
    ps.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    ps.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    ps.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    ps.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    ps.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    ps.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    ps.add_argument('--epochs', type=int, default=3,
                        help="training epochs")
    ps.add_argument('--max_num_statements', type=int, default=120,
                        help="max num of statements per function")
    ps.add_argument('--num_clusters', type=int, default=150,
                        help="")
    ps.add_argument("--no_warmup", action='store_true',
                    help="")
    args = ps.parse_args()
    # CUDA
    args.n_gpu = 1
    args.device = "cuda:0"
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", args.device, args.n_gpu)
    set_seed(args)

    tokenizer = RobertaTokenizerFast.from_pretrained("D:\LLMs\codet5-base")
    t5 = T5EncoderModel.from_pretrained("D:\LLMs\codet5-base")

    model = myModel(t5, tokenizer, args, hidden_dim=768, vul_hidden=192, num_clusters=args.num_clusters)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if not args.phase_one_training and not args.no_warmup:
            output_dir = "./saved_models/best_model.bin"
            model.load_state_dict(torch.load(output_dir, map_location=args.device), strict=False)
        train_dataset = TextDataset(tokenizer, args, file_type='train')
        eval_dataset = TextDataset(tokenizer, args, file_type='val')
        train(args, train_dataset, model, tokenizer, eval_dataset)
    if args.do_test:
        checkpoint_prefix = f'saved_models/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)
        test_dataset = TextDataset(tokenizer, args, file_type='test')
        test(args, model, tokenizer, test_dataset)


if __name__ == "__main__":
    main()
