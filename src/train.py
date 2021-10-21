import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup, AdamW
import logging
from utils import utils
import metrics
import math


def save(model, path, step):
    path += "_epoch{}.pkl".format("{}".format(step))
    # if os.path.exists(path):
    #     os.remove(path)
    #     logging.info("model remove success!!!")
    logging.info("Save model")
    torch.save(model, path)


def ComGen_valid(valid_iter, model, tokenizer, args):
    logging.info("Start valid")
    model.eval()
    predicts = []
    for item in valid_iter:
        logit = model.generate(item["input_ids"].cuda(), num_beams=16, max_length=20, early_stopping=True)
        # logit = model(input_ids=item["input_ids"].cuda(), attention_mask=item["attention_mask"].cuda()).logits
        # logit = torch.max(F.softmax(logit, dim=-1), dim=-1)[1].cpu()
        predict = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in logit]
        predicts.extend(predict)
    save(model, args.model_save, args.step)
    with open(args.model_save + "_epoch{}.txt".format(args.step), "w", encoding="utf-8") as f:
        for real, predict in zip(args.reals, predicts):
            f.write("real: " + real + "\n")
            f.write("predict: " + predict + "\n")
            f.write("----------------------------------------------\n")
            f.write("\n")


def ComGen_train(train_iter, valid_iter, model, tokenizer, args):
    optimizer = AdamW(model.parameters(), args.learning_rate, weight_decay=0.0001, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_iter), num_training_steps=len(train_iter) * args.epoch)
    mean_loss = 0
    for step in range(args.epoch):
        model.train()
        logging.info("Starting Training epoch:{}".format(step+1))
        for item in train_iter:
            loss = model(input_ids=item["input_ids"].cuda(), attention_mask=item["attention_mask"].cuda(), labels=item["label"].cuda()).loss
            mean_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        args.step = step + 1
        mean_loss /= len(train_iter)
        logging.info("Train loss:{:.4f}".format(mean_loss))
        mean_loss = 0
        ComGen_valid(valid_iter, model, tokenizer, args)


def Order_valid(valid_iter, model, tokenizer, args):
    model.eval()
    predicts = []
    for item in valid_iter:
        input_ids = item["input_ids"]
        # attention_mask = item["input_mask"]
        # output_ids = item["output_ids"]
        # output_mask = item["output_mask"]
        # decoder_input_ids = output_ids[:, 0:1].contiguous()
        logits = model.generate(input_ids=input_ids.cuda(), no_repeat_ngram_size=1, num_beams=10, max_length=args.max_length + 2)
        predict = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in logits]
        predicts.extend(predict)
    acc, PMR, kendall_score, LCS, rouge = 0, 0, 0, 0, 0
    total, total_sents = 0, 0
    for gold, predict in zip(args.gold, predicts):
        predict = predict.replace("<S", "").replace(">", "").split(" ")
        predict = [int(item) for item in predict]
        total += 1
        total_sents += len(gold)
        acc += metrics.acc_compare(gold, predict)
        ktua = metrics.kendall_tau(predict, gold)
        LCS += metrics.lcs(gold, predict)
        rouge += metrics.rouge_s(gold, predict)
        if gold == predict:
            PMR += 1
        if math.isnan(ktua):
            ktua = 0
        kendall_score += ktua
    logging.info(" Accuracy: {:.6f}".format(acc / total_sents))
    logging.info(" PMR: {:.6f}".format(PMR / total))
    logging.info(" Kendall's Tau: {:.6f}".format(kendall_score / total))
    logging.info(" LCS: {:.6f}".format(LCS / total_sents))
    logging.info(" Rouge-S: {:.6f}".format(rouge / total))
    save(model, args.model_save, args.step)
        # utils.debug("predict", predict)
        # utils.debug("gold", gold)
    # utils.debug("predict:", predict)
        # optimizer.zero_grad()
        # lm_logits = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, use_cache=False).logits
        # batch_size, length, vocab_size = lm_logits.shape
        # lm_token_ids = torch.max(F.softmax(lm_logits, dim=-1), dim=-1)[1]


def Order_train(train_iter, valid_iter, model, tokenizer, args):
    utils.debug("model", model)
    no_decay = ["bias", "LayerNorm.weight"]
    high_lr = ["lm_head"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and not any(nd in n for nd in high_lr)
            ],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate * 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": args.learning_rate * 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in high_lr)
            ],
            "weight_decay": args.weight_decay,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_iter), num_training_steps=len(train_iter) * args.epoch)
    for step in range(args.epoch):
        args.step = step
        model.train()
        loss_mean = 0
        for item in train_iter:
            input_ids = item["input_ids"]
            attention_mask = item["input_mask"]
            output_ids = item["output_ids"]
            output_mask = item["output_mask"]
            decoder_input_ids = output_ids[:, :-1].contiguous()
            lm_logits = model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(), decoder_input_ids=decoder_input_ids.cuda(), use_cache=False).logits.cpu()
            batch_size, length, vocab_size = lm_logits.shape
            Loss_fn = nn.CrossEntropyLoss(reduction="none")
            lm_labels = output_ids[:, 1:].clone().contiguous()
            # utils.debug("lm_label", lm_labels.shape)
            # utils.debug("lm_logit", lm_logits.shape)
            loss = Loss_fn(lm_logits.view(-1, vocab_size), lm_labels.view(-1)).view(batch_size, length)
            loss_mask = output_mask[:, :-1].contiguous()
            loss = torch.mul(loss_mask, loss)
            # utils.debug("loss", loss.shape)
            loss = torch.mean(loss)
            loss_mean += loss.item()
            # utils.debug("loss", loss.shape)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        logging.info(f"epoch:{step+1} loss:{loss_mean / len(train_iter)}")
        loss_meam = 0
        with torch.no_grad():
            Order_valid(valid_iter, model, tokenizer, args)


def Base