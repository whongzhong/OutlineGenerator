import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup, AdamW
import logging


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
        logit = model(input_ids=item["input_ids"].cuda(), attention_mask=item["attention_mask"].cuda()).logits
        logit = torch.max(F.softmax(logit, dim=-1), dim=-1)[1].cpu()
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