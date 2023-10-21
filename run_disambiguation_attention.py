
import argparse
import numpy as np
from transformers import BertTokenizer, \
    get_linear_schedule_with_warmup, get_constant_schedule, RobertaTokenizer
from disambiguation import *
from data_disambiguation import *
from utils import *
from datetime import datetime
from torch.optim import AdamW
from tqdm import tqdm




def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def strtime(datetime_checkpoint):
    diff = datetime.now() - datetime_checkpoint
    return str(diff).rsplit('.')[0]  # Ignore below seconds



def load_model(is_init, device, type_loss, args):
    model = ExtractInfoEncoder(args.transformer_model, device, args)

    if not is_init:
        state_dict = torch.load(args.model) if device.type == 'cuda' else \
            torch.load(args.model, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['sd'])

    return model


def configure_optimizer(args, model, num_train_examples):
    # https://github.com/google-research/bert/blob/master/optimization.py#L25
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr,
                      eps=args.adam_epsilon)

    num_train_steps = int(num_train_examples / args.batch /
                          args.gradient_accumulation_steps * args.epochs)
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps)

    return optimizer, scheduler, num_train_steps, num_warmup_steps


def configure_optimizer_simple(args, model, num_train_examples):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_train_steps = int(num_train_examples / args.B /
                          args.gradient_accumulation_steps * args.epochs)
    num_warmup_steps = 0

    scheduler = get_constant_schedule(optimizer)

    return optimizer, scheduler, num_train_steps, num_warmup_steps


def get_hit_scores(indices, labels):
    hit = 0
    nums = len(labels)
    for i in range(nums):
        indice = indices[i]
        label = labels[i]
        hit += any([label[index] for index in indice])
    return hit / nums


def evaluate(model, data_loader, device):
    data_loader = tqdm(data_loader, ncols=80)
    labels = []
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            text_input_ids, text_attention_mask, can_input_ids, can_attention_mask, mention_pos, label = batch
            score = model(text_input_ids, text_attention_mask, can_input_ids, can_attention_mask, mention_pos, label,
                           "val")
            label = label.view(-1)
            indice = score.argmax()
            labels.append(label[indice].item())

    return sum(labels) / len(labels), 0

def evaluate_test(model, tokenizer, device, args, logger):

    forgot_test_data = load_data(args.forgot_test_data)
    forgot_entities = load_entities(args.forgot_kb)
    forgot_data_loader = get_attention_mention_loader(forgot_test_data, forgot_entities, tokenizer, False, True, args)
    forgot_acc, _ = evaluate(model, forgot_data_loader, device)
    logger.log(f"forgot test acc: {forgot_acc}")

    lego_test_data = load_data(args.lego_test_data)
    lego_entities = load_entities(args.lego_kb)
    lego_data_loader = get_attention_mention_loader(lego_test_data, lego_entities, tokenizer, False, True, args)
    lego_acc, _ = evaluate(model, lego_data_loader, device)
    logger.log(f"lego test acc: {lego_acc}")

    star_test_data = load_data(args.star_test_data)
    star_entities = load_entities(args.star_kb)
    star_data_loader = get_attention_mention_loader(star_test_data, star_entities, tokenizer, False, True, args)
    star_acc, _ = evaluate(model, star_data_loader, device)
    logger.log(f"star test acc: {star_acc}")

    yugioh_test_data = load_data(args.yugioh_test_data)
    yugioh_entities = load_entities(args.yugioh_kb)
    yugioh_data_loader = get_attention_mention_loader(yugioh_test_data, yugioh_entities, tokenizer, False, True, args)
    yugioh_acc, _ = evaluate(model, yugioh_data_loader, device)
    logger.log(f"yugioh test acc: {yugioh_acc}")

    logger.log(f"macro:{(forgot_acc + lego_acc + star_acc + yugioh_acc) / 4}")
    all_correct = forgot_acc * len(forgot_test_data) + lego_acc * len(lego_test_data) + star_acc * len(
        star_test_data) + yugioh_acc * len(yugioh_test_data)
    all_len = len(forgot_test_data) + len(lego_test_data) + len(star_test_data) + len(yugioh_test_data)
    logger.log(f"micro:{all_correct / all_len}")


def train(samples_train, samples_dev, args):
    set_seeds(args)
    best_val_perf = float('-inf')
    logger = Logger(args.model + '.log', on=True)
    logger.log(str(args))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    logger.log(f'Using device: {str(device)}', force=True)

    tokenizer = RobertaTokenizer.from_pretrained(args.transformer_model)
    special_tokens = ["[E1]", "[\E1]", '[text]', "[NIL]"]
    sel_tokens = [f"[info{i}]" for i in range(args.info_token_num)]
    special_tokens += sel_tokens
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    args.tokenizer = tokenizer

    model = load_model(True, device, args.type_loss, args)
    num_train_samples = len(samples_train)
    if args.simpleoptim:
        optimizer, scheduler, num_train_steps, num_warmup_steps \
            = configure_optimizer_simple(args, model, num_train_samples)
    else:
        optimizer, scheduler, num_train_steps, num_warmup_steps \
            = configure_optimizer(args, model, num_train_samples)
    args.n_gpu = torch.cuda.device_count()
    model.to(device)
    dp = args.n_gpu > 1
    if dp:
        logger.log('Data parallel across {:d} GPUs {:s}'
                   ''.format(len(args.gpus.split(',')), args.gpus))
        model = nn.DataParallel(model)

    train_entities = load_entities(args.train_kb)
    logger.log('number of train entities {:d}'.format(len(train_entities)))
    dev_entities = load_entities(args.dev_kb)
    logger.log('number of dev entities {:d}'.format(len(dev_entities)))

    train_loader = get_attention_mention_loader(samples_train, train_entities, tokenizer, True, False, args)
    dev_loader = get_attention_mention_loader(samples_dev, dev_entities, tokenizer, False, True, args)

    effective_bsz = args.batch * args.gradient_accumulation_steps
    # train
    logger.log('***** train *****')
    logger.log('# train samples: {:d}'.format(num_train_samples))
    logger.log('# epochs: {:d}'.format(args.epochs))
    logger.log(' batch size : {:d}'.format(args.batch))
    logger.log(' gradient accumulation steps {:d}'
               ''.format(args.gradient_accumulation_steps))
    logger.log(
        ' effective training batch size with accumulation: {:d}'
        ''.format(effective_bsz))
    logger.log(' # training steps: {:d}'.format(num_train_steps))
    logger.log(' # warmup steps: {:d}'.format(num_warmup_steps))
    logger.log(' learning rate: {:g}'.format(args.lr))
    logger.log(' # parameters: {:d}'.format(count_parameters(model)))

    step_num = 0
    tr_loss, logging_loss = 0.0, 0.0
    start_epoch = 1
    model.zero_grad()

    if args.do_train:
        for epoch in range(start_epoch, args.epochs + 1):
            logger.log('\nEpoch {:d}'.format(epoch))
            epoch_start_time = datetime.now()

            epoch_train_start_time = datetime.now()
            train_loader = tqdm(train_loader)
            for step, batch in enumerate(train_loader):
                model.train()
                bsz = batch[0].size(0)
                batch = tuple(t.to(device) for t in batch)
                text_input_ids, text_attention_mask, can_input_ids, can_attention_mask, pos, labels = batch

                    # input_ids, attention_mask,labels = batch
                loss = model(text_input_ids, text_attention_mask, can_input_ids, can_attention_mask, pos, labels,
                                 "train")
                if dp:
                    loss = loss.sum() / bsz
                else:
                    loss /= bsz
                loss_avg = loss / args.gradient_accumulation_steps

                loss_avg.backward()
                tr_loss += loss_avg.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.clip)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    step_num += 1


            logger.log('training time for epoch {:3d} '
                       'is {:s}'.format(epoch, strtime(epoch_train_start_time)))

            hit1, hit5 = evaluate(model, dev_loader, device)
            logger.log('Done with epoch {:3d} | train loss {:8.4f} | '
                       'recall@1 {:8.4f}|'
                       'recall@5 {:8.4f}'
                       ' epoch time {} '.format(
                epoch,
                tr_loss / step_num,
                hit1,
                hit5,
                strtime(epoch_start_time)
            ))

            #
            save_model = (hit1 >= best_val_perf)
            if save_model:
                current_best = hit1
                logger.log('------- new best val perf: {:g} --> {:g} '
                           ''.format(best_val_perf, current_best))

                best_val_perf = current_best
                torch.save({'opt': args,
                            'sd': model.module.state_dict() if dp else model.state_dict(),
                            'perf': best_val_perf, 'epoch': epoch,
                            'opt_sd': optimizer.state_dict(),
                            'scheduler_sd': scheduler.state_dict(),
                            'tr_loss': tr_loss, 'step_num': step_num,
                            'logging_loss': logging_loss},
                           args.model[:-3] + f'_{epoch}'+ '.pt')
            else:
                logger.log('')
    if args.do_eval:
        model = load_model(False, device, args.type_loss, args).to(device)
        evaluate_test(model, tokenizer, device, args, logger)




def main(args):
    train_data = load_data(args.train_data)
    dev_data = load_data(args.dev_data)

    train(train_data, dev_data, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model",
                        default="model_disambiguation/zeshel_disambiguation_attention.pt")
    parser.add_argument("--transformer_model",
                        default="../roberta-base")
    parser.add_argument("--type_loss", type=str,
                        default="sum_log_nce",
                        choices=["log_sum", "sum_log", "sum_log_nce",
                                 "max_min", "bce_loss"])
    parser.add_argument("--max_len", default=512, type=int)
    parser.add_argument("--max_ent_len", default=256, type=int)
    parser.add_argument("--max_text_len", default=256, type=int)

    parser.add_argument("--train_data", default="data/train_candidates.json")
    parser.add_argument("--dev_data", default="data/dev_candidates.json")
    parser.add_argument("--train_kb", default="kb/train_kb.json")
    parser.add_argument("--dev_kb", default="kb/val_kb.json")

    parser.add_argument("--forgot_test_data", default="data/forgotten_realms.json")
    parser.add_argument("--lego_test_data", default="data/lego.json")
    parser.add_argument("--star_test_data", default="data/star_trek.json")
    parser.add_argument("--yugioh_test_data", default="data/yugioh.json")

    parser.add_argument("--forgot_kb", default="kb/forgotten_realms.json")
    parser.add_argument("--lego_kb", default="kb/lego.json")
    parser.add_argument("--star_kb", default="kb/star_trek.json")
    parser.add_argument("--yugioh_kb", default="kb/yugioh.json")


    parser.add_argument("--batch", default=2,type=int)
    parser.add_argument("--lr", default=4e-5, type=float)
    parser.add_argument("--epochs", default=10)
    parser.add_argument("--cand_num", default=56,type=int)
    parser.add_argument("--warmup_proportion", default=0.1)
    parser.add_argument("--weight_decay", default=0.01)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
    parser.add_argument("--seed", default=42)
    parser.add_argument("--num_workers", default=0)
    parser.add_argument("--simpleoptim", default=False)
    parser.add_argument("--clip", default=1)
    parser.add_argument("--info_token_num", default=3, type=int)
    parser.add_argument("--gpus", default="2,4")
    parser.add_argument("--logging_steps", default=100)
    parser.add_argument("--eval_step", default=10000, type=int)
    parser.add_argument("--do_train", action="store_true", default=False)
    parser.add_argument("--do_eval", action="store_true", default=False)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)
