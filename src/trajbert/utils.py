import numpy as np
import torch

def make_coocurrence_matrix(token_list, token_size, alpha=98, theta=1000):
    token_list = [list(filter(lambda x: x > 3, token)) for token in token_list]
    exchange_matrix = np.zeros(shape=(token_size, token_size))
    for token in token_list:
        for i in range(1, len(token)):
            if token[i] == token[i - 1]:
                continue
            exchange_matrix[token[i - 1]][token[i]] += 1
    
    # smoothing
    exchange_matrix = np.where(exchange_matrix >= alpha, exchange_matrix, 0) 
    exchange_matrix = exchange_matrix / theta
    exchange_matrix = np.where(exchange_matrix > 0, np.exp(exchange_matrix), 0)
    
    # row normalization
    for i in range(token_size):
        row_sum = sum(exchange_matrix[i]) + np.exp(1)
        for j in range(token_size):
            if exchange_matrix[i][j] != 0:
                exchange_matrix[i][j] = exchange_matrix[i][j] / row_sum
    
    # diagonal equal 1
    for i in range(token_size):
        exchange_matrix[i][i] = 1

    # simetrization
    for i in range(token_size):
        for j in range(token_size):
            exchange_matrix[i][j] = max(exchange_matrix[i][j], exchange_matrix[j][i])
    return exchange_matrix

def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        # Decay learning rate by half every epoch
        lr = args.lr * (0.5 ** ((epoch - 1) // 1))
    elif args.lradj == 'type2':
        # Fixed schedule for specific epochs
        lr_schedule = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
        lr = lr_schedule.get(epoch, None)  # Return None if epoch is not in the schedule
    elif args.lradj == 'type0':
        # Keep learning rate constant
        lr = args.lr
    else:
        raise ValueError(f"Unknown learning rate adjustment type: {args.lradj}")

    # Update learning rate if it was adjusted
    if lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr}')
    else:
        print(f'Epoch {epoch}: No learning rate adjustment applied.')

def topk(ground_truth, logits_lm, k):
    pred_topk = logits_lm[:, :, 0:k]
    pred_topk = torch.flatten(pred_topk, start_dim=0, end_dim=1).cpu().data.numpy()
    topk_token = 0
    for i in range(len(ground_truth)):
        if ground_truth[i] in pred_topk[i]:
            topk_token += 1
    topk_score = topk_token / len(ground_truth)
    return topk_token, topk_score

def map_score(ground_truth, logits_lm):
    MAP = 0
    pred_topk = torch.flatten(logits_lm, start_dim=0, end_dim=1).cpu().data.numpy()
    for i in range(len(ground_truth)):
        if ground_truth[i] in pred_topk[i]:
            a = ground_truth[i]
            b = pred_topk[i]
            rank = np.argwhere(ground_truth[i] == pred_topk[i]) + 1
            MAP += 1.0 / rank[0][0]
    return MAP / len(ground_truth)

def get_evalution(ground_truth, logits_lm, exchange_matrix, input_id = None,mask_len=0):
    pred_acc = logits_lm[:, :, 0]
    pred_acc = pred_acc.flatten().cpu().data.numpy()
    accuracy_token = 0
    wrong_pre = []
    per_acu = 0
    for i in range(len(ground_truth)):
        if pred_acc[i] == ground_truth[i]:
            accuracy_token += 1
            per_acu += 1
        else:
            wrong_pre.append('pre: ' + str(pred_acc[i]) + ', true: ' + str(ground_truth[i]))

    accuracy_score = accuracy_token / len(ground_truth)
    print("top1:", accuracy_token, accuracy_score)

    pred_acc = logits_lm[:, :, 0]
    pred_acc = pred_acc.flatten().cpu().data.numpy()

    fuzzy_accuracy_token = 0
    for i in range(len(pred_acc)):
        a = int(pred_acc[i])
        b = ground_truth[i]

        if exchange_matrix[b][a] > 0 or exchange_matrix[a][b] > 0:
            fuzzy_accuracy_token += 1
    fuzzy_score = fuzzy_accuracy_token / len(ground_truth)
    print("fuzzy:", fuzzy_accuracy_token, fuzzy_score)

    top3_token, top3_score = topk(ground_truth, logits_lm, 3)
    print("top3:", top3_token, top3_score)

    top5_token, top5_score = topk(ground_truth, logits_lm, 5)
    print("top5:", top5_token, top5_score)

    top10_token, top10_score = topk(ground_truth, logits_lm, 10)
    print("top10:", top10_token, top10_score)

    top30_token, top30_score = topk(ground_truth, logits_lm, 30)
    print("top30:", top30_token, top30_score)

    top50_token, top50_score = topk(ground_truth, logits_lm, 50)
    print("top50:", top50_token, top50_score)

    top100_token, top100_score = topk(ground_truth, logits_lm, 100)
    print("top100:", top100_token, top100_score)

    MAP = map_score(ground_truth, logits_lm)
    print("MAP score:", MAP)

    return accuracy_score, fuzzy_score, top3_score, top5_score, top10_score, top30_score, top50_score, top100_score, MAP, wrong_pre