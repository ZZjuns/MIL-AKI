import torch


def rank_constraint(data, label, model, A, n_classes, label_positive_list, label_negative_list):
    loss_rank = torch.tensor(0.0).to(data.device)
    for c in range(n_classes):
        if label == c:
            value, indice = torch.topk(A[0, c], k=1)
            h = data[indice.item(): indice.item() + 1] # top feature
            if label_positive_list[c].full():
                _ = label_positive_list[c].get()
            label_positive_list[c].put(h)
            if label_negative_list[c].empty():
                loss_rank = loss_rank + torch.tensor(0.0).to(data.device)
            else:
                h = label_negative_list[c].get()
                label_negative_list[c].put(h)
                Ah, _, _, _ = model(h.detach())
                if c != 0:
                    loss_rank = loss_rank + torch.clamp(torch.mean(Ah[0, c] - value), min=0.0) + torch.clamp(torch.mean(-value), min=0.0) + torch.clamp(torch.mean(Ah[0, c]), min=0.0)
                else:
                    loss_rank = loss_rank + torch.clamp(torch.mean(-value), min=0.0) + torch.clamp(torch.mean(Ah[0, c]), min=0.0)
        else:
            value, indice = torch.topk(A[0, c], k=1)
            h = data[indice.item(): indice.item() + 1] # top feature
            if label_negative_list[c].full():
                _ = label_negative_list[c].get()
            label_negative_list[c].put(h)
            if label_positive_list[c].empty():
                loss_rank = loss_rank + torch.tensor(0.0).to(data.device)
            else:
                h = label_positive_list[c].get()
                label_positive_list[c].put(h)
                Ah, _, _, _ = model(h.detach())
                if c != 0:
                    loss_rank = loss_rank + torch.clamp(torch.mean(value - Ah[0, c]), min=0.0) + torch.clamp(torch.mean(value), min=0.0)
                else:
                    loss_rank = loss_rank + torch.clamp(torch.mean(value), min=0.0) + torch.clamp(torch.mean(-Ah[0, c]), min=0.0)

    loss_rank = loss_rank / n_classes
    return loss_rank, label_positive_list, label_negative_list


def rank_loss(data, label, model, A, n_classes, label_positive_list, label_negative_list):
    loss_rank = torch.tensor(0.0).to(data.device)
    for c in range(n_classes):
        if label == c:
            value, indice = torch.topk(A[0, c], k=1)
            h = data[indice.item(): indice.item() + 1] # top feature
            if label_positive_list[c].full():
                _ = label_positive_list[c].get()
            label_positive_list[c].put(h)
            if label_negative_list[c].empty():
                loss_rank = loss_rank + torch.tensor(0.0).to(data.device)
            else:
                h = label_negative_list[c].get()
                label_negative_list[c].put(h)
                Ah, _, _, _ = model(h.detach())
                if c != 0:
                    loss_rank = loss_rank + torch.clamp(torch.mean(Ah[0, c] - value), min=0.0) + torch.clamp(torch.mean(-value), min=0.0) + torch.clamp(torch.mean(Ah[0, c]), min=0.0)
                else:
                    loss_rank = loss_rank + torch.clamp(torch.mean(-value), min=0.0) + torch.clamp(torch.mean(Ah[0, c]), min=0.0)
        else:
            value, indice = torch.topk(A[0, c], k=1)
            h = data[indice.item(): indice.item() + 1] # top feature
            if label_negative_list[c].full():
                _ = label_negative_list[c].get()
            label_negative_list[c].put(h)
            if label_positive_list[c].empty():
                loss_rank = loss_rank + torch.tensor(0.0).to(data.device)
            else:
                h = label_positive_list[c].get()
                label_positive_list[c].put(h)
                Ah, _, _, _ = model(h.detach())
                if c != 0:
                    loss_rank = loss_rank + torch.clamp(torch.mean(value - Ah[0, c]), min=0.0) + torch.clamp(torch.mean(value), min=0.0)
                else:
                    loss_rank = loss_rank + torch.clamp(torch.mean(value), min=0.0) + torch.clamp(torch.mean(-Ah[0, c]), min=0.0)

    loss_rank = loss_rank / n_classes
    return loss_rank, label_positive_list, label_negative_list