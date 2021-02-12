from multilingual_ner.metrics import FMeasureStorage
from seqeval.metrics import performance_measure

import torch
import torch.nn as nn

def clear_tags(labels, predictions, idx2tag, batch_element_length):
    """ this function removes <PAD>, CLS and SEP tags at each sentence
        and convert both ids of tags and batch elements to SeqEval input format
        [[first sentence tags], [second sentence tags], ..., [last sentence tags]]"""

    clear_labels = []
    clear_predictions = []

    sentence_labels = []
    sentence_predictions = []

    sentence_length = 0

    for idx in range(len(labels)):
        if labels[idx] != 0:
            sentence_labels.append(idx2tag[labels[idx]])
            sentence_predictions.append(idx2tag[predictions[idx]])
            sentence_length += 1

            if sentence_length == batch_element_length:
                # not including the 0 and the last element of list, because of CLS and SEP tokens
                clear_labels.append(sentence_labels[1: len(sentence_labels) - 1])
                clear_predictions.append(sentence_predictions[1: len(sentence_predictions) - 1])
                sentence_labels = []
                sentence_predictions = []
                sentence_length = 0
        else:
            if sentence_labels:
                clear_labels.append(sentence_labels[1: len(sentence_labels) - 1])
                clear_predictions.append(sentence_predictions[1: len(sentence_predictions) - 1])
                sentence_labels = []
                sentence_predictions = []
            else:
                pass

    return clear_labels, clear_predictions

def train_epoch(model, criterion, optimizer, data, tag2idx, idx2tag, device, scheduler):
    epoch_loss = 0
    epoch_metrics = FMeasureStorage()

    model.train()

    for batch in data:
        tokens = batch[0].to(device)
        tags = batch[1].to(device)

        batch_element_length = len(tags[0])

        predictions = model(tokens)
        predictions = predictions.view(-1, predictions.shape[-1])

        tags_mask = tags != tag2idx['<PAD>']
        tags_mask = tags_mask.view(-1)
        labels = torch.where(tags_mask, tags.view(-1), torch.tensor(criterion.ignore_index).type_as(tags))

        loss = criterion(predictions, labels)

        predictions = predictions.argmax(dim=1)

        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

        # clear <PAD>, CLS and SEP tags from both labels and predictions
        clear_labels, clear_predictions = clear_tags(labels, predictions, idx2tag, batch_element_length)

        iteration_result = performance_measure(clear_labels, clear_predictions)

        epoch_metrics + iteration_result
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if scheduler:
            scheduler.step()
        torch.cuda.empty_cache()

    epoch_f1_score, epoch_precision, epoch_recall = epoch_metrics.report()
    print('Train Loss = {:.5f}, F1-score = {:.3%}, Precision = {:.3%}, Recall = {:.3%}'.format(epoch_loss / len(data),
                                                                                               epoch_f1_score,
                                                                                               epoch_precision,
                                                                                               epoch_recall))


def valid_epoch(model, criterion, data, tag2idx, idx2tag, device):
    epoch_loss = 0
    epoch_metrics = FMeasureStorage()

    model.eval()

    with torch.no_grad():
        for batch in data:
            tokens = batch[0].to(device)
            tags = batch[1].to(device)

            batch_element_length = len(tags[0])

            predictions = model(tokens)
            predictions = predictions.view(-1, predictions.shape[-1])
            tags_mask = tags != tag2idx['<PAD>']
            tags_mask = tags_mask.view(-1)
            labels = torch.where(tags_mask, tags.view(-1), torch.tensor(criterion.ignore_index).type_as(tags))

            loss = criterion(predictions, labels)

            predictions = predictions.argmax(dim=1)

            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy()
            masks = masks.cpu().numpy()

            # clear <PAD>, CLS and SEP tags from both labels and predictions
            clear_labels, clear_predictions = clear_tags(labels, predictions, idx2tag, batch_element_length)

            iteration_result = performance_measure(clear_labels, clear_predictions)

            epoch_metrics + iteration_result
            epoch_loss += loss.item()

    epoch_f1_score, epoch_precision, epoch_recall = epoch_metrics.report()
    print('Valid Loss = {:.5f}, F1-score = {:.3%}, Precision = {:.3%}, Recall = {:.3%}'.format(epoch_loss / len(data),
                                                                                               epoch_f1_score,
                                                                                               epoch_precision,
                                                                                               epoch_recall))


def test_epoch(model, criterion, data, tag2idx, idx2tag, device):
    epoch_loss = 0
    epoch_metrics = FMeasureStorage()

    model.eval()

    with torch.no_grad():
        for batch in data:
            tokens = batch[0].to(device)
            tags = batch[1].to(device)

            batch_element_length = len(tags[0])

            predictions = model(tokens)
            predictions = predictions.view(-1, predictions.shape[-1])
            tags_mask = tags != tag2idx['<PAD>']
            tags_mask = tags_mask.view(-1)
            labels = torch.where(tags_mask, tags.view(-1), torch.tensor(criterion.ignore_index).type_as(tags))

            loss = criterion(predictions, labels)

            predictions = predictions.argmax(dim=1)

            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy()
            masks = masks.cpu().numpy()

            # clear <PAD>, CLS and SEP tags from both labels and predictions
            clear_labels, clear_predictions = clear_tags(labels, predictions, idx2tag, batch_element_length)

            iteration_result = performance_measure(clear_labels, clear_predictions)

            epoch_metrics + iteration_result
            epoch_loss += loss.item()

    epoch_f1_score, epoch_precision, epoch_recall = epoch_metrics.report()
    print('Test  Loss = {:.5f}, F1-score = {:.3%}, Precision = {:.3%}, Recall = {:.3%}'.format(epoch_loss / len(data),
                                                                                               epoch_f1_score,
                                                                                               epoch_precision,
                                                                                               epoch_recall))


def train_model(model, criterion, optimizer, train_data, eval_data, tag2idx, idx2tag, device, scheduler, epochs=1):
    for epoch in range(epochs):
        print('Epoch {} / {}'.format(epoch + 1, epochs))
        train_epoch(model, criterion, optimizer, train_data, tag2idx, idx2tag, device, scheduler)
        valid_epoch(model, criterion, eval_data, tag2idx, idx2tag, device)