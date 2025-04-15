from tqdm.auto import tqdm
import torch
import os
import numpy as np
import random
import mlflow
import torch.nn as nn
import torch.optim as optim


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_model(model, dataloader, tokenizer, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for src, trg in tqdm(dataloader):
            src = src.to(device)
            trg = trg.to(device)

            optimizer.zero_grad()

            output = model(src, trg[:, :-1])
            loss = criterion(output.view(-1, output.shape[-1]), trg[:, 1:].reshape(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        mlflow.log_metric("train_loss", total_loss / len(dataloader))
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')


def generate_summary(model, tokenizer, text, max_len=250):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([tokenizer.encode(text).ids]).to(device)

        embedded = model.embedding(input_tensor)
        encoder_outputs, (hidden, cell) = model.encoder(embedded)

        trg_indexes = [tokenizer.token_to_id('[BOS]')]

        for _ in range(max_len):

            context_vector, _ = model.attention(encoder_outputs, hidden[-1])

            trg_tensor = torch.tensor([[trg_indexes[-1]]]).to(device)
            embedded_trg = model.embedding(trg_tensor)

            decoder_input = torch.cat([embedded_trg, context_vector.unsqueeze(1)], dim=2)

            output, (hidden, cell) = model.decoder(decoder_input, (hidden, cell))
            prediction = model.fc(output[:, -1, :]).argmax(1).item()

            if prediction == tokenizer.token_to_id('[EOS]'):
                break

            trg_indexes.append(prediction)

        return tokenizer.decode(trg_indexes[1:])