import torch
from src.Utilities.GenerationUtilities import token_ids_to_text, text_to_token_ids, generate_text_simple
import math

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    By default, the calc_loss_loader function iterates over all batches in a given data loader, accumulates the loss in
    the total_loss variable, and then computes and averages the loss over the total number of batches. Alternatively,
    we can specify a smaller number of batches via num_batches to speed up the evaluation during model training.
    :param data_loader:
    :param model:
    :param device:
    :param num_batches:
    :return:
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:           # iterates over all batches if not specified
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))    # reduces num_batches if it exceeds the length of the loader
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()   # sums loss for each batch
        else:
            break
    return total_loss / num_batches     # averages loss across all batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()            # dropout is disabled during evaluation
    with torch.no_grad():   # no need to track gradients during evaluation
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
        model=model, idx=encoded,
        max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))              # compact print format
    model.train()

def train_model_simple(model, train_loader, val_loader,
               optimizer, device, num_epochs,
               eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []    # initializes lists to store losses and tokens seen
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):             # starts main training loop
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()               # resets gradients from previous batch iteration
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()                     # calculates loss gradient
            optimizer.step()                    # updates model parameters using loss gradient
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:            # optional evaluation
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                      )
        generate_and_print_sample(              # prints a sample after each epoch
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen

def train_model(model, train_loader, val_loader, optimizer, device,
        n_epochs, eval_freq, eval_iter, start_context, tokenizer,
        warmup_steps, initial_lr=3e-05, min_lr=1e-6):
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1

    # Retrieves the initial learning rate from the optimizer, assuming we use it as the peak learning rate
    peak_lr = optimizer.param_groups[0]["lr"]
    total_training_steps = len(train_loader) * n_epochs     # Calculates the total number of training steps
    lr_increment = (peak_lr - initial_lr) / warmup_steps    # Calculates the learning rate increment per step during warmup
    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            if global_step < warmup_steps:                  # Adjusts the learning rate based on the current step using (warmup or cosine annealing)
                lr = initial_lr + global_step * lr_increment
            else:
                progress = ((global_step - warmup_steps) /
                            (total_training_steps - warmup_steps))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (
                    1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:      # Updates the learning rate in the optimizer
                param_group["lr"] = lr
            track_lrs.append(lr)
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            if global_step >= warmup_steps:     # Applies gradient clipping to prevent exploding gradients after warmup
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

            # Everything else is the same as in the simple training loop
            optimizer.step()
            tokens_seen += input_batch.numel()
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader,
                    device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Iter {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen, track_lrs