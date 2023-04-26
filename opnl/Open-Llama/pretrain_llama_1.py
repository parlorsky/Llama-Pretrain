import os
import time
import torch
import random
import sentencepiece as spm
from torchinfo import summary
from torch.utils.data import DataLoader
from deepspeed.ops.adam import FusedAdam
from transformers import LlamaForCausalLM, LlamaConfig, get_cosine_schedule_with_warmup

from validation import val_set
from tokenizer import Tokenizer
from data_iter import create_shard_kwargs, DataIter
from collate_fn import collate_fn_gen
from pretrain_dataset import preprocess_the_pile_gen
from pretrain_config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log_interval *= 1
eval_interval *= 1
save_interval *= 1

sp_model = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
tokenizer = Tokenizer(sp_model)

paths = create_shard_kwargs(patterns)
random.shuffle(paths)
transform_dict = {
    "pile": preprocess_the_pile_gen(tokenizer, max_length),
}
data_set = DataIter(
    paths,
    transform_dict=transform_dict,
    concat_docs=False,
)
train_loader = DataLoader(
    data_set,
    batch_size=train_batch_size,
    num_workers=0,
    collate_fn=collate_fn_gen(tokenizer, max_length),
    drop_last=True,
)

raw_model = LlamaForCausalLM(
    LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        initializer_range=initializer_range,
        pad_token_id=tokenizer.pad_id,
        rms_norm_eps=1e-5,
        hidden_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        use_stable_embedding=True,
        shared_input_output_embedding=True,
    )
)


#тут происходит killed
#чтобы продолжить обучать модель, нужно загрузить старые веса, а они не влезают.
# raw_model.load_state_dict(torch.load("7b/model/consolidated.00.pth"))


raw_model = raw_model.to(device)
raw_model.eval()
with torch.no_grad():
    summary(raw_model, input_data=torch.ones(1, 64, dtype=torch.int64).to(device))

no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in raw_model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": weight_decay,
    },
    {
        "params": [
            p
            for n, p in raw_model.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]
optim = FusedAdam(optimizer_grouped_parameters, lr=lr, betas=(0.9, 0.95))
optim.zero_grad()

scheduler = get_cosine_schedule_with_warmup(
    optim,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)


    
_ = train_loader
model = raw_model.to(device)
optim = optim
scheduler = scheduler
print("start training...")
train_loader_iter = iter(train_loader)
global_step = 0
start_time = time.time()


for data_step in range(num_training_steps):
    model.train()
    batch = next(train_loader_iter)
    for k, v in batch.items():
        batch[k] = v.to(device, non_blocking=True)
    out = model(**batch, labels=batch["input_ids"])
    total_loss = out.loss
    losses = {"total_loss": total_loss}
    total_loss.backward()
    optim.step()
    scheduler.step()
    optim.zero_grad()
    global_step += 1
    
    if data_step % log_interval == 0 and data_step > 0:
        cost_time = time.time() - start_time
        start_time = time.time()
        tokens = train_batch_size * log_interval * max_length
        print({"Training/Token per second per gpu": tokens / cost_time})
        for k, v in losses.items():
            print({"Losses/{}".format(k): v})
        current_lr = optim.param_groups[0]["lr"]
        print({"Training/LR": current_lr})
        if optim.scaler is not None:
            print({"Training/Loss Scale": optim.scaler.get_scale()})
        print({"Training/Data Step": data_step})
        print({"Training/Global Step": global_step})

    if data_step % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            for data in val_set:
                raw_inputs = data
                inputs_len = len(raw_inputs)
                inputs = tokenizer(raw_inputs, return_tensors=True, add_special_tokens=False)
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
                pred = model.generate(
                    **inputs, max_new_tokens=256, do_sample=True, repetition_penalty=2.0
                )
                pred = tokenizer.decode(pred.cpu())[0]
                pred = pred[inputs_len:]

    if data_step % save_interval == 0 and data_step > 0:
        if not os.path.isdir(work_dir):
            os.mkdir(work_dir)
        torch.save(raw_model.state_dict(), "{}/{}.pt".format(work_dir, global_step))
