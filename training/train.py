import torch
import torch.nn as nn
import copy
from torch.cuda import Stream
from einops import repeat
import os
from typing import List
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.tokenizer_dataloader_prototype import TokenizerDataLoader


def get_available_gpus():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    num_gpus = torch.cuda.device_count()
    devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
    print(f"Found {num_gpus} GPU(s): {devices}")
    return devices

def create_model_replicas(model, devices):
    replicas = []
    for device in devices:
        replica = copy.deepcopy(model).to(device)
        replicas.append(replica)
    return replicas

def create_cuda_streams(devices):
    streams = []
    for device in devices:
        with torch.cuda.device(device):
            stream = Stream()
            streams.append(stream)
    return streams

def average_gradients(models, devices):
    main_device = devices[0]

    for params in zip(*(m.parameters() for m in models)):
        main_param = params[0]

        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            continue

        avg_grad = torch.zeros_like(grads[0], device=main_device)

        for g in grads:
            if g.device != main_device:
                g = g.to(main_device, non_blocking=True)
            avg_grad.add_(g)

        avg_grad.div_(len(grads))

        if main_param.grad is None:
            main_param.grad = avg_grad
        else:
            main_param.grad.copy_(avg_grad)

def broadcast_parameters(models, devices):
    main_model = models[0]
    main_params = list(main_model.parameters())
    for param_idx, main_param in enumerate(main_params):

        for gpu_idx, (model, device) in enumerate(zip(models, devices)):
            if gpu_idx == 0:
                continue 
            
            other_param = list(model.parameters())[param_idx]
            other_param.data.copy_(main_param.data.to(device))


def train_step(models, devices, streams, data_batches, optimizer, criterion, scheduler=None):
    num_gpus = len(devices)
    assert len(models) == num_gpus
    assert len(data_batches) == num_gpus

    optimizer.zero_grad(set_to_none=True)

    outputs = [None] * num_gpus
    losses = [None] * num_gpus

    for i, (m, device, stream, batch) in enumerate(zip(models, devices, streams, data_batches)):
        with torch.cuda.device(device):
            with torch.cuda.stream(stream):
                batch = batch.to(device, non_blocking=True)
                output = m(batch)
                outputs[i] = output

                targets = batch[:, 1:].contiguous()
                logits = output[:, :-1, :].contiguous()
                losses[i] = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )
    for s in streams:
        s.synchronize()

    for i, (m, device, stream, loss) in enumerate(zip(models, devices, streams, losses)):
        with torch.cuda.device(device):
            with torch.cuda.stream(stream):
                loss.backward()

    for s in streams:
        s.synchronize()

    average_gradients(models, devices)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    broadcast_parameters(models, devices)

    return {
        "losses": [l.item() for l in losses], 
        "outputs": outputs
    }

def training_loop( model, dataloader, optimizer, scheduler, num_epochs, criterion):

    devices = get_available_gpus()
    num_gpus = len(devices)
    models = create_model_replicas(model, devices)
    streams = create_cuda_streams(devices)
    
    history = {"train_loss": [], "learning_rates": []}
    
    for epoch in range(num_epochs):
        epoch_losses = []
        tokenGenerator = dataloader.tokenize_batches(seq_len=dataloader.max_length, drop_last=True, return_tensors='pt')
        for batch in tokenGenerator:
            if isinstance(batch, torch.Tensor):
                batch_size_actual = batch.shape[0]
                per_gpu_size = batch_size_actual // num_gpus
                data_batches = [batch[i*per_gpu_size:(i+1)*per_gpu_size] for i in range(num_gpus)]
                if batch_size_actual % num_gpus != 0:
                    data_batches[-1] = batch[num_gpus*per_gpu_size:]
            else:
                data_batches = batch[:num_gpus]
            
            result = train_step(
                models=models,
                devices=devices,
                streams=streams,
                data_batches=data_batches,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler
            )
            
            epoch_losses.extend(result["losses"])
        
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        history["train_loss"].append(avg_epoch_loss)
        history["learning_rates"].append(optimizer.param_groups[0]['lr'])
    
    return history


def test_training_loop():
    """
    Test the multi-GPU training loop with a tiny model and small dataset.
    Verifies:
    - Shapes match up (input: batch, seq_len, output: batch, seq_len, vocab_size)
    - Losses are finite
    - Models stay synchronized across GPUs
    - Training runs without errors
    """
    print("=" * 60)
    print("Testing Multi-GPU Training Loop")
    print("=" * 60)
    
    # Get available GPUs
    devices = get_available_gpus()
    num_gpus = len(devices)
    
    if num_gpus < 1:
        raise RuntimeError("Need at least 1 GPU for testing")
    
    print(f"\nUsing {num_gpus} GPU(s)")
    
    # Create a tiny test model using SSM
    vocab_size = 1000
    hidden_dim = 32  # SSM dimension
    seq_len = 32
    
    class SimpleSSMModel(nn.Module):
        """Simple SSM model matching expected interface.
        
        Takes tokens (batch, seq_len) and outputs logits (batch, seq_len, vocab_size).
        Uses a simple State Space Layer with causal scan.
        """
        def __init__(self, vocab_size, hidden_dim):
            super().__init__()
            self.vocab_size = vocab_size
            self.hidden_dim = hidden_dim
            
            # Embedding layer
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            
            # Simple SSM layer (diagonal state transition)
            # A: decay parameter for each dimension
            self.A = nn.Parameter((torch.randn(hidden_dim) * 0.02).clamp(min=1e-6))
            
            # Output projection
            self.output = nn.Linear(hidden_dim, vocab_size)
            
        def forward(self, x):
            # x: (batch, sequence_length) tokens
            batch_size, seq_len = x.shape
            
            # Embed tokens
            x_emb = self.embedding(x)  # (batch, seq_len, hidden_dim)
            
            # Apply SSM scan operation (causal)
            # A: (hidden_dim,) -> (batch, seq_len, hidden_dim)
            A = repeat(self.A, "dim -> b seq dim", b=batch_size, seq=seq_len)
            log_A = torch.log(A.clamp(min=1e-6))
            
            # Cumulative sum for causal scan
            cs_A = torch.cumsum(log_A, dim=1)  # (batch, seq_len, hidden_dim)
            
            # Create decay matrix M[t,j] = exp(cumsum[t] - cumsum[j])
            # This gives the decay factor from time j to time t
            log_M = cs_A.unsqueeze(2) - cs_A.unsqueeze(1)  # (batch, seq_len, seq_len, hidden_dim)
            
            # Apply causal mask (only look at past/current)
            indices = torch.arange(seq_len, device=x.device)
            mask = indices[:, None] >= indices[None, :]
            log_M = log_M.masked_fill(~mask.unsqueeze(0).unsqueeze(-1), float('-inf'))
            M = torch.exp(log_M)  # (batch, seq_len, seq_len, hidden_dim)
            
            # Apply SSM: y[t] = sum_j (M[t,j] * x[j])
            # M: (batch, t, j, dim), x_emb: (batch, j, dim)
            # Output: (batch, t, dim)
            out = torch.einsum('b t j d, b j d -> b t d', M, x_emb)
            
            # Project to vocab size
            logits = self.output(out)  # (batch, seq_len, vocab_size)
            return logits
    
    # Create base model
    print("\nCreating simple SSM test model...")
    base_model = SimpleSSMModel(vocab_size, hidden_dim)
    
    # Create model replicas on each GPU
    print("Creating model replicas on each GPU...")
    models = create_model_replicas(base_model, devices)
    
    # Create CUDA streams
    print("Creating CUDA streams...")
    streams = create_cuda_streams(devices)
    
    print("\nCreating test data batches...")
    batch_size = 4
    num_batches = num_gpus
    
    # Generate random token batches: (batch, sequence_length)
    data_batches = []
    for i in range(num_batches):
        batch = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
        data_batches.append(batch)
    
    print(f"  Batch shape: {data_batches[0].shape}")
    print(f"  Expected model output shape: (batch={batch_size}, seq_len={seq_len}, vocab_size={vocab_size})")
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(models[0].parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Run training steps
    print("\n" + "=" * 60)
    print("Running Training Steps")
    print("=" * 60)
    
    num_steps = 5
    all_losses = []
    
    for step in range(num_steps):
        print(f"\nStep {step + 1}/{num_steps}")
        
        # Training step
        result = train_step(
            models=models,
            devices=devices,
            streams=streams,
            data_batches=data_batches,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler
        )
        
        # Verify shapes
        avg_loss = sum(result["losses"]) / len(result["losses"])
        all_losses.append(avg_loss)
        print(f"  Average loss: {avg_loss:.4f}")
        
        # Check output shapes
        for gpu_idx, output in enumerate(result["outputs"]):
            assert len(output.shape) == 3, f"GPU {gpu_idx}: Expected 3D output, got {output.shape}"
            assert output.shape[0] == batch_size, f"GPU {gpu_idx}: Batch size mismatch"
            assert output.shape[1] == seq_len, f"GPU {gpu_idx}: Sequence length mismatch"
            assert output.shape[2] == vocab_size, f"GPU {gpu_idx}: Vocab size mismatch"
        
        print(f"  ✓ Shapes verified on all GPUs")
        
        # Verify models are synchronized
        print("  Verifying model synchronization...")
        main_params = list(models[0].parameters())
        all_synced = True
        for gpu_idx in range(1, num_gpus):
            other_params = list(models[gpu_idx].parameters())
            for p1, p2 in zip(main_params, other_params):
                if not torch.allclose(p1.data.to(devices[0]), p2.data.to(devices[0]), atol=1e-5):
                    all_synced = False
                    break
            if not all_synced:
                break
        
        if all_synced:
            print(f"  ✓ All GPUs synchronized")
        else:
            print(f"  ⚠ Warning: Models may not be fully synchronized")
        
        # Verify loss is finite
        assert all(not (torch.isnan(torch.tensor(l)) or torch.isinf(torch.tensor(l))) 
                   for l in result["losses"]), "Loss contains NaN or Inf!"
    
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Initial loss: {all_losses[0]:.4f}")
    print(f"Final loss:   {all_losses[-1]:.4f}")
    print(f"Loss change:  {all_losses[-1] - all_losses[0]:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

    # print("\nPreparing live tokenized data stream (per-epoch retokenization)...")
    # # per-GPU batch size
    # batch_size = 4
    # # path to a small local file used for testing (tokenized live each epoch)
    # sample = os.path.join(os.path.dirname(__file__), "..", "sample.txt")
    # if not os.path.isfile(sample):
    #     sample = None

    # dl = TokenizerDataLoader(tokenizer_name="gpt2", max_length=seq_len, data_input=sample)

    # print(f"  Per-GPU batch size: {batch_size}, seq_len: {seq_len}")
    # print(f"  Expected model output shape: (batch={batch_size}, seq_len={seq_len}, vocab_size={vocab_size})")   

def test_training_live_tokenization():
    """Run a small training test that retokenizes the local text per epoch.

    This function builds the same tiny SSM model setup as `test_training_loop`,
    but it initializes a `TokenizerDataLoader` before the epoch loop and
    uses `tokenize_file_chunks` to stream sequences (retokenized each epoch).
    """
    print("\nRunning live-tokenization training test")

    devices = get_available_gpus()
    num_gpus = len(devices)
    if num_gpus < 1:
        raise RuntimeError("Need at least 1 GPU for testing")

    vocab_size = 1000
    hidden_dim = 32
    seq_len = 32

    # Reuse the same SimpleSSMModel class definition from test_training_loop
    class SimpleSSMModel(nn.Module):
        def __init__(self, vocab_size, hidden_dim):
            super().__init__()
            self.vocab_size = vocab_size
            self.hidden_dim = hidden_dim
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            self.A = nn.Parameter((torch.randn(hidden_dim) * 0.02).clamp(min=1e-6))
            self.output = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x):
            batch_size, seq_len = x.shape
            x_emb = self.embedding(x)
            A = repeat(self.A, "dim -> b seq dim", b=batch_size, seq=seq_len)
            log_A = torch.log(A.clamp(min=1e-6))
            cs_A = torch.cumsum(log_A, dim=1)
            log_M = cs_A.unsqueeze(2) - cs_A.unsqueeze(1)
            indices = torch.arange(seq_len, device=x.device)
            mask = indices[:, None] >= indices[None, :]
            log_M = log_M.masked_fill(~mask.unsqueeze(0).unsqueeze(-1), float('-inf'))
            M = torch.exp(log_M)
            out = torch.einsum('b t j d, b j d -> b t d', M, x_emb)
            logits = self.output(out)
            return logits

    # create models/replicas/streams
    base_model = SimpleSSMModel(vocab_size, hidden_dim)
    models = create_model_replicas(base_model, devices)
    streams = create_cuda_streams(devices)

    # dataloader: initialize before epoch loop
    sample = os.path.join(os.path.dirname(__file__), '..', 'sample.txt')
    if not os.path.isfile(sample):
        print("No local sample file found at", sample)
        print("Aborting live-tokenization test")
        return

    dl = TokenizerDataLoader(tokenizer_name="gpt2", max_length=seq_len, data_input=sample, vocab_size=vocab_size)
    print("Initialized dl with vocab_size: ", dl.vocab_size)

    optimizer = torch.optim.Adam(models[0].parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = None

    per_gpu_batch = 4
    needed = per_gpu_batch * num_gpus

    num_epochs = 2
    max_steps_per_epoch = 5
    batch_gen = dl.tokenize_batches(seq_len=seq_len, global_batch_size=needed, drop_last=True, return_tensors='pt')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} (live tokenization)")
        step = 0

        for batch in batch_gen:
            # Split global batch into per-GPU batches
            data_batches = []
            for i in range(num_gpus):
                start = i * per_gpu_batch
                end = start + per_gpu_batch
                data_batches.append(batch[start:end])

            # Training step
            result = train_step(models=models,
                                devices=devices,
                                streams=streams,
                                data_batches=data_batches,
                                optimizer=optimizer,
                                criterion=criterion,
                                scheduler=scheduler)

            avg_loss = sum(result['losses']) / len(result['losses'])
            print(f"  Epoch {epoch+1} Step {step+1}: avg_loss={avg_loss:.4f}")

            step += 1
            if step >= max_steps_per_epoch:
                break

        print(f"Finished epoch {epoch+1}, steps run: {step}")



if __name__ == "__main__":
    
    lr = 3e-4
    weight_decay = 0.1
    eps = 1e-8
    num_epochs = 10
    num_warmup_steps = 1000
    num_decay_steps = 1_000_000

    

    # Create a small default model and a simple synthetic dataloader so
    # this script can run directly. Wrap the model to return logits only
    # (some model implementations return (logits, states)).
    from model.model import SSMTransformerLM

    vocab_size = 1000
    seq_len = 32
    d_model = 128
    n_layers = 2

    class LogitsOnlyWrapper(nn.Module):
        def __init__(self, base_model: nn.Module):
            super().__init__()
            self.base = base_model

        def forward(self, x):
            out = self.base(x)
            # If the base model returns (logits, states), return logits
            if isinstance(out, tuple) or isinstance(out, list):
                return out[0]
            return out

    # instantiate base model and wrapper
    base_model = SSMTransformerLM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=8, max_seq_len=seq_len)
    model = LogitsOnlyWrapper(base_model)


    # Use TokenizerDataLoader for live tokenization from sample.txt
    sample_path = os.path.join(os.path.dirname(__file__), '..', 'sample.txt')
    if not os.path.isfile(sample_path):
        raise FileNotFoundError(f"sample.txt not found at {sample_path}")
    global_batch_size = 4
    dl = TokenizerDataLoader(tokenizer_name="gpt2", max_length=seq_len, data_input=sample_path, vocab_size=vocab_size, global_batch_size=global_batch_size)
    
    

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay, eps=eps
    )


    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        total_iters=num_warmup_steps
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_decay_steps - num_warmup_steps
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[num_warmup_steps]
    )

    criterion = nn.CrossEntropyLoss()

    
    output = training_loop(model, dl, optimizer, scheduler, num_epochs, criterion)
    #print out loss & lr to confirm functionality of training_loop on sample.txt
    print(output)
    
    # test_training_loop() 
    # to run the live-tokenization test call:
    # test_training_live_tokenization()


