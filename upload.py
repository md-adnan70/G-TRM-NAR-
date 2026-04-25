from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from common import trunc_normal_init_
from layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_L: torch.Tensor



@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense
    causal: bool = False

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=config.causal
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int, seq_len: int = None, device: torch.device = None):
        if seq_len is None:
            seq_len = self.config.seq_len
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_L=torch.empty(batch_size, seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Slice RoPE to match actual sequence length (supports dynamic-length generation)
        if seq_info["cos_sin"] is not None:
            actual_len = input_embeddings.shape[1]
            cos, sin = seq_info["cos_sin"]
            seq_info["cos_sin"] = (cos[:actual_len], sin[:actual_len])

        # Forward iterations
        it = 0
        z_L = carry.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L + input_embeddings, **seq_info)
                z_L = self.L_level(z_L, **seq_info)
        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L + input_embeddings, **seq_info)
        z_L = self.L_level(z_L, **seq_info)
        z_out = z_L

        # LM Outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_out)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_out[:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        seq_len = batch["inputs"].shape[1]
        device = batch["inputs"].device

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size, seq_len, device),

            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=device),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs



    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor,  max_new_tokens: int, temperature: float = 0.1, repetition_penalty: float = 1.2):
        self.eval()
        b_size = input_ids.shape[0]
        
        # We need to know the EOS token ID to stop properly
        # Usually 3 in HF Tokenizers, but you should verify your specific ID!
        eos_token_id = tokenizer.token_to_id("[EOS]")
        
        for step in range(max_new_tokens):
            batch = {
                "inputs": input_ids,
                "puzzle_identifiers": torch.zeros((b_size, self.config.num_puzzle_identifiers), device=input_ids.device, dtype=torch.int32)
            }
            carry = self.initial_carry(batch)
            
            for _ in range(self.config.halt_max_steps):
                carry, outputs = self.forward(carry, batch)
                if carry.halted.all():
                    break
            
            next_token_logits = outputs["logits"][:, -1, :].clone()
            
            # --- NEW: REPETITION PENALTY ---
            # Penalize tokens that have already been generated in the sequence
            for i in range(b_size):
                for token_id in set(input_ids[i].tolist()):
                    if next_token_logits[i, token_id] < 0:
                        next_token_logits[i, token_id] *= repetition_penalty
                    else:
                        next_token_logits[i, token_id] /= repetition_penalty
            
            # --- NEW: PREVENT IMMEDIATE EOS PANIC ---
            # Forbid the model from outputting EOS as the very first generated token
            if step == 0:
                next_token_logits[:, eos_token_id] = float('-inf')

            next_token_logits = next_token_logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # --- THE BRAKES ---
            if next_token.item() == eos_token_id: 
                break
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
        return input_ids
    # @torch.no_grad()
    # def generate_pseudo_ar_fixed(
    #     self,
    #     input_ids: torch.Tensor,
    #     max_new_tokens: int,
    #     *,
    #     tokenizer,
    #     **kwargs,
    # ):
    # def generate(self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 0.1, repetition_penalty: float = 1.2):
    #     self.eval()
    #     b_size = input_ids.shape[0]
    #     mask_token_id = tokenizer.token_to_id("[MASK]")
    #     pad_token_id = tokenizer.token_to_id("[PAD]")
        
    #     # We need to know the EOS token ID to stop properly
    #     # Usually 3 in HF Tokenizers, but you should verify your specific ID!
    #     eos_token_id = tokenizer.token_to_id("[EOS]")
        
    #     # --- THE CANVAS FIX ---
    #     # Create a padded universe so the bidirectional heads have a "right-hand wall"
    #     pads = torch.full((b_size, max_new_tokens), pad_token_id, dtype=torch.long, device=input_ids.device)
    #     current_sequence = torch.cat([input_ids, pads], dim=1)
        
    #     print("Decoding in Pseudo-AR Mode (Fixed Recursion + Canvas)...")

    #     for step in range(max_new_tokens):
    #         target_idx = input_ids.shape[1] + step
            
    #         # Place exactly ONE mask at the target generation spot
    #         current_sequence[:, target_idx] = mask_token_id
            
        
    #     for step in range(max_new_tokens):
    #         batch = {
    #             "inputs": input_ids,
    #             "puzzle_identifiers": torch.zeros((b_size, self.config.num_puzzle_identifiers), device=input_ids.device, dtype=torch.int32)
    #         }
    #         }
    #         carry = self.initial_carry(batch)
            
    #         # --- THE OVER-RECURSION FIX ---
    #         # Call forward EXACTLY ONCE, identically to your train_epoch logic.
    #         # Your internal model architecture handles the H_cycles naturally.
    #         _, outputs = self.forward(carry, batch)
                
    #         logits = outputs["logits"]
            
    #         # Isolate the logits strictly for the single mask
    #         mask_logits = logits[:, target_idx, :]
    #         mask_logits[:, mask_token_id] = -float('inf') 
    #         mask_logits[:, pad_token_id] = -float('inf') # Do not allow it to guess [PAD]
            
    #         # --- GREEDY DECODING ---
    #         # Remove temperature entropy. We want the network's most confident syntactic choice.
    #         predicted_token = torch.argmax(mask_logits, dim=-1)
            
    #         # Replace the mask with the predicted token
    #         current_sequence[:, target_idx] = predicted_token.squeeze(-1)
            
    #         # Stop early if the model naturally ends the story
    #         if (predicted_token == eos_token_id).any():
    #             break

    #     return current_sequence[:, input_ids.shape[1]:]

    # @torch.no_grad()
    # def generate_parallel_diluted(
    #     self,
    #     input_ids: torch.Tensor,
    #     max_new_tokens: int,
    #     iterations: int = 15, # 15 passes is a good sweet spot for TRMs
    #     *,
    #     tokenizer,
    #     **kwargs,
    # ):
    #     self.eval()
    #     b_size = input_ids.shape[0]

    #     mask_token_id = tokenizer.token_to_id("[MASK]")
    #     pad_token_id = tokenizer.token_to_id("[PAD]")
    #     eos_token_id = tokenizer.token_to_id("[EOS]")

    #     # --- THE PADDING DILUTION TRICK ---
    #     # Calculate how much padding we need to keep the mask ratio <= 40%
    #     prompt_len = input_ids.shape[1]
        
    #     # 0.40 = max_new_tokens / (prompt_len + max_new_tokens + pad_len)
    #     required_total_len = int(max_new_tokens / 0.40)
    #     pad_len = max(0, required_total_len - (prompt_len + max_new_tokens))
        
    #     # Construct the perfectly stable 40% masked canvas
    #     masks = torch.full((b_size, max_new_tokens), mask_token_id, dtype=torch.long, device=input_ids.device)
    #     pads = torch.full((b_size, pad_len), pad_token_id, dtype=torch.long, device=input_ids.device)
        
    #     current_sequence = torch.cat([input_ids, masks, pads], dim=1)
    #     gen_start_idx = prompt_len
    #     gen_end_idx = prompt_len + max_new_tokens
        
    #     print(f"Diluted Sequence Length: {current_sequence.shape[1]} (Mask Ratio: 40%)")

    #     # Iterative Parallel Unmasking
    #     for step in range(iterations):
    #         batch = {
    #             "inputs": current_sequence,
    #             "puzzle_identifiers": torch.zeros((b_size, self.config.num_puzzle_identifiers), device=input_ids.device, dtype=torch.int32)
    #         }
    #         carry = self.initial_carry(batch)
            
    #         # Ensure this matches your trained H_cycles!
    #         for _ in range(self.config.halt_max_steps): 
    #             carry, outputs = self.forward(carry, batch)
    #             if carry.halted.all():
    #                 break
                    
    #         logits = outputs["logits"]
    #         logits[:, :, mask_token_id] = -float('inf')
            
    #         probs = torch.softmax(logits, dim=-1)
    #         confidence, predicted_tokens = torch.max(probs, dim=-1)
            
    #         # Isolate our target generation zone (ignore prompt and padding)
    #         target_preds = predicted_tokens[:, gen_start_idx:gen_end_idx]
    #         target_conf = confidence[:, gen_start_idx:gen_end_idx]
            
    #         # NAR Repetition Penalty
    #         for b in range(b_size):
    #             for seq_idx in range(1, target_preds.shape[1]):
    #                 if target_preds[b, seq_idx] == target_preds[b, seq_idx-1]:
    #                     target_conf[b, seq_idx] *= 0.2
            
    #         # Unmasking Decay Schedule
    #         ratio_to_mask = 1.0 - ((step + 1) / iterations)
    #         num_to_mask = int(max_new_tokens * ratio_to_mask)
            
    #         if num_to_mask > 0:
    #             lowest_conf_indices = torch.topk(-target_conf, k=num_to_mask, dim=-1).indices
    #             current_sequence[:, gen_start_idx:gen_end_idx] = target_preds
    #             current_sequence[:, gen_start_idx:gen_end_idx].scatter_(1, lowest_conf_indices, mask_token_id)
    #         else:
    #             current_sequence[:, gen_start_idx:gen_end_idx] = target_preds

    #     # Post-process: Extract only the generated tokens and enforce EOS
    #     final_generation = current_sequence[:, gen_start_idx:gen_end_idx]
    #     for b in range(b_size):
    #         eos_positions = (final_generation[b] == eos_token_id).nonzero(as_tuple=True)[0]
    #         if len(eos_positions) > 0:
    #             final_generation[b, eos_positions[0]+1:] = pad_token_id

    #     return final_generation

    # @torch.no_grad()
    # def generate_chunked(
    #     self,
    #     input_ids: torch.Tensor,
    #     max_new_tokens: int,
    #     chunk_size: int = 8,
    #     iterations_per_chunk: int = 5,
    #     *,
    #     tokenizer,
    #     **kwargs,
    # ):
    #     self.eval()
    #     b_size = input_ids.shape[0]

    #     mask_token_id = tokenizer.token_to_id("[MASK]")
    #     pad_token_id = tokenizer.token_to_id("[PAD]")
    #     eos_token_id = tokenizer.token_to_id("[EOS]")
        
    #     current_sequence = input_ids.clone()
    #     tokens_generated = 0
        
    #     print(f"Generating {max_new_tokens} tokens in chunks of {chunk_size}...")

    #     while tokens_generated < max_new_tokens:
    #         # 1. Append only a small chunk of masks (Keeps Mask Ratio < 40%)
    #         current_chunk_size = min(chunk_size, max_new_tokens - tokens_generated)
    #         new_masks = torch.full((b_size, current_chunk_size), mask_token_id, dtype=torch.long, device=input_ids.device)
    #         current_sequence = torch.cat([current_sequence, new_masks], dim=1)
            
    #         chunk_start_idx = current_sequence.shape[1] - current_chunk_size
            
    #         # 2. Iterative Unmasking specifically for this chunk
    #         for step in range(iterations_per_chunk):
    #             batch = {
    #                 "inputs": current_sequence,
    #                 "puzzle_identifiers": torch.zeros((b_size, self.config.num_puzzle_identifiers), device=input_ids.device, dtype=torch.int32)
    #             }
    #             carry = self.initial_carry(batch)
                
    #             # Forward pass (Ensure your ACT loop matches your trained depth!)
    #             for _ in range(self.config.halt_max_steps):
    #                 carry, outputs = self.forward(carry, batch)
    #                 if carry.halted.all():
    #                     break
                
    #             logits = outputs["logits"]
    #             logits[:, :, mask_token_id] = -float('inf')
                
    #             probs = torch.softmax(logits, dim=-1)
    #             confidence, predicted_tokens = torch.max(probs, dim=-1)
                
    #             # Extract predictions for the current chunk
    #             chunk_preds = predicted_tokens[:, chunk_start_idx:]
    #             chunk_conf = confidence[:, chunk_start_idx:]
                
    #             # Repetition Penalty (Only penalize within the chunk to force vocabulary variety)
    #             for b in range(b_size):
    #                 for seq_idx in range(1, chunk_preds.shape[1]):
    #                     if chunk_preds[b, seq_idx] == chunk_preds[b, seq_idx-1]:
    #                         chunk_conf[b, seq_idx] *= 0.2 
                
    #             # Masking schedule for the chunk
    #             ratio_to_mask = 1.0 - ((step + 1) / iterations_per_chunk)
    #             num_to_mask = int(current_chunk_size * ratio_to_mask)
                
    #             if num_to_mask > 0:
    #                 lowest_conf_indices = torch.topk(-chunk_conf, k=num_to_mask, dim=-1).indices
    #                 current_sequence[:, chunk_start_idx:] = chunk_preds
    #                 current_sequence[:, chunk_start_idx:].scatter_(1, lowest_conf_indices, mask_token_id)
    #             else:
    #                 current_sequence[:, chunk_start_idx:] = chunk_preds
            
    #         tokens_generated += current_chunk_size
            
    #         # Early stopping if EOS is generated
    #         if (current_sequence == eos_token_id).any():
    #             break

    #     # Post-process to pad everything after the first EOS
    #     final_generation = current_sequence[:, input_ids.shape[1]:]
    #     for b in range(b_size):
    #         eos_positions = (final_generation[b] == eos_token_id).nonzero(as_tuple=True)[0]
    #         if len(eos_positions) > 0:
    #             first_eos = eos_positions[0]
    #             final_generation[b, first_eos+1:] = pad_token_id

    #     return final_generation

    # @torch.no_grad()
    # def generate(
    #     self,
    #     input_ids: torch.Tensor,
    #     max_new_tokens: int,
    #     iterations: int = 30,
    #     *,
    #     tokenizer,
    #     **kwargs,
    # ):
    #     self.eval()
    #     b_size = input_ids.shape[0]

    #     mask_token_id = tokenizer.token_to_id("[MASK]")
    #     pad_token_id = tokenizer.token_to_id("[PAD]")
    #     eos_token_id = tokenizer.token_to_id("[EOS]")
            
    #     current_masks = torch.full((b_size, max_new_tokens), mask_token_id, dtype=torch.long, device=input_ids.device)
    #     current_sequence = torch.cat([input_ids, current_masks], dim=1)
            
    #     for step in range(iterations):
    #         batch = {
    #             "inputs": current_sequence,
    #             "puzzle_identifiers": torch.zeros((b_size, self.config.num_puzzle_identifiers), device=input_ids.device, dtype=torch.int32)
    #             }
    #         carry = self.initial_carry(batch)
                
    #         for _ in range(self.config.halt_max_steps):
    #             carry, outputs = self.forward(carry, batch)
    #             if carry.halted.all():
    #                 break
                
    #         logits = outputs["logits"]
    #         logits[:, :, mask_token_id] = -float('inf')
                
    #         probs = torch.softmax(logits, dim=-1)
    #         confidence, predicted_tokens = torch.max(probs, dim=-1)
                
    #         gen_start_idx = input_ids.shape[1]
    #         generated_preds = predicted_tokens[:, gen_start_idx:]
    #         generated_conf = confidence[:, gen_start_idx:]
                
    #             # --- THE NAR REPETITION PENALTY ---
    #             # Artificially shatter the confidence of repeating tokens so they get re-masked
    #         for b in range(b_size):
    #             for seq_idx in range(1, generated_preds.shape[1]):
    #                 if generated_preds[b, seq_idx] == generated_preds[b, seq_idx-1]:
    #                     # If a token repeats, drop its confidence significantly
    #                     generated_conf[b, seq_idx] *= 0.2 
            
        #     for _ in range(self.config.halt_max_steps):
        #         carry, outputs = self.forward(carry, batch)
        #         if carry.halted.all():
        #             break
            
        #     next_token_logits = outputs["logits"][:, -1, :].clone()
            
        #     # --- NEW: REPETITION PENALTY ---
        #     # Penalize tokens that have already been generated in the sequence
        #     for i in range(b_size):
        #         for token_id in set(input_ids[i].tolist()):
        #             if next_token_logits[i, token_id] < 0:
        #                 next_token_logits[i, token_id] *= repetition_penalty
        #             else:
        #                 next_token_logits[i, token_id] /= repetition_penalty
            
        #     # --- NEW: PREVENT IMMEDIATE EOS PANIC ---
        #     # Forbid the model from outputting EOS as the very first generated token
        #     if step == 0:
        #         next_token_logits[:, eos_token_id] = float('-inf')

        #     next_token_logits = next_token_logits / temperature
        #     probs = F.softmax(next_token_logits, dim=-1)
        #     next_token = torch.multinomial(probs, num_samples=1)
            
        #     # --- THE BRAKES ---
        #     if next_token.item() == eos_token_id: 
        #         break
            
        #     input_ids = torch.cat([input_ids, next_token], dim=1)
            
        # return input_ids
    # @torch.no_grad()
    # def generate(
    #     self,
    #     input_ids: torch.Tensor,
    #     max_new_tokens: int,
    #     iterations: int = 30,
    #     *,
    #     tokenizer,
    #     **kwargs,
    # ):
    #     self.eval()
    #     b_size = input_ids.shape[0]

    #     mask_token_id = tokenizer.token_to_id("[MASK]")
    #     pad_token_id = tokenizer.token_to_id("[PAD]")
    #     eos_token_id = tokenizer.token_to_id("[EOS]")
            
    #     current_masks = torch.full((b_size, max_new_tokens), mask_token_id, dtype=torch.long, device=input_ids.device)
    #     current_sequence = torch.cat([input_ids, current_masks], dim=1)
            
    #     for step in range(iterations):
    #         batch = {
    #             "inputs": current_sequence,
    #             "puzzle_identifiers": torch.zeros((b_size, self.config.num_puzzle_identifiers), device=input_ids.device, dtype=torch.int32)
    #             }
    #         carry = self.initial_carry(batch)
                
    #         for _ in range(self.config.halt_max_steps):
    #             carry, outputs = self.forward(carry, batch)
    #             if carry.halted.all():
    #                 break
                
    #         logits = outputs["logits"]
    #         logits[:, :, mask_token_id] = -float('inf')
                
    #         probs = torch.softmax(logits, dim=-1)
    #         confidence, predicted_tokens = torch.max(probs, dim=-1)
                
    #         gen_start_idx = input_ids.shape[1]
    #         generated_preds = predicted_tokens[:, gen_start_idx:]
    #         generated_conf = confidence[:, gen_start_idx:]
                
    #             # --- THE NAR REPETITION PENALTY ---
    #             # Artificially shatter the confidence of repeating tokens so they get re-masked
    #         for b in range(b_size):
    #             for seq_idx in range(1, generated_preds.shape[1]):
    #                 if generated_preds[b, seq_idx] == generated_preds[b, seq_idx-1]:
    #                     # If a token repeats, drop its confidence significantly
    #                     generated_conf[b, seq_idx] *= 0.2 
                
    #             # The [EOS] Enforcer
    #         for b in range(b_size):
    #             eos_positions = (generated_preds[b] == eos_token_id).nonzero(as_tuple=True)[0]
    #             if len(eos_positions) > 0:
    #                 first_eos = eos_positions[0]
    #                 generated_preds[b, first_eos+1:] = pad_token_id
    #                 generated_conf[b, first_eos+1:] = 1.0 
    #             # The [EOS] Enforcer
    #         for b in range(b_size):
    #             eos_positions = (generated_preds[b] == eos_token_id).nonzero(as_tuple=True)[0]
    #             if len(eos_positions) > 0:
    #                 first_eos = eos_positions[0]
    #                 generated_preds[b, first_eos+1:] = pad_token_id
    #                 generated_conf[b, first_eos+1:] = 1.0 
                
    #         ratio_to_mask = 1.0 - ((step + 1) / iterations)
    #         num_to_mask = int(max_new_tokens * ratio_to_mask)
    #         ratio_to_mask = 1.0 - ((step + 1) / iterations)
    #         num_to_mask = int(max_new_tokens * ratio_to_mask)
                
    #         if num_to_mask > 0:
    #             lowest_conf_indices = torch.topk(-generated_conf, k=num_to_mask, dim=-1).indices
    #             current_sequence[:, gen_start_idx:] = generated_preds
    #             current_sequence[:, gen_start_idx:].scatter_(1, lowest_conf_indices, mask_token_id)
    #         else:
    #             current_sequence[:, gen_start_idx:] = generated_preds
    #         if num_to_mask > 0:
    #             lowest_conf_indices = torch.topk(-generated_conf, k=num_to_mask, dim=-1).indices
    #             current_sequence[:, gen_start_idx:] = generated_preds
    #             current_sequence[:, gen_start_idx:].scatter_(1, lowest_conf_indices, mask_token_id)
    #         else:
    #             current_sequence[:, gen_start_idx:] = generated_preds

    #     final_generation = current_sequence[:, input_ids.shape[1]:]
    #     return final_generation
    #     final_generation = current_sequence[:, input_ids.shape[1]:]
    #     return final_generation