# ------------------------------------------------------------------------
# Semi-Autoregressive RoadNetTransformer Head
# Reuses ARRNTRHead structure but plugs a block-causal attention mask and
# a transformer that accepts tgt_mask (e.g., LssPlBzTransformer).
# ------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmdet3d.registry import MODELS, TASK_UTILS

from .ar_rntr_head import ARRNTRHead, PryDecoderEmbeddings, MLP


def _build_block_causal_mask(T: int, block_len: int, device) -> torch.Tensor:
    """Build a block-level causal mask of shape [T, T].

    Semantics:
    - Tokens can attend to any token in previous blocks and within the same block.
    - Tokens cannot attend to tokens in future blocks.

    Args:
        T: total sequence length.
        block_len: number of tokens per block (e.g., clause_length).
        device: torch device.
    Returns:
        mask (Tensor): [T, T], 0 for allowed, -inf for masked (as in nn.MultiheadAttention).
    """
    if T <= 0:
        return torch.zeros(0, 0, device=device)
    num_blocks = math.ceil(T / max(block_len, 1))
    mask = torch.full((T, T), float('-inf'), device=device)
    for b in range(num_blocks):
        start = b * block_len
        end = min((b + 1) * block_len, T)
        # allow attending to all tokens up to current block end
        mask[start:end, :end] = 0.0
    return mask


def _build_group_mask_from_ids(group_ids: torch.Tensor, allow_mlm_intra: bool = True) -> torch.Tensor:
    """Build a [T, T] attention mask from per-token group ids.

    Rule: token i can attend to any token j in previous groups (gj < gi), and
    within the same group either fully (MLM style) or causally (j <= i) per flag.
    Returns a float mask with 0 for allow and -inf for masked.
    """
    T = int(group_ids.numel())
    gids = group_ids.view(T)
    gi = gids.view(T, 1).expand(T, T)
    gj = gids.view(1, T).expand(T, T)
    allow = gj < gi
    same = gj == gi
    if allow_mlm_intra:
        allow = allow | same
    else:
        # causal inside group: j <= i
        idx = torch.arange(T, device=gids.device)
        causal = idx.view(1, T).expand(T, T) <= idx.view(T, 1).expand(T, T)
        allow = allow | (same & causal)
    mask = torch.full((T, T), float('-inf'), device=gids.device)
    mask[allow] = 0.0
    return mask


def _expand_group_ids_from_pos(T: int, clause_length: int, sar_group_ids_pos: torch.Tensor) -> torch.Tensor:
    """Expand positive-only per-token group ids to full length [T].

    Layout assumption: seq = [START] + POS_TOKENS + NEG_TOKENS (padded/noise).
    - Assign START group 0.
    - Copy positive token group ids to next positions (min(pos_len, T-1)).
    - Remaining tokens are negatives; assign new group ids per clause, i.e.,
      repeat id for `clause_length` tokens, increasing group id per clause.
    """
    device = sar_group_ids_pos.device
    gids = torch.zeros((T,), device=device, dtype=torch.long)
    # place positives
    pos_len = int(min(int(sar_group_ids_pos.numel()), max(0, T - 1)))
    if pos_len > 0:
        gids[1:1 + pos_len] = sar_group_ids_pos[:pos_len]
        max_gid = int(sar_group_ids_pos[:pos_len].max().item())
    else:
        max_gid = 0
    # assign negatives per clause
    neg_tokens = T - 1 - pos_len
    if clause_length <= 0 or neg_tokens <= 0:
        return gids
    num_neg_clauses = (neg_tokens + clause_length - 1) // clause_length
    base = 1 + pos_len
    for i in range(num_neg_clauses):
        s = base + i * clause_length
        e = min(s + clause_length, T)
        gids[s:e] = max_gid + 1 + i
    return gids


@MODELS.register_module()
class SARRNTRHead(ARRNTRHead):
    """Semi-Autoregressive variant of ARRNTRHead.

    Differences vs ARRNTRHead:
    - Uses a transformer that accepts tgt_mask (e.g., LssPlBzTransformer).
    - Builds a block-causal mask per batch using clause_length as block size.
    - Training: full-seq forward with block mask to enable within-block parallelism.
    - Inference: keeps token-by-token decoding for stability (can be extended to block decoding later).
    """

    def __init__(self,
                 *args,
                 sar_group_clauses: int = 1,
                 sar_group_strategy: str = 'contiguous',  # ['contiguous','meta_groups']
                 sar_intra_group_mlm: bool = True,
                 # keypoint branch configs
                 kp_num_query: int = 200,
                 kp_num_classes: int = 4,
                 kp_transformer: dict = None,
                 # prompt coupling
                 kp_prompt_enable: bool = True,
                 kp_prompt_detach: bool = True,
                 kp_prompt_type: str = 'add',  # ['add', 'none']
                 kp_prompt_topk: int = 50,
                 kp_prompt_weighted: bool = True,
                 # SD-Map prompt
                 sd_prompt_enable: bool = True,
                 sd_prompt_type: str = 'cross',  # ['cross','none']
                 # NAR inference
                 nar_infer: bool = False,
                 nar_max_len: int = None,
                 # NAR-MLM training & iterative inference controls
                 nar_mlm_train: bool = False,
                 nar_mask_ratio: float = 0.9,
                 nar_iters: int = 1,
                 nar_conf_thresh: float = 0.5,
                 nar_keep_ratio: float = 0.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # number of clauses per block. default 1 (per-clause parallel intra-block)
        self.sar_group_clauses = max(1, int(sar_group_clauses))
        self.sar_group_strategy = sar_group_strategy
        self.sar_intra_group_mlm = bool(sar_intra_group_mlm)

        # ---------- Keypoint parallel branch ----------
        self.kp_num_query = int(kp_num_query)
        self.kp_num_classes = int(kp_num_classes)
        if kp_transformer is None:
            kp_transformer = dict(type='KeypointTransformer',
                                  decoder=dict(
                                      type='PETRTransformerLineDecoder',
                                      return_intermediate=True,
                                      num_layers=3,
                                      transformerlayers=dict(
                                          type='PETRLineTransformerDecoderLayer',
                                          attn_cfgs=[
                                              dict(
                                                  type='RNTR2MultiheadAttention',
                                                  embed_dims=self.embed_dims,
                                                  num_heads=8,
                                                  dropout=0.1),
                                              dict(
                                                  type='RNTR2MultiheadAttention',
                                                  embed_dims=self.embed_dims,
                                                  num_heads=8,
                                                  dropout=0.1),
                                          ],
                                          ffn_cfgs=dict(
                                              type='FFN',
                                              embed_dims=self.embed_dims,
                                              feedforward_channels=self.embed_dims * 4,
                                              num_fcs=2,
                                              ffn_drop=0.1,
                                              act_cfg=dict(type='ReLU', inplace=True),
                                          ),
                                          with_cp=False,
                                          operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
                                      ),
                                  ))
        self.kp_transformer = MODELS.build(kp_transformer)
        self.kp_query_embed = nn.Embedding(self.kp_num_query, self.embed_dims)
        self.kp_cls_head = Linear(self.embed_dims, self.kp_num_classes)
        self.kp_reg_head = MLP(self.embed_dims, self.embed_dims, 2, 3)
        # prompt adapter
        self.kp_prompt_enable = bool(kp_prompt_enable)
        self.kp_prompt_detach = bool(kp_prompt_detach)
        self.kp_prompt_type = kp_prompt_type
        if self.kp_prompt_enable and self.kp_prompt_type == 'add':
            self.kp_prompt_adapter = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims)
            )
        # pos embedding for KP coordinates if needed in future (e.g., cross-attn prompt_pos)
        self.kp_pos_mlp = nn.Sequential(
            nn.Linear(2, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims)
        )
        self.kp_prompt_topk = int(kp_prompt_topk)
        self.kp_prompt_weighted = bool(kp_prompt_weighted)

        # ---------- SD-Map prompt controls ----------
        self.sd_prompt_enable = bool(sd_prompt_enable)
        self.sd_prompt_type = sd_prompt_type
        # NAR controls
        self.nar_infer = bool(nar_infer)
        self.nar_max_len = None if nar_max_len is None else int(nar_max_len)
        # NAR-MLM train and iterative inference
        self.nar_mlm_train = bool(nar_mlm_train)
        try:
            self.nar_mask_ratio = float(nar_mask_ratio)
        except Exception:
            self.nar_mask_ratio = 0.9
        self.nar_iters = max(1, int(nar_iters))
        try:
            self.nar_conf_thresh = float(nar_conf_thresh)
        except Exception:
            self.nar_conf_thresh = 0.5
        try:
            self.nar_keep_ratio = float(nar_keep_ratio)
        except Exception:
            self.nar_keep_ratio = 0.0
        
        # Special token IDs (aligned with model layer constants)
        self.no_known = 575  # padding/unknown token
        self.start = 574     # START token
        self.end = 573       # END token

    def _gather_sd_prompt(self, img_metas, device, embed_dims):
        """Collect sd_prompt_pos from img_metas and map to prompt/prompt_pos.

        Returns:
            prompt (Tensor): [B, K, D]
            prompt_pos (Tensor): [B, K, D]
        or (None, None) if not available/disabled.
        """
        if not self.sd_prompt_enable or self.sd_prompt_type != 'cross':
            return None, None
        try:
            # 1) Prefer sd_prompt_seq if present in all metas
            seq_list = []
            all_have_seq = True
            for m in img_metas:
                s = m.get('sd_prompt_seq', None)
                if s is None:
                    all_have_seq = False
                    break
                seq_list.append(torch.as_tensor(s, device=device, dtype=torch.long).view(-1))
            if all_have_seq and len(seq_list) > 0:
                # unify by min length to avoid padding complexity
                K = min(int(s.numel()) for s in seq_list)
                if K > 0:
                    tokens = torch.stack([s[:K] for s in seq_list], dim=0)  # [B, K]
                    # embed tokens into D
                    prompt = self.embedding.word_embeddings(tokens)  # [B, K, D]
                    # zero pos as a generic prompt position embedding
                    prompt_pos = torch.zeros_like(prompt, dtype=prompt.dtype, device=prompt.device)
                    return prompt, prompt_pos

            # 2) Fallback: coordinate prompt positions
            pos_list = []
            for m in img_metas:
                sp = m.get('sd_prompt_pos', None)
                if sp is None:
                    return None, None
                sp = torch.as_tensor(sp, device=device, dtype=torch.float32)
                if sp.ndim != 2 or sp.shape[1] != 2:
                    return None, None
                pos_list.append(sp)
            K = min(p.shape[0] for p in pos_list)
            if K <= 0:
                return None, None
            pos_bt = torch.stack([p[:K] for p in pos_list], dim=0)  # [B,K,2]
            prompt_pos = self.kp_pos_mlp(pos_bt)  # [B,K,D]
            prompt = prompt_pos
            return prompt, prompt_pos
        except Exception:
            return None, None

    def forward(self, mlvl_feats, input_seqs, img_metas):
        """Forward function with SAR mask.

        Args:
            mlvl_feats (Tensor): BEV features, shape [B, C, H, W].
            input_seqs (Tensor): token ids, shape [B, T].
            img_metas (list[dict]): Contains 'n_control' for clause_length.
        Returns:
            If training: logits for all layers [L, B, T, V].
            If infer: (decoded_seq [B, T_out], values [B, T_steps]).
        """
        x = mlvl_feats
        if self.in_channels != self.embed_dims:
            x = self.bev_proj(x)
        pos_embed = self.bev_position_encoding(x)
        B, _, H, W = x.shape
        masks = torch.zeros(B, H, W).bool().to(x.device)

        # compute clause_length from meta
        try:
            n_control = int(img_metas[0].get('n_control', 3))
        except Exception:
            n_control = 3
        coeff_dim = max(0, (n_control - 2) * 2)
        clause_length = 4 + coeff_dim
        block_len = clause_length * self.sar_group_clauses

        if self.training:
            tgt = self.embedding(input_seqs.long())  # [B, T, D]
            query_embed = self.embedding.position_embeddings.weight  # [T_max, D]
            # Keypoint prompt (global) injection (additive baseline)
            if self.kp_prompt_enable and self.kp_prompt_type == 'add':
                try:
                    kp_q = self.kp_query_embed.weight
                    kp_dec, _ = self.kp_transformer(x, masks, kp_q, pos_embed)
                    kp_feats = torch.nan_to_num(kp_dec)[-1]  # [B, Q, D]
                    # select top-k keypoints by cls confidence if available
                    kp_feats_cls = kp_feats.detach() if self.kp_prompt_detach else kp_feats
                    kp_cls_logits = self.kp_cls_head(kp_feats_cls)  # [B, Q, C]
                    with torch.no_grad():
                        kp_scores = kp_cls_logits.softmax(-1).max(dim=-1)[0]  # [B, Q]
                        k = max(1, min(self.kp_prompt_topk, kp_scores.shape[1]))
                        topk_scores, topk_idx = torch.topk(kp_scores, k=k, dim=-1)
                    # gather features
                    B = kp_feats.shape[0]
                    gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, kp_feats.shape[-1])  # [B, k, D]
                    kp_feats_prompt = kp_feats.detach() if self.kp_prompt_detach else kp_feats
                    kp_sel = torch.gather(kp_feats_prompt, 1, gather_idx)
                    if self.kp_prompt_weighted:
                        w = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-6)
                        kp_global = (kp_sel * w.unsqueeze(-1)).sum(dim=1)  # [B, D]
                    else:
                        kp_global = kp_sel.mean(dim=1)
                    kp_bias = self.kp_prompt_adapter(kp_global)  # [B, D]
                    tgt = tgt + kp_bias.unsqueeze(1)
                except Exception:
                    pass
            T = tgt.shape[1]
            # ---------------- SAR FLOW DEBUG (head: shapes & params) ----------------
            try:
                print(f"[SAR FLOW] Head: B={B}, T={T}, n_control={n_control}, clause_length={clause_length}, block_len={block_len}, strategy={self.sar_group_strategy}, intra_mlm={self.sar_intra_group_mlm}")
                if isinstance(img_metas, (list, tuple)) and B > 0:
                    for bi in range(min(2, B)):
                        meta_i = img_metas[bi] if isinstance(img_metas[bi], dict) else {}
                        gpos = meta_i.get('sar_group_ids_pos', None)
                        gfull = meta_i.get('sar_group_ids', None)
                        gpos_len = int(len(gpos)) if gpos is not None else None
                        gfull_len = int(len(gfull)) if gfull is not None else None
                        print(f"[SAR FLOW] Head: meta[{bi}] sar_group_ids_pos_len={gpos_len}, sar_group_ids_len={gfull_len}")
            except Exception as e:
                print(f"[SAR FLOW] Head: param debug error: {e}")
            # -----------------------------------------------------------------------
            # Build batch-wise block/group mask [B, T, T]
            use_meta_groups = (self.sar_group_strategy == 'meta_groups' and isinstance(img_metas, (list, tuple)))
            masks_bt = []
            if use_meta_groups:
                for bi in range(B):
                    meta = img_metas[bi] if isinstance(img_metas, (list, tuple)) else {}
                    gids_full = None
                    if isinstance(meta, dict) and 'sar_group_ids_pos' in meta:
                        sar_pos = meta['sar_group_ids_pos']
                        if not torch.is_tensor(sar_pos):
                            sar_pos = torch.as_tensor(sar_pos, device=tgt.device, dtype=torch.long)
                        gids_full = _expand_group_ids_from_pos(T, clause_length, sar_pos)
                    elif isinstance(meta, dict) and 'sar_group_ids' in meta:
                        sar_full = meta['sar_group_ids']
                        if torch.is_tensor(sar_full) and sar_full.numel() == T:
                            gids_full = sar_full.to(tgt.device).long()
                        elif isinstance(sar_full, (list, tuple)) and len(sar_full) == T:
                            gids_full = torch.as_tensor(sar_full, device=tgt.device, dtype=torch.long)
                    if gids_full is None:
                        masks_bt = []
                        break
                    m = _build_group_mask_from_ids(gids_full, allow_mlm_intra=self.sar_intra_group_mlm)
                    masks_bt.append(m)
            if len(masks_bt) == B and len(masks_bt) > 0:
                tgt_mask = torch.stack(masks_bt, dim=0)
            else:
                base_mask = _build_block_causal_mask(T, block_len, tgt.device)
                tgt_mask = base_mask.unsqueeze(0).repeat(B, 1, 1)
            # ---------------- SAR FLOW DEBUG (head: mask summary) ----------------
            try:
                def _mask_stats(m):
                    allow = (m == 0).sum().item()
                    total = int(m.numel())
                    ratio = float(allow) / float(total) if total > 0 else 0.0
                    return allow, total, ratio
                print(f"[SAR FLOW] Head: tgt_mask.shape={tuple(tgt_mask.shape)}, use_meta_groups={use_meta_groups}")
                for bi in range(min(2, B)):
                    mb = masks_bt[bi] if (len(masks_bt) == B and len(masks_bt) > 0) else tgt_mask[bi]
                    a, tot, r = _mask_stats(mb)
                    print(f"[SAR FLOW] Head: sample[{bi}] mask allow={a}/{tot} ({r:.4f})")
            except Exception as e:
                print(f"[SAR FLOW] Head: mask debug error: {e}")
            # ---------------------------------------------------------------------
            # For NAR-MLM finetuning, use full-visible mask
            if getattr(self, 'nar_mlm_train', False):
                tgt_mask = torch.zeros_like(tgt_mask)
            # Try cross-attention prompt if requested and supported (prefer SD-Map over KP)
            transformer_name = self.transformer.__class__.__name__
            cross_family = transformer_name in (
                'LssPlPrySeqLineTransformer', 'LssMLMPlPrySeqLineTransformer', 'LssSARPrmSeqLineTransformer')
            sd_prompt, sd_prompt_pos = (None, None)
            if cross_family:
                sd_prompt, sd_prompt_pos = self._gather_sd_prompt(img_metas, tgt.device, self.embed_dims)

            def _call_transformer_with_sign(fname: str, tgt_, x_, tgt_mask_, masks_, q_embed_, pos_embed_, prm=None, prm_pos=None):
                """Dispatch to transformer with correct signature using 3D tgt."""
                if fname == 'LssSARPrmSeqLineTransformer':
                    # supports tgt_mask and optional prompt
                    if prm is not None and prm_pos is not None:
                        return self.transformer(tgt_, x_, tgt_mask_, masks_, q_embed_, pos_embed_, prm, prm_pos)
                    return self.transformer(tgt_, x_, tgt_mask_, masks_, q_embed_, pos_embed_)
                elif fname in ('LssPlPrySeqLineTransformer', 'LssMLMPlPrySeqLineTransformer'):
                    # signature: (tgt, x, prompt, mask, query_embed, pos_embed, prompt_pos)
                    tgt_cross = tgt_.unsqueeze(1)
                    return self.transformer(tgt_cross, x_, prm, masks_, q_embed_, pos_embed_, prm_pos)
                else:
                    # DETR-like: (tgt, x, mask, query_embed, pos_embed)
                    return self.transformer(tgt_, x_, masks_, q_embed_, pos_embed_)

            try:
                print(f"[SAR FLOW] Head: call transformer={transformer_name} with tgt_mask.shape={tuple(tgt_mask.shape)}")
                if cross_family and sd_prompt is not None and sd_prompt_pos is not None:
                    outs_dec, _ = _call_transformer_with_sign(transformer_name, tgt, x, tgt_mask, masks, query_embed, pos_embed, sd_prompt, sd_prompt_pos)
                else:
                    # fallback to KP cross prompt if configured
                    use_kp_cross = (self.kp_prompt_enable and self.kp_prompt_type == 'cross' and cross_family)
                    if use_kp_cross:
                        kp_q = self.kp_query_embed.weight
                        kp_dec, _ = self.kp_transformer(x, masks, kp_q, pos_embed)
                        kp_feats = torch.nan_to_num(kp_dec)[-1]  # [B, Q, D]
                        kp_cls_logits = self.kp_cls_head(kp_feats)  # [B, Q, C]
                        kp_coords_norm = torch.sigmoid(self.kp_reg_head(kp_feats))  # [B, Q, 2]
                        with torch.no_grad():
                            kp_scores = kp_cls_logits.softmax(-1).max(dim=-1)[0]
                            k = max(1, min(self.kp_prompt_topk, kp_scores.shape[1]))
                            topk_scores, topk_idx = torch.topk(kp_scores, k=k, dim=-1)
                        gather_idx_d = topk_idx.unsqueeze(-1).expand(-1, -1, kp_feats.shape[-1])
                        gather_idx_p = topk_idx.unsqueeze(-1).expand(-1, -1, kp_coords_norm.shape[-1])
                        prompt = torch.gather(kp_feats, 1, gather_idx_d)
                        prompt_pos = torch.gather(kp_coords_norm, 1, gather_idx_p)
                        prompt_pos = self.kp_pos_mlp(prompt_pos)
                        outs_dec, _ = _call_transformer_with_sign(transformer_name, tgt, x, tgt_mask, masks, query_embed, pos_embed, prompt, prompt_pos)
                    else:
                        outs_dec, _ = _call_transformer_with_sign(transformer_name, tgt, x, tgt_mask, masks, query_embed, pos_embed, None, None)
            except TypeError:
                # Be defensive to signature mismatch
                outs_dec, _ = _call_transformer_with_sign(transformer_name, tgt, x, tgt_mask, masks, query_embed, pos_embed, None, None)
            outs_dec = torch.nan_to_num(outs_dec)
            try:
                print(f"[SAR FLOW] Head: outs_dec.shape={tuple(outs_dec.shape)} (expect [L,B,T,D])")
            except Exception:
                pass
            out = self.vocab_embed(outs_dec)
            try:
                print(f"[SAR FLOW] Head: logits(out).shape={tuple(out.shape)} (expect [L,B,T,V])")
            except Exception:
                pass
            return out
        else:
            # NAR iterative inference (if enabled); else keep AR loop
            if getattr(self, 'nar_infer', False):
                B = input_seqs.shape[0]
                T = 1 + int(getattr(self, 'max_iteration', 0))
                if getattr(self, 'nar_max_len', None) is not None:
                    T = min(T, int(self.nar_max_len))
                max_pos = int(self.embedding.position_embeddings.weight.shape[0])
                T = min(T, max_pos)
                # Initialize with [START] + [NO_KNOWN]
                seq = torch.full((B, T), int(self.no_known), device=x.device, dtype=torch.long)
                seq[:, 0] = int(self.start)

                # Precompute additive KP bias (optional)
                kp_bias = None
                if self.kp_prompt_enable and self.kp_prompt_type == 'add':
                    try:
                        kp_q = self.kp_query_embed.weight
                        kp_dec, _ = self.kp_transformer(x, masks, kp_q, pos_embed)
                        kp_feats = torch.nan_to_num(kp_dec)[-1]
                        kp_feats_cls = kp_feats.detach() if self.kp_prompt_detach else kp_feats
                        kp_cls_logits = self.kp_cls_head(kp_feats_cls)
                        with torch.no_grad():
                            kp_scores = kp_cls_logits.softmax(-1).max(dim=-1)[0]
                            k = max(1, min(self.kp_prompt_topk, kp_scores.shape[1]))
                            topk_scores, topk_idx = torch.topk(kp_scores, k=k, dim=-1)
                        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, kp_feats.shape[-1])
                        kp_feats_prompt = kp_feats.detach() if self.kp_prompt_detach else kp_feats
                        kp_sel = torch.gather(kp_feats_prompt, 1, gather_idx)
                        if self.kp_prompt_weighted:
                            w = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-6)
                            kp_global = (kp_sel * w.unsqueeze(-1)).sum(dim=1)
                        else:
                            kp_global = kp_sel.mean(dim=1)
                        kp_bias = self.kp_prompt_adapter(kp_global)
                    except Exception:
                        kp_bias = None

                query_embed = self.embedding.position_embeddings.weight
                transformer_name = self.transformer.__class__.__name__
                # Prefer SD-Map cross prompt
                sd_prompt, sd_prompt_pos = (None, None)
                if transformer_name in ('LssPlPrySeqLineTransformer', 'LssMLMPlPrySeqLineTransformer', 'LssSARPrmSeqLineTransformer'):
                    sd_prompt, sd_prompt_pos = self._gather_sd_prompt(img_metas, x.device, self.embed_dims)

                last_values = None
                iters = max(1, int(getattr(self, 'nar_iters', 1)))

                # 统一签名分发（含 LssPlBzTransformer），tgt 始终为 3D [B,T,D]
                def _call_transformer_with_sign(fname: str, tgt_, x_, tgt_mask_, masks_, q_embed_, pos_embed_, prm=None, prm_pos=None):
                    if fname == 'LssSARPrmSeqLineTransformer':
                        if prm is not None and prm_pos is not None:
                            return self.transformer(tgt_, x_, tgt_mask_, masks_, q_embed_, pos_embed_, prm, prm_pos)
                        return self.transformer(tgt_, x_, tgt_mask_, masks_, q_embed_, pos_embed_)
                    elif fname in ('LssPlPrySeqLineTransformer', 'LssMLMPlPrySeqLineTransformer'):
                        tgt_cross = tgt_.unsqueeze(1)
                        return self.transformer(tgt_cross, x_, prm, masks_, q_embed_, pos_embed_, prm_pos)
                    elif fname == 'LssPlBzTransformer':
                        return self.transformer(tgt_, x_, tgt_mask_, masks_, q_embed_, pos_embed_)
                    else:
                        return self.transformer(tgt_, x_, masks_, q_embed_, pos_embed_)

                for it in range(iters):
                    tgt = self.embedding(seq.long())
                    if kp_bias is not None:
                        tgt = tgt + kp_bias.unsqueeze(1)
                    
                    # Debug NAR iteration
                    if it == 0:
                        print(f"\n[NAR DEBUG] Iteration {it}/{iters}")
                        print(f"  Initial seq shape: {seq.shape}")
                        print(f"  no_known count: {(seq == int(self.no_known)).sum().item()}/{seq.numel()}")

                    tgt_mask_zero = torch.zeros(T, T, device=tgt.device)
                    try:
                        if sd_prompt is not None and sd_prompt_pos is not None and \
                           transformer_name in ('LssPlPrySeqLineTransformer', 'LssMLMPlPrySeqLineTransformer', 'LssSARPrmSeqLineTransformer'):
                            outs_dec, _ = _call_transformer_with_sign(transformer_name, tgt, x, tgt_mask_zero, masks, query_embed, pos_embed, sd_prompt, sd_prompt_pos)
                        else:
                            outs_dec, _ = _call_transformer_with_sign(transformer_name, tgt, x, tgt_mask_zero, masks, query_embed, pos_embed, None, None)
                    except TypeError:
                        outs_dec, _ = _call_transformer_with_sign(transformer_name, tgt, x, tgt_mask_zero, masks, query_embed, pos_embed, None, None)

                    outs_dec = torch.nan_to_num(outs_dec)
                    logits = self.vocab_embed(outs_dec)
                    probs = logits.softmax(-1)
                    values_i, tokens = probs.max(dim=-1)
                    seq_pred = tokens[-1]
                    seq_pred[:, 0] = int(self.start)
                    last_values = values_i[-1]

                    # Last iteration: accept all
                    if it == iters - 1:
                        seq = seq_pred
                        print(f"  Final iteration: accepting all predictions")
                        print(f"  Final no_known count: {(seq == int(self.no_known)).sum().item()}/{seq.numel()}")
                        break

                    # Confidence-based re-masking for next iteration
                    conf = last_values  # [B, T]
                    keep_mask = (conf >= float(getattr(self, 'nar_conf_thresh', 0.5)))
                    print(f"  Iteration {it}: confidence-based keep: {keep_mask.sum().item()}/{keep_mask.numel()} tokens")
                    # Optional keep ratio: keep top-k confident tokens (excluding pos 0)
                    kr = float(getattr(self, 'nar_keep_ratio', 0.0))
                    if kr > 0.0 and T > 1:
                        k_top = max(1, int(round((T - 1) * kr)))
                        conf_tail = conf[:, 1:]
                        topk_vals, topk_idx = torch.topk(conf_tail, k=k_top, dim=-1)
                        keep_topk = torch.zeros_like(conf_tail, dtype=torch.bool)
                        gather = torch.arange(keep_topk.shape[0], device=keep_topk.device).unsqueeze(-1)
                        keep_topk[gather, topk_idx] = True
                        # merge with threshold-based mask on tail
                        keep_tail = keep_mask[:, 1:] | keep_topk
                        keep_mask = torch.cat([keep_mask[:, :1], keep_tail], dim=-1)
                    # Always keep START
                    keep_mask[:, 0] = True
                    # Apply: keep confident preds, remask others
                    remask = torch.full_like(seq_pred, int(self.no_known))
                    seq = torch.where(keep_mask, seq_pred, remask)

                return seq, last_values
            # Fallback: original AR decoding loop
            values = []
            seq = input_seqs
            # Precompute KP prompt bias
            kp_bias = None
            if self.kp_prompt_enable and self.kp_prompt_type == 'add':
                try:
                    kp_q = self.kp_query_embed.weight
                    kp_dec, _ = self.kp_transformer(x, masks, kp_q, pos_embed)
                    kp_feats = torch.nan_to_num(kp_dec)[-1]
                    kp_feats_cls = kp_feats.detach() if self.kp_prompt_detach else kp_feats
                    kp_cls_logits = self.kp_cls_head(kp_feats_cls)
                    with torch.no_grad():
                        kp_scores = kp_cls_logits.softmax(-1).max(dim=-1)[0]
                        k = max(1, min(self.kp_prompt_topk, kp_scores.shape[1]))
                        topk_scores, topk_idx = torch.topk(kp_scores, k=k, dim=-1)
                    B = kp_feats.shape[0]
                    gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, kp_feats.shape[-1])
                    kp_feats_prompt = kp_feats.detach() if self.kp_prompt_detach else kp_feats
                    kp_sel = torch.gather(kp_feats_prompt, 1, gather_idx)
                    if self.kp_prompt_weighted:
                        w = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-6)
                        kp_global = (kp_sel * w.unsqueeze(-1)).sum(dim=1)
                    else:
                        kp_global = kp_sel.mean(dim=1)
                    kp_bias = self.kp_prompt_adapter(kp_global)
                except Exception:
                    kp_bias = None
            for _ in range(self.max_iteration):
                tgt = self.embedding(seq.long())
                if kp_bias is not None:
                    tgt = tgt + kp_bias.unsqueeze(1)
                query_embed = self.embedding.position_embeddings.weight
                T = tgt.shape[1]
                use_meta_groups = (self.sar_group_strategy == 'meta_groups' and isinstance(img_metas, (list, tuple)))
                masks_bt = []
                if use_meta_groups:
                    for bi in range(B):
                        meta = img_metas[bi] if isinstance(img_metas, (list, tuple)) else {}
                        gids_full = None
                        if isinstance(meta, dict) and 'sar_group_ids_pos' in meta:
                            sar_pos = meta['sar_group_ids_pos']
                            if not torch.is_tensor(sar_pos):
                                sar_pos = torch.as_tensor(sar_pos, device=tgt.device, dtype=torch.long)
                            gids_full = _expand_group_ids_from_pos(T, clause_length, sar_pos)
                        elif isinstance(meta, dict) and 'sar_group_ids' in meta:
                            sar_full = meta['sar_group_ids']
                            if torch.is_tensor(sar_full) and sar_full.numel() == T:
                                gids_full = sar_full.to(tgt.device).long()
                            elif isinstance(sar_full, (list, tuple)) and len(sar_full) == T:
                                gids_full = torch.as_tensor(sar_full, device=tgt.device, dtype=torch.long)
                        if gids_full is None:
                            masks_bt = []
                            break
                        m = _build_group_mask_from_ids(gids_full, allow_mlm_intra=self.sar_intra_group_mlm)
                        masks_bt.append(m)
                if len(masks_bt) == B and len(masks_bt) > 0:
                    tgt_mask = torch.stack(masks_bt, dim=0)
                else:
                    base_mask = _build_block_causal_mask(T, block_len, tgt.device)
                    tgt_mask = base_mask.unsqueeze(0).repeat(B, 1, 1)
                transformer_name = self.transformer.__class__.__name__
                sd_prompt, sd_prompt_pos = (None, None)
                if transformer_name in ('LssPlPrySeqLineTransformer', 'LssMLMPlPrySeqLineTransformer', 'LssSARPrmSeqLineTransformer'):
                    sd_prompt, sd_prompt_pos = self._gather_sd_prompt(img_metas, tgt.device, self.embed_dims)

                # 统一签名分发（含 LssPlBzTransformer），tgt 始终为 3D [B,T,D]
                def _call_transformer_with_sign(fname: str, tgt_, x_, tgt_mask_, masks_, q_embed_, pos_embed_, prm=None, prm_pos=None):
                    if fname == 'LssSARPrmSeqLineTransformer':
                        if prm is not None and prm_pos is not None:
                            return self.transformer(tgt_, x_, tgt_mask_, masks_, q_embed_, pos_embed_, prm, prm_pos)
                        return self.transformer(tgt_, x_, tgt_mask_, masks_, q_embed_, pos_embed_)
                    elif fname in ('LssPlPrySeqLineTransformer', 'LssMLMPlPrySeqLineTransformer'):
                        tgt_cross = tgt_.unsqueeze(1)
                        return self.transformer(tgt_cross, x_, prm, masks_, q_embed_, pos_embed_, prm_pos)
                    elif fname == 'LssPlBzTransformer':
                        return self.transformer(tgt_, x_, tgt_mask_, masks_, q_embed_, pos_embed_)
                    else:
                        return self.transformer(tgt_, x_, masks_, q_embed_, pos_embed_)

                try:
                    if sd_prompt is not None and sd_prompt_pos is not None:
                        outs_dec, _ = _call_transformer_with_sign(transformer_name, tgt, x, tgt_mask, masks, query_embed, pos_embed, sd_prompt, sd_prompt_pos)
                    else:
                        use_kp_cross = (self.kp_prompt_enable and self.kp_prompt_type == 'cross' and transformer_name in ('LssPlPrySeqLineTransformer', 'LssMLMPlPrySeqLineTransformer', 'LssSARPrmSeqLineTransformer'))
                        if use_kp_cross:
                            kp_q = self.kp_query_embed.weight
                            kp_dec, _ = self.kp_transformer(x, masks, kp_q, pos_embed)
                            kp_feats = torch.nan_to_num(kp_dec)[-1]
                            kp_cls_logits = self.kp_cls_head(kp_feats)
                            kp_coords_norm = torch.sigmoid(self.kp_reg_head(kp_feats))
                            with torch.no_grad():
                                kp_scores = kp_cls_logits.softmax(-1).max(dim=-1)[0]
                                k = max(1, min(self.kp_prompt_topk, kp_scores.shape[1]))
                                topk_scores, topk_idx = torch.topk(kp_scores, k=k, dim=-1)
                            gather_idx_d = topk_idx.unsqueeze(-1).expand(-1, -1, kp_feats.shape[-1])
                            gather_idx_p = topk_idx.unsqueeze(-1).expand(-1, -1, kp_coords_norm.shape[-1])
                            prompt = torch.gather(kp_feats, 1, gather_idx_d)
                            prompt_pos = torch.gather(kp_coords_norm, 1, gather_idx_p)
                            prompt_pos = self.kp_pos_mlp(prompt_pos)
                            outs_dec, _ = _call_transformer_with_sign(transformer_name, tgt, x, tgt_mask, masks, query_embed, pos_embed, prompt, prompt_pos)
                        else:
                            outs_dec, _ = _call_transformer_with_sign(transformer_name, tgt, x, tgt_mask, masks, query_embed, pos_embed, None, None)
                except TypeError:
                    outs_dec, _ = _call_transformer_with_sign(transformer_name, tgt, x, tgt_mask, masks, query_embed, pos_embed, None, None)
                step_feats = torch.nan_to_num(outs_dec)[-1, :, -1, :]
                step_logits = self.vocab_embed(step_feats)
                step_probs = step_logits.softmax(-1)
                value, next_token = step_probs.topk(dim=-1, k=1)
                seq = torch.cat([seq, next_token], dim=-1)
                values.append(value)
            values = torch.cat(values, dim=-1)
            return seq, values

    # ---------------- Keypoint branch API ----------------
    def forward_keypoints(self, mlvl_feats, img_metas):
        """Forward keypoint parallel branch.

        Returns:
            kp_cls_logits: [B, Q, C]
            kp_coords_norm: [B, Q, 2] in [0, 1]
        """
        x = mlvl_feats
        if self.in_channels != self.embed_dims:
            x = self.bev_proj(x)
        pos_embed = self.bev_position_encoding(x)
        B, _, H, W = x.shape
        masks = torch.zeros(B, H, W).bool().to(x.device)
        query_embed = self.kp_query_embed.weight  # [Q, D]
        # Use KeypointTransformer interface
        outs_dec, _ = self.kp_transformer(x, masks, query_embed, pos_embed)
        feats = torch.nan_to_num(outs_dec)[-1]  # [B, Q, D]
        kp_cls_logits = self.kp_cls_head(feats)
        kp_coords_norm = torch.sigmoid(self.kp_reg_head(feats))
        return kp_cls_logits, kp_coords_norm
