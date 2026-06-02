<!-- Encoder Memory (memory)  
  shape: [B, Σ(Hi*Wi), C]  
  └── multi-scale features (flattened + projected)
        │
        ▼
 [UMQS] Top-K Query Selection
   - enc_outputs_class = enc_score_head(enc_output(memory))  
        [B, N, num_classes]
   - enc_outputs_coord_unact = enc_bbox_head(enc_output(memory)) + anchors  
        [B, N, 4]
   - topk_ind = torch.topk(enc_outputs_class.max(-1).values, num_queries)  
        [B, num_queries]

   Selected:
     - reference_points_unact = gather(enc_outputs_coord_unact, topk_ind)  
          [B, num_queries, 4]  (raw boxes, unactivated)
     - enc_topk_bboxes = sigmoid(reference_points_unact)  
          [B, num_queries, 4]  (normalized boxes)
     - enc_topk_logits = gather(enc_outputs_class, topk_ind)  
          [B, num_queries, num_classes]

        │
        ▼
 Denoising Queries (optional, denoising.py)
   - input_query_logits = class_embed(noisy_labels)  
        [B, num_denoising, C]
   - input_query_bbox_unact = inverse_sigmoid(noisy_boxes)  
        [B, num_denoising, 4]
   - attn_mask = restricts denoising queries visibility  
        [num_denoising+num_queries, num_denoising+num_queries]

        │
        ▼
 Initial Queries
   - target =  
       if self.learnt_init_query → self.tgt_embed.weight  
          [B, num_queries, C]
       else → gather(output_memory, topk_ind)  
          [B, num_queries, C]
   - Concatenate denoising_class if provided  
          [B, num_denoising+num_queries, C]
   - reference_points_unact = concat(denoising_bbox_unact, reference_points_unact)  
          [B, num_denoising+num_queries, 4]

        │
        ▼
 ┌─────────────────────────────────────────────────┐
 │   TransformerDecoder (stack of L layers)        │
 │ Inputs:                                         │
 │   - target (queries) [B, Q, C]                  │
 │   - reference_points_unact [B, Q, 4]            │
 │   - memory (encoder features) [B, N, C]         │
 │   - spatial_shapes [n_levels, 2]                │
 │   - level_start_index [n_levels]                │
 │   - attn_mask [Q, Q] (optional, from dn.py)     │
 │                                                 │
 │ ┌─────────────────────────────────────────────┐ │
 │ │ Decoder Layer i:                            │ │
 │ │                                             │ │
 │ │ 1. Self-Attention                           │ │
 │ │    queries ↔ queries                        │ │
 │ │    output: [B, Q, C]                        │ │
 │ │                                             │ │
 │ │ 2. Cross-Attention (MS Deformable Attention)│ │
 │ │    - deformable_attention_core(             │ │
 │ │        value=memory,                        │ │
 │ │        value_spatial_shapes=spatial_shapes, │ │
 │ │        sampling_locations,                  │ │
 │ │        attention_weights                    │ │
 │ │      )                                      │ │
 │ │    output: [B, Q, C]                        │ │
 │ │                                             │ │
 │ │ 3. Feed-Forward Network                     │ │
 │ │    Linear → Activation → Linear             │ │
 │ │    output: [B, Q, C]                        │ │
 │ │                                             │ │
 │ │ Residual + LayerNorm after each block       │ │
 │ │                                             │ │
 │ │ Detection Heads (per-layer):                │ │
 │ │   - dec_score_head[i](queries) → logits     │ │
 │ │       [B, Q, num_classes]                   │ │
 │ │   - dec_bbox_head[i](queries) +             │ │
 │ │       inverse_sigmoid(ref_points_detach)    │ │
 │ │       → inter_ref_bbox [B, Q, 4]            │ │
 │ │                                             │ │
 │ │ Iterative Refinement:                       │ │
 │ │   ref_points ← sigmoid(inter_ref_bbox)      │ │
 │ └─────────────────────────────────────────────┘ │
 │                                                 │
 └─────────────────────────────────────────────────┘
        │
        ▼
 Final Decoder Outputs:
   - dec_out_logits [L, B, Q, num_classes]  
   - dec_out_bboxes [L, B, Q, 4]  
   - intermediate predictions (deep supervision) -->
