#ifndef ATTENTION_H
#define ATTENTION_H

#include "tensor.h"
#include "autodiff.h"

typedef struct {
    int embed_dim, n_heads;
    Tensor *w_q, *w_k, *w_v, *w_o; // Weight matrices
} MultiHeadAttention;

/**
 * create_multihead_attention - allocate a multi-head attention block
 * @embed_dim: embedding dimension
 * @n_heads: number of heads
 * returns: pointer to attention block or NULL on failure
 */
MultiHeadAttention* create_multihead_attention(int embed_dim, int n_heads);

/**
 * free_multihead_attention - free a multi-head attention block
 * @mha: pointer to attention block
 */
void free_multihead_attention(MultiHeadAttention* mha);

/**
 * multihead_attention_forward - run forward pass of multi-head attention
 * @out: output tensor
 * @q_in: query input
 * @k_in: key input
 * @v_in: value input
 * @mha: attention block
 * @mask: optional mask tensor
 */
void multihead_attention_forward(Tensor* out, Tensor* q_in, Tensor* k_in, Tensor* v_in, MultiHeadAttention* mha, Tensor* mask);

/**
 * multihead_attention_forward_ad - autodiff forward pass for multi-head attention
 * @arena: memory arena
 * @q_in: query input
 * @k_in: key input
 * @v_in: value input
 * @mha: attention block
 * @mask: optional mask tensor
 * returns: autodiff value
 */
Value* multihead_attention_forward_ad(Arena* arena, Value* q_in, Value* k_in, Value* v_in, MultiHeadAttention* mha, Tensor* mask);

/**
 * save_multihead_attention - write attention block to file
 * @mha: attention block
 * @fp: file pointer
 * returns: 1 on success, 0 on failure
 */
int save_multihead_attention(MultiHeadAttention* mha, FILE* fp);

/**
 * load_multihead_attention - read attention block from file
 * @mha: attention block
 * @fp: file pointer
 * returns: 1 on success, 0 on failure
 */
int load_multihead_attention(MultiHeadAttention* mha, FILE* fp);

/**
 * split_heads - split tensor into multiple heads
 * @arena: memory arena
 * @x: input tensor
 * @n_heads: number of heads
 * returns: new tensor or NULL on failure
 */
Tensor* split_heads(Arena* arena, Tensor* x, int n_heads);

/**
 * combine_heads - combine multi-head tensor into single tensor
 * @arena: memory arena
 * @x: input tensor
 * returns: new tensor or NULL on failure
 */
Tensor* combine_heads(Arena* arena, Tensor* x);

#endif // ATTENTION_H
