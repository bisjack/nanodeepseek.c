#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <time.h>

#define M_PI 3.14159265358979323846
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

/*

    DATA STRUCTURE

*/


typedef struct
{
	//Block
	float* atn_n;
	float* ffn_n;
	
	//MLA
	float* wq_a;
	float* wq_b;
	float* wq_n;
	
	float* wkv_a;
	float* wkv_b;
	float* wkv_n;
	
	float* wo;
	
	float* wkv_b0;
	float* wkv_b1;
	
	//MoE
	float* w_gate;
	
	float* routed_experts_w1;
	float* routed_experts_w2;
	float* routed_experts_w3;
	
	float* shared_experts_w1;
	float* shared_experts_w2;
	float* shared_experts_w3;
	
	float* score_bias;
	
	//MLP
	float* w1;
	float* w2;
	float* w3;
	
	////Transformer
	//float* tfm_n;
	
} W;

typedef struct
{
	//block
	float* u_0  ;
	float* u_n0 ;
	float* u_atn;
	float* u_1  ;
	float* u_n1 ;
	float* u_ffn;
	float* u_2  ;
	
	
	//MLA
	float* q_a ;
	float* q_n ;
	float* q   ;
	float* q_no;
	float* q_pe;
	
	float* qb_no;
	
	float* kv0 ;
	float* kv1 ;
	float* kv  ;
	float* k_pe;
	
	float* scores  ;
	float* smscores;
	float* scoresb ;
	float* ctx     ;
	
	float* kv_history;
	float* pe_history;
	
	float* kv_cache;
	float* pe_cache;
	
	
	//MoE
	float* y;
	float* z;
	
	float* v_gate;
	int*   topk  ;
	
	float* score;
	
	float *orginal_score;
	float *masked_score;
	float *group_score;
	
	int*  counts  ;
	int** act_exps;
	
} A;

/*

    Model

*/

void precompute_freqs_cis(float* fcos, float* fsin, int MAXT, int ORIGINAL_SEQLEN, int HEAD_DIM)
{	
	const float ROPE_THETA  = 10000.0;
    const float ROPE_FACTOR = 40.0;
    const int   BETA_FAST = 32;
    const int   BETA_SLOW = 1;
	
	int HALFHEAD_DIM = HEAD_DIM / 2;
	
	float* freq = (float *)malloc(HALFHEAD_DIM * sizeof(float));
	
	for(int i = 0; i < HALFHEAD_DIM; i++)
	{
		freq[i] = 1.0f / powf(ROPE_THETA, (float)i / (float)HALFHEAD_DIM);
	}
	
	if(MAXT > ORIGINAL_SEQLEN)
	{
	    int low, high;
		
		low  = MAX(floor(HEAD_DIM * logf(ORIGINAL_SEQLEN / (BETA_FAST * 2 * M_PI)) / (2 * logf(ROPE_THETA))),          0);
		high = MIN( ceil(HEAD_DIM * logf(ORIGINAL_SEQLEN / (BETA_SLOW * 2 * M_PI)) / (2 * logf(ROPE_THETA))), HEAD_DIM-1);
		
		if(low == high)
			high += 0.001;
		
		for(int i = 0; i < HALFHEAD_DIM; i++)
		{
			float smooth = (float)(i - low) / (float)(high - low);
			
			smooth = 1 - MIN(MAX(smooth, 0.0f), 1.0f);
			
			freq[i] = freq[i] / ROPE_FACTOR * (1 - smooth) + freq[i] * smooth;
		}
	}
	
	for(int i = 0; i < HALFHEAD_DIM; i++)
	{
		for(int t = 0; t < MAXT; t++)
		{
			float pos = t * freq[i];

			fcos[t * HALFHEAD_DIM + i] = cosf(pos);
			fsin[t * HALFHEAD_DIM + i] = sinf(pos);	
		}
	}
	
	free(freq);
}

void apply_rotary_emb
(
    float* x,
	
	float* fcos,
	float* fsin,
	
	int start_pos,
	
	int B,
	int T,
	int N_HEADS,
	int ROPE_HEAD_DIM
)
{
	int HALFHEAD_DIM = ROPE_HEAD_DIM / 2;
	
	for(int b = 0; b < B; b++)
	{
		for(int t = 0; t < T; t++)
		{
			for(int h = 0; h < N_HEADS; h++)
			{
				float* x_bth  = x + b * T * N_HEADS * ROPE_HEAD_DIM + t * N_HEADS * ROPE_HEAD_DIM + h * ROPE_HEAD_DIM;
				float* fcos_t = fcos + (t + start_pos) * HALFHEAD_DIM;
				float* fsin_t = fsin + (t + start_pos) * HALFHEAD_DIM;
				
				for(int d = 0; d < ROPE_HEAD_DIM; d += 2)
				{
					float x0 = x_bth[d    ];
					float x1 = x_bth[d + 1];
					
					x_bth[d    ] = fcos_t[d / 2] * x0 - fsin_t[d / 2] * x1;
					x_bth[d + 1] = fsin_t[d / 2] * x0 + fcos_t[d / 2] * x1;			
				}
			}
		}
	}
}

void split(float* x0, float* x1, float* y, int M, int N0, int N1)
{
	int N = N0 + N1;
	
	for(int m = 0; m < M; m++)
	{
		float* y_m  = y  + m * N ;
		float* x0_m = x0 + m * N0;
		float* x1_m = x1 + m * N1;
		
		for(int n = 0; n < N; n++)
		{
			if(n < N0)
				x0_m[n] = y_m[n];
			else
				x1_m[n-N0] = y_m[n];
		}
	}
}

void split1(float* x0, float* x1, float* y, int M, int K, int N0, int N1)
{
	int N = N0 + N1;
	
	for(int m = 0; m < M; m++)
	{
		float* y_m  = y  + m * N  * K;
		float* x0_m = x0 + m * N0 * K;
		float* x1_m = x1 + m * N1 * K;
		
		for(int n = 0; n < N; n++)
		{
			if(n < N0)
			{
				for(int k = 0; k < K; k++)
					x0_m[n * K + k] = y_m[n * K + k];
			}
			else
			{
				for(int k = 0; k < K; k++)
					x1_m[(n-N0) * K + k] = y_m[n * K + k];
			}	
		}
	}
	
}

void rmsnorm(float* y, float* x, float* w, int B, int T, int C)
{
	float eps = 1e-6f;
	
	for(int b = 0; b < B; b++)
	{
		for(int t = 0; t < T; t++)
		{
			float* x_bt = x + b * T * C + t * C;
			float* y_bt = y + b * T * C + t * C;
			
			float rms = 0.0f;
			for(int i = 0; i < C; i++)
	        {
	        	rms += x_bt[i] * x_bt[i];
	        }
			rms /= C;
			rms += eps;
			rms = 1.0 / sqrtf(rms + eps);
			
			for(int i = 0; i < C; i++)
	        {
	        	y_bt[i] = rms * x_bt[i] * w[i];
	        }
		}
	}
}

void softmax(float* y, float* x, int M, int N)
{
	#pragma omp parallel for collapse(2)
	for(int m = 0; m < M; m++)
	{
		float* y_m = y + m * N;
		float* x_m = x + m * N;
		
		float max_val = -10000.0f;
		float sum     =      0.0f;
		float rec_sum =      0.0f;
		
		for(int n = 0; n < N; n++)
		{
			if(x_m[n] > max_val)
				max_val = x_m[n];
		}
		
		for(int n = 0; n < N; n++)
		{
			y_m[n] = expf(x_m[n] - max_val);
		    sum += y_m[n];
		}
		
		rec_sum = sum == 0.0f ? 0.0f : 1.0f / sum;
		
		for(int n = 0; n < N; n++)
			y_m[n] = y_m[n] * rec_sum;
		
	}
}

void elemwise_add(float* y, float* x0, float* x1, int N)
{
	for(int i = 0; i < N; i++)
		y[i] = x0[i] + x1[i];
}

void linear(float* y, float* x, float* w, int B, int T, int C, int OC)
{
	#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++)
	{
        for (int t = 0; t < T; t++)
		{
            int bt = b * T + t;
            for (int o = 0; o < OC; o++)
			{
                float val = 0.0f;
                for (int i = 0; i < C; i++)
				{
                    val += x[bt * C + i] * w[o * C + i];
                }
                y[bt * OC + o] = val;
            }
        }
    }
}

void einsum_bshd_hdc_bshc(float* y, float* x0, float* x1, int B, int S, int H, int D, int C)
{
	for(int b = 0; b < B; b++)
	{
		for(int s = 0; s < S; s++)
		{
			for(int h = 0; h < H; h++)
			{
				float* x0_bsh = x0 + b * S * H * D + s * H * D + h * D;
				float* y_bsh  =  y + b * S * H * C + s * H * C + h * C;
				for(int c = 0; c < C; c++)
				{
					float* x1_hc = x1 + h * D * C + c;
					
					float tmp = 0.0f;
					for(int d = 0; d < D; d++)
						tmp += x0_bsh[d] * x1_hc[d * C];
					
					y_bsh[c] = tmp;
				}
			}
		}	
	}
}

void einsum_bshc_btc_bsht_acc(float* y, float* x0, float* x1, int B, int S, int H, int T, int C)
{
	for(int b = 0; b < B; b++)
	{
		for(int s = 0; s < S; s++)
		{
			for(int h = 0; h < H; h++)
			{
				float* y_bsh  = y  + b * S * H * T + s * H * T + h * T;
				float* x0_bsh = x0 + b * S * H * C + s * H * C + h * C;
				for(int t = 0; t < T; t++)
				{	
					float* x1_bt  = x1 + b * T * C + t * C;
					
					float tmp = 0.0f;
					for(int c = 0; c < C; c++)
						tmp += x0_bsh[c] * x1_bt[c];
					
					y_bsh[t] += tmp;
				}
			}
		}	
	}
}

void einsum_bsht_btc_bshc(float* y, float* x0, float* x1, int B, int S, int H, int T, int C)
{
	for(int b = 0; b < B; b++)
	{
		for(int s = 0; s < S; s++)
		{
			for(int h = 0; h < H; h++)
			{
				float* x0_bsh = x0 + b * S * H * T + s * H * T + h * T;
				float* y_bsh  = y  + b * S * H * C + s * H * C + h * C;
				for(int c = 0; c < C; c++)	
				{
					float* x1_bt = x1 + b * T * C + c;
					
					float tmp = 0.0f;
					for(int t = 0; t < T; t++)
						tmp += x0_bsh[t] * x1_bt[t * C];
					
					y_bsh[c] = tmp;
				}
			}
		}	
	}
}

void einsum_bshc_hdc_bshd(float* y, float* x0, float* x1, int B, int S, int H, int D, int C)
{
	for(int b = 0; b < B; b++)
	{
		for(int s = 0; s < S; s++)
		{
			for(int h = 0; h < H; h++)
			{
				float* x0_bsh = x0 + b * S * H * C + s * H * C + h * C;
				float* y_bsh  = y  + b * S * H * D + s * H * D + h * D;
				for(int d = 0; d < D; d++)	
				{
					float* x1_hd = x1 + h * D * C + d * C;
					
					float tmp = 0.0f;
					for(int c = 0; c < C; c++)
						tmp += x0_bsh[c] * x1_hd[c];
					
					y_bsh[d] = tmp;
				}
			}
		}	
	}
}

void cache_load(float* cache, float* data, int B, int MAXT, int T, int start_pos, int D)
{
	for(int b = 0; b < B; b++)
	{
		for(int t = 0; t < T; t++)
		{
			float* cache_bt = cache + b * MAXT * D + (t + start_pos) * D;
			float* data_bt  = data  + b * T * D + t * D;
			
			for(int d = 0; d < D; d++)
				cache_bt[d] = data_bt[d];
		}
	}
}

void clip(float* data, float* cache, int B, int MAXT, int T, int start_pos, int D)
{
	int end_pos = T + start_pos;
	
	for(int b = 0; b < B; b++)
	{
		for(int t = 0; t < end_pos; t++)
		{
			float* cache_bt = cache + b * MAXT    * D + t * D;
			float* data_bt  = data  + b * end_pos * D + t * D;
			
			for(int d = 0; d < D; d++)
				data_bt[d] = cache_bt[d];
		}
	}
}

void casualmask(float* scores, int B, int T, int N)
{
    for(int b = 0; b < B; b++)
	{
		for(int t0 = 0; t0 < T; t0++)
		{
			for(int n = 0; n < N; n++)
			{
				float* scores_btn = scores + b * T * N * T + t0 * N * T + n * T;
				for(int t1 = t0+1; t1 < T; t1++)
				{
					scores_btn[t1] = -FLT_MAX;
				}
			}
		}
	}
}

void gate
(
	float *v_gate,
	int   *topk  ,
    
    float *score,	
	
	float *score_bias,  
	
	float *orginal_score,
	float *masked_score,
	float *group_score,
	
	const float ROUTE_SCALE,
	
	const int B,
	const int T,
	const int N_ROUTED_EXPERTS,
	const int N_ACTIVATED_EXPERTS,
	const int N_EXPERT_GROUPS,
	const int N_LIMITED_GROUPS
)
{ 
	const int N_EXPERT_PERGROUP = N_ROUTED_EXPERTS / N_EXPERT_GROUPS;
	
	
	int*   topkgroupi = (int   *)malloc(B * T * N_LIMITED_GROUPS * sizeof(int  ));
	float* topkgroupv = (float *)malloc(B * T * N_LIMITED_GROUPS * sizeof(float));
	
	for(int i = 0; i < B * T; i++)
	{
		float*         score_i =         score + i * N_ROUTED_EXPERTS;
		float* orginal_score_i = orginal_score + i * N_ROUTED_EXPERTS;
		for(int j = 0; j < N_ROUTED_EXPERTS; j++)
		{
			score_i[j] = 1.0f / (1.0f + expf(-score_i[j]));
			orginal_score_i[j] = score_i[j];
			score_i[j] += score_bias[j];
		}
	}
	
	for(int i = 0; i < B * T * N_EXPERT_GROUPS; i++)
	{	
        float* score_i = score + i * N_EXPERT_PERGROUP;

		float max0 = -FLT_MAX;
		float max1 = -FLT_MAX;
	    
 		for(int j = 0; j < N_EXPERT_PERGROUP; j++)
		{
		    if(score_i[j] > max0)
			{
				if(score_i[j] > max1)
				{
					max0 = max1;
					max1 = score_i[j];
				}
				else
				{
					max0 = score_i[j];
				}
			}	
		}
		
		group_score[i] = max0 + max1;
	}
	
	for(int b = 0; b < B * T; b++)
	{
		int*   topkgroupi_b = topkgroupi + b * N_LIMITED_GROUPS;
        float* topkgroupv_b = topkgroupv + b * N_LIMITED_GROUPS;
		
		float* group_score_b = group_score + b * N_EXPERT_GROUPS;
		
		for(int i = 0; i < N_LIMITED_GROUPS; i++)
		{
			topkgroupi_b[i] = -1;
			topkgroupv_b[i] = -FLT_MAX;
		}
		
		for(int i = 0; i < N_EXPERT_GROUPS; i++)
		{
			int pos = -1;
			
			for(int k = 0; k < N_LIMITED_GROUPS; k++)
			{
				if(group_score_b[i] >= topkgroupv_b[k])
				{
					pos = k;
					continue;
				}
				else
					break;
			}
			
			if(pos == -1)
				continue;
			else
			{
				for(int k = 0; k < pos; k++)
				{
					topkgroupv_b[k] = topkgroupv_b[k + 1];
					topkgroupi_b[k] = topkgroupi_b[k + 1];
				}
				
				topkgroupv_b[pos] = group_score_b[i];
				topkgroupi_b[pos] = i;
			}
		}
	}
	
	for(int i = 0; i < B * T * N_ROUTED_EXPERTS; i++)
		masked_score[i] = -FLT_MAX;
	
	for(int b = 0; b < B * T; b++)
	{
		int* topkgroupi_b = topkgroupi + b * N_LIMITED_GROUPS;
		
		for(int i = 0; i < N_LIMITED_GROUPS; i++)
		{
			int topkgroup = topkgroupi_b[i];
			
			float* masked_score_bg = masked_score + b * N_EXPERT_GROUPS * N_EXPERT_PERGROUP + topkgroup * N_EXPERT_PERGROUP;
			float*        score_bg =        score + b * N_EXPERT_GROUPS * N_EXPERT_PERGROUP + topkgroup * N_EXPERT_PERGROUP;
			
			for(int j = 0; j < N_EXPERT_PERGROUP; j++)
				masked_score_bg[j] = score_bg[j];
		}
	}
	
	for (int b = 0; b < B * T; b++)
	{
		int offset0 = b * N_ROUTED_EXPERTS   ;
		int offset1 = b * N_ACTIVATED_EXPERTS;
		
		float sum = 0.0f;
		float coe;
		
		float*         score_b  =  masked_score + offset0;
		float* orginal_score_b  = orginal_score + offset0;
		
		float* v_gate_b = v_gate + offset1;
		int  * topk_b   = topk   + offset1;
		
		for(int k = 0; k < N_ACTIVATED_EXPERTS; k++)
		{
			v_gate_b[k] = -FLT_MAX;
			topk_b[k] = -1;
		}
		
		for(int i = 0; i < N_ROUTED_EXPERTS; i++)
		{
			int pos = -1;
			
			for(int k = 0; k < N_ACTIVATED_EXPERTS; k++)
			{
				if(score_b[i] >= v_gate_b[k])
				{
					pos = k;
					continue;
				}
				else
					break;
			}
			
			if(pos == -1)
				continue;
			else
			{
				for(int k = 0; k < pos; k++)
				{
					v_gate_b[k] = v_gate_b[k + 1];
					topk_b[k] = topk_b[k + 1];
				}
				
				v_gate_b[pos] = score_b[i];
				topk_b[pos] = i;
			}
		}
		
		for(int k = 0; k < N_ACTIVATED_EXPERTS; k++)
		{
			v_gate_b[k] = orginal_score_b[topk_b[k]];
			sum += v_gate_b[k];
		}
		
		coe = (sum == 0.0f) ? 0.0f : ROUTE_SCALE * 1.0f / sum;
		
		for(int k = 0; k < N_ACTIVATED_EXPERTS; k++)
		{
			v_gate_b[k] *= coe;
		}
    }	
}

void matmul_acc(float* y, float* x, float* w, int B, int T, int C, int OC)
{
	#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++)
	{
        for (int t = 0; t < T; t++)
		{
            int bt = b * T + t;
            for (int o = 0; o < OC; o++)
			{
                float val = 0.0f;
                for (int i = 0; i < C; i++)
				{
                    val += x[bt * C + i] * w[o*C + i];
                }
                y[bt * OC + o] += val;
            }
        }
    }
}

void silu(float* y, float* x, int N)
{
    for (int i = 0; i < N; i++)
    {
        float sigmoid = 1.0 / (1.0 + expf(-x[i]));
        y[i] = x[i] * sigmoid;
    }
}

void elemwise_dot(float* y, float* x0, float* x1, int N)
{
	for(int i = 0; i < N; i++)
		y[i] = x0[i] * x1[i];
}

void mlp
(
    float* z_idx,
	
    float* u_idx,

    float* w1,
    float* w2,
    float* w3,
	
	int B,
	int T,
    int DIM,
    int INTER_DIM
)
{
	float* u1 = (float *)malloc(B * T * INTER_DIM * sizeof(float));
	float* s1 = (float *)malloc(B * T * INTER_DIM * sizeof(float));
	float* u3 = (float *)malloc(B * T * INTER_DIM * sizeof(float));
	float* u2 = (float *)malloc(B * T * INTER_DIM * sizeof(float));
	
	linear(u1, u_idx, w1, B, T, DIM, INTER_DIM);
	linear(u3, u_idx, w3, B, T, DIM, INTER_DIM);
	
	silu(s1, u1, B * T * INTER_DIM);
	elemwise_dot(u2, s1, u3, B * T * INTER_DIM);
	
	matmul_acc(z_idx, u2, w2, B, T, INTER_DIM, DIM);
	
	free(u1);
	free(s1);
	free(u3);
	free(u2);
}

void weighted_expert
(
    float* y_idx,
	
    float* u_idx,

    float* routed_expert_w1_i,
    float* routed_expert_w2_i,
    float* routed_expert_w3_i,

    float weight,
	
    int DIM,
    int MOE_INTER_DIM
)
{
	float* u1 = (float *)malloc(MOE_INTER_DIM * sizeof(float));
	float* s1 = (float *)malloc(MOE_INTER_DIM * sizeof(float));
	float* u3 = (float *)malloc(MOE_INTER_DIM * sizeof(float));
	float* u2 = (float *)malloc(MOE_INTER_DIM * sizeof(float));
	
	linear(u1, u_idx, routed_expert_w1_i, 1, 1, DIM, MOE_INTER_DIM);
	linear(u3, u_idx, routed_expert_w3_i, 1, 1, DIM, MOE_INTER_DIM);
	
	silu(s1, u1, MOE_INTER_DIM);
	elemwise_dot(u2, s1, u3, MOE_INTER_DIM);
	
	for(int i = 0; i < MOE_INTER_DIM; i++)
		u2[i] = u2[i] * weight;
	
	matmul_acc(y_idx, u2, routed_expert_w2_i, 1, 1, MOE_INTER_DIM, DIM);
	
	free(u1);
	free(s1);
	free(u3);
	free(u2);
}

void attn
(
    float* h         ,
	float* u         ,
	float* q_a       ,
    float* q_n       ,
    float* q         ,
    float* q_no      ,
    float* q_pe      ,
    float* qb_no     ,
    float* kv0       ,
    float* kv1       ,
    float* kv        ,
    float* k_pe      ,
    float* scores    ,
    float* smscores  ,
    float* scoresb   ,
    float* ctx       ,
    float* kv_history,
    float* pe_history,
	float* kv_cache  ,
	float* pe_cache  ,
	
	float* wq_a      ,
	float* wq_b      ,
	float* wq_n      ,
	float* wkv_a     ,
	float* wkv_b     ,
	float* wkv_n     ,
	float* wo        ,
	float* wkv_b0    ,
	float* wkv_b1    ,

	float* fcos      ,
	float* fsin      ,
	
	int start_pos    ,
	
	const float softmax_scale   ,
	
	const int   BATCH           , 
	const int   SEQLEN          ,
	const int   DIM             ,
	const int   MAX_BATCH       ,
	const int   MAX_SEQLEN      , 
	const int   Q_LORA_RANK     ,     
	const int   KV_LORA_RANK    ,    
	const int   N_HEADS         ,         
	const int   QK_NOPE_HEAD_DIM,
	const int   QK_ROPE_HEAD_DIM,
	const int   V_HEAD_DIM      ,      
	const int   QK_HEAD_DIM     
)
{
	//q
	linear(q_a, u, wq_a, BATCH, SEQLEN, DIM, Q_LORA_RANK);
	
	rmsnorm(q_n, q_a, wq_n, BATCH, SEQLEN, Q_LORA_RANK);
	
	linear(q, q_n, wq_b, BATCH, SEQLEN, Q_LORA_RANK, N_HEADS * QK_HEAD_DIM);
	
	split(q_no, q_pe, q, BATCH * SEQLEN * N_HEADS, QK_NOPE_HEAD_DIM, QK_ROPE_HEAD_DIM);
	
	apply_rotary_emb(q_pe, fcos, fsin, start_pos, BATCH, SEQLEN, N_HEADS, QK_ROPE_HEAD_DIM);
	
	//k_pe, kv
	linear(kv0, u, wkv_a, BATCH, SEQLEN, DIM, (KV_LORA_RANK + QK_ROPE_HEAD_DIM));
	
	split(kv1, k_pe, kv0, BATCH * SEQLEN, KV_LORA_RANK, QK_ROPE_HEAD_DIM);
	
	apply_rotary_emb(k_pe, fcos, fsin, start_pos, BATCH, SEQLEN, 1, QK_ROPE_HEAD_DIM);
    
	rmsnorm(kv, kv1, wkv_n, BATCH, SEQLEN, KV_LORA_RANK);
	
	//score
	split1(wkv_b0, wkv_b1, wkv_b, N_HEADS, KV_LORA_RANK, QK_NOPE_HEAD_DIM, V_HEAD_DIM);
	
	einsum_bshd_hdc_bshc(qb_no, q_no, wkv_b0, BATCH, SEQLEN, N_HEADS, QK_NOPE_HEAD_DIM, KV_LORA_RANK);
	
	cache_load(kv_cache, kv  , BATCH, MAX_SEQLEN, SEQLEN, start_pos, KV_LORA_RANK    );
	cache_load(pe_cache, k_pe, BATCH, MAX_SEQLEN, SEQLEN, start_pos, QK_ROPE_HEAD_DIM);
	
	clip(kv_history, kv_cache, BATCH, MAX_SEQLEN, SEQLEN, start_pos, KV_LORA_RANK    );
	clip(pe_history, pe_cache, BATCH, MAX_SEQLEN, SEQLEN, start_pos, QK_ROPE_HEAD_DIM);
	
	einsum_bshc_btc_bsht_acc(scores, qb_no, kv_history, BATCH, SEQLEN, N_HEADS, SEQLEN + start_pos, KV_LORA_RANK);
    einsum_bshc_btc_bsht_acc(scores, q_pe , pe_history, BATCH, SEQLEN, N_HEADS, SEQLEN + start_pos, QK_ROPE_HEAD_DIM);

	for(int i = 0; i < (BATCH * SEQLEN * N_HEADS * (SEQLEN + start_pos)); i++)
		scores[i] *= softmax_scale;
	
	if(SEQLEN > 1)
		casualmask(scores, BATCH, SEQLEN, N_HEADS);
	
	softmax(smscores, scores, BATCH * SEQLEN * N_HEADS, SEQLEN + start_pos);
	
	einsum_bsht_btc_bshc(scoresb, smscores, kv_history, BATCH, SEQLEN, N_HEADS, SEQLEN + start_pos, KV_LORA_RANK);	
	einsum_bshc_hdc_bshd(ctx, scoresb, wkv_b1, BATCH, SEQLEN, N_HEADS, V_HEAD_DIM, KV_LORA_RANK);

    linear(h, ctx, wo, BATCH, SEQLEN, N_HEADS * V_HEAD_DIM, DIM);
}

void moe
(
    float* h                 ,
	float* u                 ,
	float* y                 ,
	float* z                 ,
	float* v_gate            ,
	int*   topk              ,
	float* score             ,
	float* orginal_score     ,
	float* masked_score      ,
	float* group_score       ,
	int*   counts            ,
	int**  act_exps          ,
	
	float* w_gate            ,
	float* routed_experts_w1 ,
	float* routed_experts_w2 ,
	float* routed_experts_w3 ,
	float* shared_experts_w1 ,
	float* shared_experts_w2 ,
	float* shared_experts_w3 ,
	float* score_bias        ,
	
	const float ROUTE_SCALE,
	
	const int BATCH,
    const int SEQLEN,
    const int DIM,
    const int MOE_INTER_DIM,
    const int N_ACTIVATED_EXPERTS,
    const int N_ROUTED_EXPERTS,   
    const int N_SHARED_EXPERTS,
	const int N_EXPERT_GROUPS,
	const int N_LIMITED_GROUPS
)
{
	linear(score, u, w_gate, 1, BATCH * SEQLEN, DIM, N_ROUTED_EXPERTS);
	gate(v_gate, topk, score, score_bias, orginal_score, masked_score, group_score, 
	ROUTE_SCALE, BATCH, SEQLEN, N_ROUTED_EXPERTS, N_ACTIVATED_EXPERTS, N_EXPERT_GROUPS, N_LIMITED_GROUPS);

    for(int i = 0; i < BATCH * SEQLEN * N_ACTIVATED_EXPERTS; i++)
		counts[topk[i]] += 1;
	
	for(int i = 0; i < N_ROUTED_EXPERTS; i++)
	{
	    act_exps[i] = malloc(counts[i] * sizeof(int));
	    counts[i]   = 0;
	}
	
	for(int i = 0; i < BATCH * SEQLEN * N_ACTIVATED_EXPERTS; i++)
	{
		int exps = topk[i];
		act_exps[exps][counts[exps]] = i;
		
		counts[exps] += 1;
	}
	
	for(int i = 0; i < N_ROUTED_EXPERTS; i++)
	{
		if(counts[i] == 0)
			continue;
		
		int count = counts[i];
		
		float* routed_expert_w1_i = routed_experts_w1 + i * MOE_INTER_DIM * DIM;
		float* routed_expert_w2_i = routed_experts_w2 + i * MOE_INTER_DIM * DIM;
		float* routed_expert_w3_i = routed_experts_w3 + i * MOE_INTER_DIM * DIM;
		
		for(int j = 0; j < count; j++)
		{
			float* y_idx = y + (act_exps[i][j] / N_ACTIVATED_EXPERTS) * DIM;
			float* u_idx = u + (act_exps[i][j] / N_ACTIVATED_EXPERTS) * DIM;
			
			weighted_expert(y_idx, u_idx, routed_expert_w1_i, routed_expert_w2_i, routed_expert_w3_i, v_gate[act_exps[i][j]], DIM, MOE_INTER_DIM);
		}
	}
	
	mlp(z, u, shared_experts_w1, shared_experts_w2, shared_experts_w3, BATCH, SEQLEN, DIM, MOE_INTER_DIM * N_SHARED_EXPERTS);
	
	elemwise_add(h, y, z, BATCH * SEQLEN * DIM);

}


void block
(   
    A* a,
	W* w,
	
	float* fcos,
	float* fsin,
	
	int layer_id,
	
	const int   start_pos    ,
	const float softmax_scale,
	const float ROUTE_SCALE  ,
	
	const int BATCH           ,
	const int SEQLEN          ,
	const int MAX_BATCH       ,
	const int MAX_SEQLEN      ,
	const int N_LAYERS        ,      
	const int N_DENSE_LAYERS  ,
	const int N_MOE_LAYERS    ,  
	const int DIM             ,
	const int INTER_DIM       ,
	const int Q_LORA_RANK     ,
	const int KV_LORA_RANK    ,
	const int N_HEADS         , 
	const int QK_NOPE_HEAD_DIM,
	const int QK_ROPE_HEAD_DIM,
	const int V_HEAD_DIM      ,
	const int QK_HEAD_DIM     ,
	const int MOE_INTER_DIM   ,  
	const int N_ACTIVATED_EXPERTS,
	const int N_ROUTED_EXPERTS,
	const int N_SHARED_EXPERTS,  
	const int N_EXPERT_GROUPS ,  
	const int N_LIMITED_GROUPS   
)
{
	rmsnorm
	(
	    a->u_n0  + layer_id * BATCH * SEQLEN * DIM,
		
		a->u_0   + layer_id * BATCH * SEQLEN * DIM,
		
		w->atn_n + layer_id * DIM                 ,
		
		BATCH ,
		SEQLEN,
		DIM
	);
	
	attn
	(
	    a->u_atn      + layer_id * BATCH * SEQLEN * DIM                                    ,
		a->u_n0       + layer_id * BATCH * SEQLEN * DIM                                    ,
		a->q_a        + layer_id * BATCH * SEQLEN * Q_LORA_RANK                            ,
	    a->q_n        + layer_id * BATCH * SEQLEN * Q_LORA_RANK                            ,
	    a->q          + layer_id * BATCH * SEQLEN * (N_HEADS * QK_HEAD_DIM     )           ,
	    a->q_no       + layer_id * BATCH * SEQLEN * (N_HEADS * QK_NOPE_HEAD_DIM)           ,
	    a->q_pe       + layer_id * BATCH * SEQLEN * (N_HEADS * QK_ROPE_HEAD_DIM)           ,
	    a->qb_no      + layer_id * BATCH * SEQLEN * N_HEADS * KV_LORA_RANK                 ,
	    a->kv0        + layer_id * BATCH * SEQLEN * (KV_LORA_RANK + QK_ROPE_HEAD_DIM)      ,
	    a->kv1        + layer_id * BATCH * SEQLEN * KV_LORA_RANK                           ,
	    a->kv         + layer_id * BATCH * SEQLEN * KV_LORA_RANK                           ,
	    a->k_pe       + layer_id * BATCH * SEQLEN * QK_ROPE_HEAD_DIM                       ,
        a->scores     + layer_id * BATCH * SEQLEN * N_HEADS * (SEQLEN + start_pos)         ,
	    a->smscores   + layer_id * BATCH * SEQLEN * N_HEADS * (SEQLEN + start_pos)         ,
	    a->scoresb    + layer_id * BATCH * SEQLEN * N_HEADS * KV_LORA_RANK                 ,
	    a->ctx        + layer_id * BATCH * SEQLEN * N_HEADS * V_HEAD_DIM                   ,
	    a->kv_history + layer_id * BATCH * (SEQLEN + start_pos) * KV_LORA_RANK             ,
	    a->pe_history + layer_id * BATCH * (SEQLEN + start_pos) * QK_ROPE_HEAD_DIM         ,
		a->kv_cache   + layer_id * MAX_BATCH * MAX_SEQLEN * KV_LORA_RANK                   ,
		a->pe_cache   + layer_id * MAX_BATCH * MAX_SEQLEN * QK_ROPE_HEAD_DIM               ,
		
		w->wq_a       + layer_id * DIM * Q_LORA_RANK                                       ,
		w->wq_b       + layer_id * Q_LORA_RANK * (N_HEADS * QK_HEAD_DIM)                   ,
		w->wq_n       + layer_id * Q_LORA_RANK                                             ,
		w->wkv_a      + layer_id * DIM * (KV_LORA_RANK + QK_ROPE_HEAD_DIM)                 ,
		w->wkv_b      + layer_id * N_HEADS * KV_LORA_RANK * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
		w->wkv_n      + layer_id * KV_LORA_RANK                                            ,
		w->wo         + layer_id * (N_HEADS * V_HEAD_DIM) * DIM                            ,
		w->wkv_b0     + layer_id * KV_LORA_RANK * N_HEADS * QK_NOPE_HEAD_DIM               ,
		w->wkv_b1     + layer_id * KV_LORA_RANK * N_HEADS * V_HEAD_DIM                     ,
	
		fcos,
		fsin,
		
		start_pos    ,
		
		softmax_scale,
		
		BATCH           , 
		SEQLEN          ,
		DIM             ,
		MAX_BATCH       ,
		MAX_SEQLEN      , 
		Q_LORA_RANK     ,     
		KV_LORA_RANK    ,    
		N_HEADS         ,         
		QK_NOPE_HEAD_DIM,
		QK_ROPE_HEAD_DIM,
		V_HEAD_DIM      ,      
		QK_HEAD_DIM     
	);
	
	elemwise_add
	(
	    a->u_1   + layer_id * BATCH * SEQLEN * DIM,
		
		a->u_0   + layer_id * BATCH * SEQLEN * DIM,
		a->u_atn + layer_id * BATCH * SEQLEN * DIM,
		
		BATCH * SEQLEN * DIM
	);
	
	rmsnorm
	(
	    a->u_n1  + layer_id * BATCH * SEQLEN * DIM,
		
		a->u_1   + layer_id * BATCH * SEQLEN * DIM,
		
		w->ffn_n + layer_id * DIM                 ,
		
		BATCH,
		SEQLEN,
		DIM
	);
	
	if(layer_id < N_DENSE_LAYERS)
	{
		mlp
		(
		    a->u_ffn,
			
			a->u_n1,
			
			w->w1 + layer_id * DIM * INTER_DIM,
			w->w2 + layer_id * DIM * INTER_DIM,
			w->w3 + layer_id * DIM * INTER_DIM,
			
			BATCH,
	        SEQLEN,
            DIM,
            INTER_DIM
		);
	}
	else
    {
		moe
        (
            a->u_ffn             + layer_id * BATCH * SEQLEN * DIM,
        	
			a->u_n1              + layer_id * BATCH * SEQLEN * DIM,
        	
			a->y                 + (layer_id - 1) * BATCH * SEQLEN * DIM,
        	a->z                 + (layer_id - 1) * BATCH * SEQLEN * DIM,
        	a->v_gate            + (layer_id - 1) * BATCH * SEQLEN * N_ACTIVATED_EXPERTS,
        	a->topk              + (layer_id - 1) * BATCH * SEQLEN * N_ACTIVATED_EXPERTS,
        	a->score             + (layer_id - 1) * BATCH * SEQLEN * N_ROUTED_EXPERTS,
        	a->orginal_score     + (layer_id - 1) * BATCH * SEQLEN * N_ROUTED_EXPERTS,
        	a->masked_score      + (layer_id - 1) * BATCH * SEQLEN * N_ROUTED_EXPERTS,
        	a->group_score       + (layer_id - 1) * BATCH * SEQLEN * N_EXPERT_GROUPS ,
        	a->counts            + (layer_id - 1) * N_ROUTED_EXPERTS,
        	a->act_exps          + (layer_id - 1) * N_ROUTED_EXPERTS,
        	
        	w->w_gate            + (layer_id - 1) * N_ROUTED_EXPERTS * DIM                ,
        	w->routed_experts_w1 + (layer_id - 1) * N_ROUTED_EXPERTS * MOE_INTER_DIM * DIM,
        	w->routed_experts_w2 + (layer_id - 1) * N_ROUTED_EXPERTS * MOE_INTER_DIM * DIM,
        	w->routed_experts_w3 + (layer_id - 1) * N_ROUTED_EXPERTS * MOE_INTER_DIM * DIM,
        	w->shared_experts_w1 + (layer_id - 1) * N_SHARED_EXPERTS * MOE_INTER_DIM * DIM,
        	w->shared_experts_w2 + (layer_id - 1) * N_SHARED_EXPERTS * MOE_INTER_DIM * DIM,
        	w->shared_experts_w3 + (layer_id - 1) * N_SHARED_EXPERTS * MOE_INTER_DIM * DIM,
        	w->score_bias        + (layer_id - 1) * N_ROUTED_EXPERTS                      ,
        	
        	ROUTE_SCALE,
        	
        	BATCH,
            SEQLEN,
            DIM,
            MOE_INTER_DIM,
            N_ACTIVATED_EXPERTS,
            N_ROUTED_EXPERTS,   
            N_SHARED_EXPERTS,
        	N_EXPERT_GROUPS,
        	N_LIMITED_GROUPS
        );
	}
	
	elemwise_add
	(
		a->u_2   + layer_id * BATCH * SEQLEN * DIM,
		
		a->u_1   + layer_id * BATCH * SEQLEN * DIM,
		a->u_ffn + layer_id * BATCH * SEQLEN * DIM,
		
		BATCH * SEQLEN * DIM
	);
}	


/*

    Verification tools

*/

void copy_f(float* y, float* x, int size)
{
	for(int i = 0; i < size; i++)
		y[i] = x[i];
}

void reset_f(float* v, int size)
{
	for(int i = 0; i < size; i++)
		v[i] = 0.0f;
}

void reset_i(int* v, int size)
{
	for(int i = 0; i < size; i++)
		v[i] = 0;
}

void read_v(const char *filename, float *v, int size)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
	{
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < size; i++)
	{
        if (fscanf(file, "%f", &v[i]) != 1)
		{
            perror("Error reading data");
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }

    fclose(file);
}

void compare(float* v_c, float* v_py, int size)
{
	int pass = 1;
	int count = 0;
	
	for(int i = 0; i < size; i++)
	{
		if(fabs((v_c[i] - v_py[i]) / (v_py[i] + 1e-7f)) > 1e-3f)
		{
			//printf("%d\n", i);
			//printf("%f, %f\n", v_c[i], v_py[i]);
			//pass = 0;
			//break;
			count += 1;
		}
	}
	
	printf("Rate: %f\n", (float)(size - count) / (float)(size));
	
	//if(pass == 1)
	//	printf("passed\n");
	//else
	//	printf("failed\n");
}


int main()
{
	//Feature of Input
	const int BATCH  = 4 ;
	const int SEQLEN = 64;

	//Feature of Model
	const int MAX_BATCH        = 4  ;
	const int MAX_SEQLEN       = 256;
	const int ORIGINAL_SEQLEN  = 64 ;
	const int N_LAYERS         = 8  ;
	const int N_DENSE_LAYERS   = 1  ;
	const int N_MOE_LAYERS     = N_LAYERS - N_DENSE_LAYERS;
	
	const int DIM              = 256;
	
	//MLP
	const int INTER_DIM        = 704;
	
	//MLA
	const int Q_LORA_RANK      = 108;
	const int KV_LORA_RANK     = 36 ;
	const int N_HEADS          = 4  ;
	const int QK_NOPE_HEAD_DIM = 32 ;
	const int QK_ROPE_HEAD_DIM = 16 ;
	const int V_HEAD_DIM       = 32 ;
	const int QK_HEAD_DIM      = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;
	
	//MoE
	const int MOE_INTER_DIM       = 88;
	const int N_ACTIVATED_EXPERTS = 4 ;
	const int N_ROUTED_EXPERTS    = 32;
	const int N_SHARED_EXPERTS    = 1 ;
	const int N_EXPERT_GROUPS     = 4 ;
	const int N_LIMITED_GROUPS    = 2 ;
	
	const float ROUTE_SCALE = 2.5f;
	const float MSCALE = 1.0f;
	
	float softmax_scale = sqrtf(1.0f / QK_HEAD_DIM);
	
	int layer_id;
	int start_pos = 0;
	
	if(MAX_SEQLEN > ORIGINAL_SEQLEN)
	{
		float mscale = 0.1f * MSCALE * logf(40.0f) + 1.0f;
		softmax_scale = softmax_scale * mscale * mscale;
	}
	
	W w;
	A a;
	
	//output
	float* h      = (float *)malloc(BATCH * SEQLEN * DIM * sizeof(float));
	float* h_py   = (float *)malloc(BATCH * SEQLEN * DIM * sizeof(float));
	float* x_py   = (float *)malloc(BATCH * SEQLEN * DIM * sizeof(float));
	float* y_py   = (float *)malloc(BATCH * SEQLEN * DIM * sizeof(float));
	
	//input
	float* u      = (float *)malloc(BATCH * SEQLEN * DIM * sizeof(float));
	
	float* fcos  = (float *)malloc(MAX_SEQLEN * QK_ROPE_HEAD_DIM / 2 * sizeof(float));
	float* fsin  = (float *)malloc(MAX_SEQLEN * QK_ROPE_HEAD_DIM / 2 * sizeof(float));
	
	w.atn_n  = (float *)malloc(N_LAYERS * DIM * sizeof(float));
	w.ffn_n  = (float *)malloc(N_LAYERS * DIM * sizeof(float));
	
	w.wq_a   = (float *)malloc(N_LAYERS * DIM * Q_LORA_RANK                                        * sizeof(float));
	w.wq_b   = (float *)malloc(N_LAYERS * Q_LORA_RANK * (N_HEADS * QK_HEAD_DIM)                    * sizeof(float));
	w.wq_n   = (float *)malloc(N_LAYERS * Q_LORA_RANK                                              * sizeof(float));
	w.wkv_a  = (float *)malloc(N_LAYERS * DIM * (KV_LORA_RANK + QK_ROPE_HEAD_DIM)                  * sizeof(float));
	w.wkv_b  = (float *)malloc(N_LAYERS * N_HEADS * KV_LORA_RANK * (QK_NOPE_HEAD_DIM + V_HEAD_DIM) * sizeof(float));
	w.wkv_n  = (float *)malloc(N_LAYERS * KV_LORA_RANK                                             * sizeof(float));
	w.wo     = (float *)malloc(N_LAYERS * (N_HEADS * V_HEAD_DIM) * DIM                             * sizeof(float));
	w.wkv_b0 = (float *)malloc(N_LAYERS * KV_LORA_RANK * N_HEADS * QK_NOPE_HEAD_DIM                * sizeof(float));
	w.wkv_b1 = (float *)malloc(N_LAYERS * KV_LORA_RANK * N_HEADS * V_HEAD_DIM                      * sizeof(float));
	
	w.w_gate            = (float *)malloc(N_MOE_LAYERS * N_ROUTED_EXPERTS * DIM                 * sizeof(float));
	w.routed_experts_w1 = (float *)malloc(N_MOE_LAYERS * N_ROUTED_EXPERTS * MOE_INTER_DIM * DIM * sizeof(float));
	w.routed_experts_w2 = (float *)malloc(N_MOE_LAYERS * N_ROUTED_EXPERTS * MOE_INTER_DIM * DIM * sizeof(float));
	w.routed_experts_w3 = (float *)malloc(N_MOE_LAYERS * N_ROUTED_EXPERTS * MOE_INTER_DIM * DIM * sizeof(float));
	w.shared_experts_w1 = (float *)malloc(N_MOE_LAYERS * N_SHARED_EXPERTS * MOE_INTER_DIM * DIM * sizeof(float));
	w.shared_experts_w2 = (float *)malloc(N_MOE_LAYERS * N_SHARED_EXPERTS * MOE_INTER_DIM * DIM * sizeof(float));
	w.shared_experts_w3 = (float *)malloc(N_MOE_LAYERS * N_SHARED_EXPERTS * MOE_INTER_DIM * DIM * sizeof(float));
	w.score_bias        = (float *)malloc(N_MOE_LAYERS * N_ROUTED_EXPERTS                       * sizeof(float));
	
	w.w1 = (float *)malloc(N_DENSE_LAYERS * INTER_DIM * DIM * sizeof(float));
	w.w2 = (float *)malloc(N_DENSE_LAYERS * INTER_DIM * DIM * sizeof(float));
	w.w3 = (float *)malloc(N_DENSE_LAYERS * INTER_DIM * DIM * sizeof(float));
	
	//block
	a.u_0   = (float *)malloc(N_LAYERS * BATCH * SEQLEN * DIM * sizeof(float));
	a.u_n0  = (float *)malloc(N_LAYERS * BATCH * SEQLEN * DIM * sizeof(float));
	a.u_atn = (float *)malloc(N_LAYERS * BATCH * SEQLEN * DIM * sizeof(float));
	a.u_1   = (float *)malloc(N_LAYERS * BATCH * SEQLEN * DIM * sizeof(float));
	a.u_n1  = (float *)malloc(N_LAYERS * BATCH * SEQLEN * DIM * sizeof(float));
	a.u_ffn = (float *)malloc(N_LAYERS * BATCH * SEQLEN * DIM * sizeof(float));
	a.u_2   = (float *)malloc(N_LAYERS * BATCH * SEQLEN * DIM * sizeof(float));
	
	//MLA
	a.q_a        = (float *)malloc(N_LAYERS * BATCH * SEQLEN * Q_LORA_RANK                       * sizeof(float));
	a.q_n        = (float *)malloc(N_LAYERS * BATCH * SEQLEN * Q_LORA_RANK                       * sizeof(float));
	a.q          = (float *)malloc(N_LAYERS * BATCH * SEQLEN * (N_HEADS * QK_HEAD_DIM     )      * sizeof(float));
	a.q_no       = (float *)malloc(N_LAYERS * BATCH * SEQLEN * (N_HEADS * QK_NOPE_HEAD_DIM)      * sizeof(float));
	a.q_pe       = (float *)malloc(N_LAYERS * BATCH * SEQLEN * (N_HEADS * QK_ROPE_HEAD_DIM)      * sizeof(float));
    a.qb_no      = (float *)malloc(N_LAYERS * BATCH * SEQLEN * N_HEADS * KV_LORA_RANK            * sizeof(float));
	a.kv0        = (float *)malloc(N_LAYERS * BATCH * SEQLEN * (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * sizeof(float));
	a.kv1        = (float *)malloc(N_LAYERS * BATCH * SEQLEN * KV_LORA_RANK                      * sizeof(float));
	a.kv         = (float *)malloc(N_LAYERS * BATCH * SEQLEN * KV_LORA_RANK                      * sizeof(float));
	a.k_pe       = (float *)malloc(N_LAYERS * BATCH * SEQLEN * QK_ROPE_HEAD_DIM                  * sizeof(float));
	a.scores     = (float *)malloc(N_LAYERS * BATCH * SEQLEN * N_HEADS * (SEQLEN + start_pos)    * sizeof(float));
	a.smscores   = (float *)malloc(N_LAYERS * BATCH * SEQLEN * N_HEADS * (SEQLEN + start_pos)    * sizeof(float));
	a.scoresb    = (float *)malloc(N_LAYERS * BATCH * SEQLEN * N_HEADS * KV_LORA_RANK            * sizeof(float));
	a.ctx        = (float *)malloc(N_LAYERS * BATCH * SEQLEN * N_HEADS * V_HEAD_DIM              * sizeof(float));
	a.kv_history = (float *)malloc(N_LAYERS * BATCH * (SEQLEN + start_pos) * KV_LORA_RANK        * sizeof(float));
	a.pe_history = (float *)malloc(N_LAYERS * BATCH * (SEQLEN + start_pos) * QK_ROPE_HEAD_DIM    * sizeof(float));
	a.kv_cache   = (float *)malloc(N_LAYERS * MAX_BATCH * MAX_SEQLEN * KV_LORA_RANK              * sizeof(float));
	a.pe_cache   = (float *)malloc(N_LAYERS * MAX_BATCH * MAX_SEQLEN * QK_ROPE_HEAD_DIM          * sizeof(float));
	
	//MoE
	a.y             = (float *)malloc(N_MOE_LAYERS * BATCH * SEQLEN * DIM                 * sizeof(float));
	a.z             = (float *)malloc(N_MOE_LAYERS * BATCH * SEQLEN * DIM                 * sizeof(float));
	a.v_gate        = (float *)malloc(N_MOE_LAYERS * BATCH * SEQLEN * N_ACTIVATED_EXPERTS * sizeof(float));
	a.topk          = (int   *)malloc(N_MOE_LAYERS * BATCH * SEQLEN * N_ACTIVATED_EXPERTS * sizeof(int  ));
	a.score         = (float *)malloc(N_MOE_LAYERS * BATCH * SEQLEN * N_ROUTED_EXPERTS    * sizeof(float));
	a.orginal_score = (float *)malloc(N_MOE_LAYERS * BATCH * SEQLEN * N_ROUTED_EXPERTS    * sizeof(float));
	a.masked_score  = (float *)malloc(N_MOE_LAYERS * BATCH * SEQLEN * N_ROUTED_EXPERTS    * sizeof(float));
	a.group_score   = (float *)malloc(N_MOE_LAYERS * BATCH * SEQLEN * N_EXPERT_GROUPS     * sizeof(float));
	a.counts        = (int   *)malloc(N_MOE_LAYERS * N_ROUTED_EXPERTS                     * sizeof(int  ));
	a.act_exps      = (int  **)malloc(N_MOE_LAYERS * N_ROUTED_EXPERTS                     * sizeof(int* ));
	
	reset_f(a.y     , N_MOE_LAYERS * BATCH * SEQLEN * DIM);
	reset_f(a.z     , N_MOE_LAYERS * BATCH * SEQLEN * DIM);
	reset_i(a.counts, N_MOE_LAYERS * N_ROUTED_EXPERTS);
	
	reset_f(a.scores, N_LAYERS * BATCH * SEQLEN * N_HEADS * (SEQLEN + start_pos));
	reset_f(a.ctx   , N_LAYERS * BATCH * SEQLEN * N_HEADS * V_HEAD_DIM);
	
	read_v("atn_n.txt" , w.atn_n, N_LAYERS * DIM);
	read_v("ffn_n.txt" , w.ffn_n, N_LAYERS * DIM);
	
	read_v("wq_a.txt"  , w.wq_a , N_LAYERS * DIM * Q_LORA_RANK                                       );
	read_v("wq_b.txt"  , w.wq_b , N_LAYERS * Q_LORA_RANK * (N_HEADS * QK_HEAD_DIM)                   );
	read_v("wq_n.txt"  , w.wq_n , N_LAYERS * Q_LORA_RANK                                             );
	read_v("wkv_a.txt" , w.wkv_a, N_LAYERS * DIM * (KV_LORA_RANK + QK_ROPE_HEAD_DIM)                 );
	read_v("wkv_b.txt" , w.wkv_b, N_LAYERS * N_HEADS * KV_LORA_RANK * (QK_NOPE_HEAD_DIM + V_HEAD_DIM));
	read_v("wkv_n.txt" , w.wkv_n, N_LAYERS * KV_LORA_RANK                                            );
	read_v("wo.txt"    , w.wo   , N_LAYERS * (N_HEADS * V_HEAD_DIM) * DIM                            );
	
	read_v("w_gate.txt"           , w.w_gate           , N_MOE_LAYERS * N_ROUTED_EXPERTS * DIM                );
	read_v("routed_experts_w1.txt", w.routed_experts_w1, N_MOE_LAYERS * N_ROUTED_EXPERTS * MOE_INTER_DIM * DIM);
	read_v("routed_experts_w2.txt", w.routed_experts_w2, N_MOE_LAYERS * N_ROUTED_EXPERTS * MOE_INTER_DIM * DIM);
	read_v("routed_experts_w3.txt", w.routed_experts_w3, N_MOE_LAYERS * N_ROUTED_EXPERTS * MOE_INTER_DIM * DIM);
	read_v("shared_experts_w1.txt", w.shared_experts_w1, N_MOE_LAYERS * N_SHARED_EXPERTS * MOE_INTER_DIM * DIM);
	read_v("shared_experts_w2.txt", w.shared_experts_w2, N_MOE_LAYERS * N_SHARED_EXPERTS * MOE_INTER_DIM * DIM);
	read_v("shared_experts_w3.txt", w.shared_experts_w3, N_MOE_LAYERS * N_SHARED_EXPERTS * MOE_INTER_DIM * DIM);
	read_v("score_bias.txt"       , w.score_bias       , N_MOE_LAYERS * N_ROUTED_EXPERTS                      );
	
	read_v("w1.txt", w.w1, N_DENSE_LAYERS * INTER_DIM * DIM);
	read_v("w2.txt", w.w2, N_DENSE_LAYERS * INTER_DIM * DIM);
	read_v("w3.txt", w.w3, N_DENSE_LAYERS * INTER_DIM * DIM);
	
	read_v("u.txt"     , u        , BATCH * SEQLEN * DIM);
	read_v("h.txt"     , h_py     , BATCH * SEQLEN * DIM);
	read_v("x.txt"     , x_py     , BATCH * SEQLEN * DIM);
	read_v("y.txt"     , y_py     , BATCH * SEQLEN * DIM);
	
	precompute_freqs_cis(fcos, fsin, MAX_SEQLEN, ORIGINAL_SEQLEN, QK_ROPE_HEAD_DIM);
	
	//block
	
	copy_f(a.u_0 + 0 * BATCH * SEQLEN * DIM, u, BATCH * SEQLEN * DIM);
	
	for(layer_id = 0; layer_id < N_LAYERS; layer_id++)
	{
		block
		(
		    &a, &w, fcos, fsin, layer_id, start_pos, softmax_scale, ROUTE_SCALE,
	        BATCH,
	        SEQLEN,
	        MAX_BATCH,
	        MAX_SEQLEN,
	        N_LAYERS,      
	        N_DENSE_LAYERS,
	        N_MOE_LAYERS,  
	        DIM,
	        INTER_DIM,
	        Q_LORA_RANK,
	        KV_LORA_RANK,
	        N_HEADS,
	        QK_NOPE_HEAD_DIM,
	        QK_ROPE_HEAD_DIM,
	        V_HEAD_DIM,
	        QK_HEAD_DIM,
	        MOE_INTER_DIM,  
	        N_ACTIVATED_EXPERTS,
	        N_ROUTED_EXPERTS,
	        N_SHARED_EXPERTS,  
	        N_EXPERT_GROUPS,  
	        N_LIMITED_GROUPS   
        );
		
		if(layer_id < N_LAYERS - 1)
			copy_f(a.u_0 + (layer_id + 1) * BATCH * SEQLEN * DIM, a.u_2 + layer_id * BATCH * SEQLEN * DIM, BATCH * SEQLEN * DIM);
	    else
			copy_f(h, a.u_2 + (N_LAYERS - 1) * BATCH * SEQLEN * DIM, BATCH * SEQLEN * DIM);
	}
	
	compare(h, h_py, BATCH * SEQLEN * DIM);
	compare(a.u_2, x_py, BATCH * SEQLEN * DIM);
	compare(a.u_2 + BATCH * SEQLEN * DIM, y_py, BATCH * SEQLEN * DIM);
	
	
	return 0;
}