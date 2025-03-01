#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <time.h>

void matmul(float* y, float* x, float* w, int B, int T, int C, int OC)
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
                y[bt * OC + o] = val;
            }
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

void elemwise_add(float* y, float* x0, float* x1, int N)
{
	for(int i = 0; i < N; i++)
		y[i] = x0[i] + x1[i];
}

void gate
(
	float *v_gate,
	int   *topk  ,
    
    float *score,	
	
	int B,
	int T,
	int N_ROUTED_EXPERTS,
	int N_ACTIVATED_EXPERTS
)
{
	for (int b = 0; b < B; b++)
	{
        for (int t = 0; t < T; t++)
		{
			int offset0 = b * T * N_ROUTED_EXPERTS    + t * N_ROUTED_EXPERTS   ;
			int offset1 = b * T * N_ACTIVATED_EXPERTS + t * N_ACTIVATED_EXPERTS;
            
			float vmax;
			float expsum = 0.0f;
			
			float* score_bt  = score  + offset0;
			float* v_gate_bt = v_gate + offset1;
			int  * topk_bt   = topk   + offset1;
			
			for(int k = 0; k < N_ACTIVATED_EXPERTS; k++)
			{
				v_gate_bt[k] = -FLT_MAX;
				topk_bt[k] = -1;
			}
			
			for(int i = 0; i < N_ROUTED_EXPERTS; i++)
			{
				int pos = -1;
				
				for(int k = 0; k < N_ACTIVATED_EXPERTS; k++)
				{
					if(score_bt[i] > v_gate_bt[k])
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
						v_gate_bt[k] = v_gate_bt[k + 1];
						topk_bt[k] = topk_bt[k + 1];
					}
					
					v_gate_bt[pos] = score_bt[i];
					topk_bt[pos] = i;
				}
			}
			
			vmax = v_gate_bt[N_ACTIVATED_EXPERTS - 1];
			
			for(int i = 0; i < N_ROUTED_EXPERTS; i++)
			{
				score_bt[i] = score_bt[i] - vmax;
				score_bt[i] = expf(score_bt[i]);
				
				expsum += score_bt[i];
			}
			
			for(int k = 0; k < N_ACTIVATED_EXPERTS; k++)
			{
				v_gate_bt[k] = score_bt[topk_bt[k]] / expsum;
			}
        }
    }
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
	
	matmul(u1, u_idx, routed_expert_w1_i, 1, 1, DIM, MOE_INTER_DIM);
	matmul(u3, u_idx, routed_expert_w3_i, 1, 1, DIM, MOE_INTER_DIM);
	
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

void mlp
(
    float* z_idx,
	
    float* u_idx,

    float* shared_expert_w1,
    float* shared_expert_w2,
    float* shared_expert_w3,
	
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
	
	matmul(u1, u_idx, shared_expert_w1, B, T, DIM, INTER_DIM);
	matmul(u3, u_idx, shared_expert_w3, B, T, DIM, INTER_DIM);
	
	silu(s1, u1, B * T * INTER_DIM);
	elemwise_dot(u2, s1, u3, B * T * INTER_DIM);
	
	matmul_acc(z_idx, u2, shared_expert_w2, B, T, INTER_DIM, DIM);
	
	free(u1);
	free(s1);
	free(u3);
	free(u2);
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
	
	for(int i = 0; i < size; i++)
	{
		if(fabs((v_c[i] - v_py[i]) / (v_py[i] + 1e-7f)) > 5e-2f)
		{
			printf("%d\n", i);
			printf("%f, %f\n", v_c[i], v_py[i]);
			pass = 0;
			break;
		}
	}
	
	if(pass == 1)
		printf("passed\n");
	else
		printf("failed\n");
}

int main()
{
	const int BATCH  = 8 ;
	const int SEQLEN = 64;
	
	const int DIM = 256;
	const int MOE_INTER_DIM = 176;
	
	const int N_ACTIVATED_EXPERTS = 6 ;
	const int N_ROUTED_EXPERTS    = 32;
	const int N_SHARED_EXPERTS    = 2 ;
	
	//input
	float* u      = (float *)malloc(BATCH * SEQLEN * DIM * sizeof(float));
	
	//output
	float* h      = (float *)malloc(BATCH * SEQLEN * DIM * sizeof(float));
	float* h_py   = (float *)malloc(BATCH * SEQLEN * DIM * sizeof(float));
	
	//activation
	float* y      = (float *)malloc(BATCH * SEQLEN * DIM * sizeof(float));
	float* z      = (float *)malloc(BATCH * SEQLEN * DIM * sizeof(float));
	
	float* v_gate = (float *)malloc(BATCH * SEQLEN * N_ACTIVATED_EXPERTS * sizeof(float));
	int*   topk   = (int   *)malloc(BATCH * SEQLEN * N_ACTIVATED_EXPERTS * sizeof(int));
	
	float* score  = (float *)malloc(BATCH * SEQLEN * N_ROUTED_EXPERTS * sizeof(float));
	
	int*  counts   = (int *)malloc(N_ROUTED_EXPERTS * sizeof(int ));
	int** act_exps =        malloc(N_ROUTED_EXPERTS * sizeof(int*));
	
	//weights
	float* w_gate  = (float *)malloc(N_ROUTED_EXPERTS * DIM * sizeof(float));
	
	float* routed_experts_w1 = (float *)malloc(N_ROUTED_EXPERTS * MOE_INTER_DIM * DIM * sizeof(float));
	float* routed_experts_w2 = (float *)malloc(N_ROUTED_EXPERTS * MOE_INTER_DIM * DIM * sizeof(float));
	float* routed_experts_w3 = (float *)malloc(N_ROUTED_EXPERTS * MOE_INTER_DIM * DIM * sizeof(float));
	
	float* shared_experts_w1 = (float *)malloc(N_SHARED_EXPERTS * MOE_INTER_DIM * DIM * sizeof(float));
	float* shared_experts_w2 = (float *)malloc(N_SHARED_EXPERTS * MOE_INTER_DIM * DIM * sizeof(float));
	float* shared_experts_w3 = (float *)malloc(N_SHARED_EXPERTS * MOE_INTER_DIM * DIM * sizeof(float));
	
	reset_f(u     , BATCH * SEQLEN * DIM);
	reset_f(h     , BATCH * SEQLEN * DIM);
	reset_f(y     , BATCH * SEQLEN * DIM);
	reset_f(z     , BATCH * SEQLEN * DIM);
	reset_f(v_gate, BATCH * SEQLEN * N_ACTIVATED_EXPERTS);
	reset_f(score , BATCH * SEQLEN * N_ROUTED_EXPERTS);
	reset_i(counts, N_ROUTED_EXPERTS);
	
	read_v("u.txt"     , u        , BATCH * SEQLEN * DIM);
	read_v("h.txt"     , h_py     , BATCH * SEQLEN * DIM);
	read_v("w_gate.txt", w_gate   , N_ROUTED_EXPERTS * DIM);
	
	read_v("routed_experts_w1.txt", routed_experts_w1, N_ROUTED_EXPERTS * MOE_INTER_DIM * DIM);
	read_v("routed_experts_w2.txt", routed_experts_w2, N_ROUTED_EXPERTS * MOE_INTER_DIM * DIM);
	read_v("routed_experts_w3.txt", routed_experts_w3, N_ROUTED_EXPERTS * MOE_INTER_DIM * DIM);
	
	read_v("shared_experts_w1.txt", shared_experts_w1, N_SHARED_EXPERTS * MOE_INTER_DIM * DIM);
	read_v("shared_experts_w2.txt", shared_experts_w2, N_SHARED_EXPERTS * MOE_INTER_DIM * DIM);
	read_v("shared_experts_w3.txt", shared_experts_w3, N_SHARED_EXPERTS * MOE_INTER_DIM * DIM);
	
	//MOE Forward
	clock_t t0; 
    t0 = clock();
	
	matmul(score, u, w_gate, 1, BATCH * SEQLEN, DIM, N_ROUTED_EXPERTS);
	gate(v_gate, topk, score, BATCH, SEQLEN, N_ROUTED_EXPERTS, N_ACTIVATED_EXPERTS);
	
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
	
	t0 = clock() - t0; 
    double time_taken = ((double)t0)/CLOCKS_PER_SEC; // in seconds 
	
	printf("moe forward took %f seconds to execute. \n", time_taken);
	
	//
	
	compare(h, h_py, BATCH * SEQLEN * DIM);
	
	for(int i = 0; i < N_ROUTED_EXPERTS; i++)
	    free(act_exps[i]);
	free(u);
	free(h);
	free(y);
	free(z);
	free(h_py);
	free(v_gate);
	free(topk);
	free(score);
	free(counts);
	free(act_exps);
	free(w_gate);
	free(routed_experts_w1);
	free(routed_experts_w2);
	free(routed_experts_w3);
	free(shared_experts_w1);
	free(shared_experts_w2);
	free(shared_experts_w3);
	
	return 0;
}