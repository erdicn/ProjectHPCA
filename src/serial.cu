#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <curand_kernel.h>

// Function that catches the error
void testCUDA(cudaError_t error, const char *file, int line)
{

	if (error != cudaSuccess)
	{
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__, __LINE__))

void printArray(int *a, int len)
{
	for (int i = 0; i < len; i++)
	{
		std::cout << a[i] << std::endl;
	}
}

// Given an empty vector fills it with the weigt and biases vector from file
int readArrayFromFile(int sizeWB, std::string path, std::vector<float> &WB_data)
{
	// std::string path = "weights.txt";  // path of the file with one value per row

	// Reading the file
	std::ifstream file(path);
	if (!file.is_open())
	{
		std::cerr << "Error: impossible to open " << path << std::endl;
		return 1;
	}

	// std::vector<float> WB_data;
	WB_data.reserve(sizeWB);

	float value;
	while (file >> value)
	{
		WB_data.push_back(value);
	}
	file.close();

	if (WB_data.size() != sizeWB)
	{
		std::cerr << "Error: " << WB_data.size() << " read values " << sizeWB << std::endl;
		return 1;
	}

	std::cout << "read done of weights: " << WB_data.size() << " floats." << std::endl;
	return 0;
}

void initializeDomainsArray(int nb_layers, int input_layer_dim, int output_layer_dim, int *domains)
{
	domains[0] = input_layer_dim; // input layer
	for (int i = 1; i < nb_layers - 1; i++)
		domains[i] = 4;			// hidden layers
	domains[nb_layers - 1] = 1; // output layer
}

int getSizeOfWeightAndBiases(int nb_layers, int *domains)
{
	int sizeWB = 0;
	for (int i = 0; i < nb_layers - 1; i++)
	{
		sizeWB += (domains[i] + 1) * domains[i + 1];
	}
	return sizeWB;
}

int getNbOfPossibleCOmbinationsOfNeurons(int nb_layers, int *domains)
{
	int nb_subpolytopes = 0;
	for (int i = 1; i < nb_layers - 1; i++)
	{
		nb_subpolytopes += domains[i];
	}
	nb_subpolytopes = 1 << nb_subpolytopes; // 2^nb_neurons
	return nb_subpolytopes;
}

void getDimensionsCB(int m0, int* deltaSize, int* sizeB, int* sizeCB, int nb_layers, int* domains_CPU){
	*deltaSize = m0;
	*sizeB     = m0;
	*sizeCB    = m0;
	for (int i = 1; i < nb_layers; i++)
	{
		*deltaSize += domains_CPU[i];
		*sizeB     += domains_CPU[i];
		*sizeCB    += domains_CPU[i];
	}
	*sizeCB *= 1 + domains_CPU[0];

	for (int k = 1; k < 1 + domains_CPU[0]; k++)
	{
		*sizeCB += (*deltaSize - k) * (1 + domains_CPU[0] - k);
		*sizeB  += (*deltaSize - k);
	}
}

long long nCr(int n, int r)
{
	if (r > n)
		return 0;
	if (r == 0 || r == n)
		return 1;

	// Optimization: nCr(n, r) == nCr(n, n-r)
	// We want to calculate the smaller number of multiplications
	if (r > n / 2)
		r = n - r;

	long long res = 1;
	for (int i = 1; i <= r; ++i)
	{
		res = res * (n - i + 1);
		res = res / i;
	}
	return res;
}

#define INPUT_LAYER_DIM 3
#define NB_LAYERS 4 + 1 // TODO read from external file so it syncs with python

typedef struct
{
	int ver;
	int lvl;
} Num_t;

typedef struct
{
	int min;
	int max;
	int final_max;
} MinMax_t;

#define MIN(x, y) ((x < y) ? x : y)
#define MAX(x, y) ((x > y) ? x : y)

__device__ __forceinline__ float leakyReLU(float z){
	return MAX(z, 0) + 0.01*MIN(z, 0);
}

// Shifts sl int by i then gets that bit *2 to get 0 or 2 then -1 to get 1 if 1 -1 if 0
#define GET_SL(sl, i) (sl >> i & 1U)*2 -1

// It goes from the start polytope index for nb_polytopes for now we supose everything fits on the gpu so we take all
__global__ void partitionPolytope_k(int *d, float *WB, int nb_layers, 
									int nb_polytopes, int start_polytope, int total_polytopes
									, float* in_CdBd, float*  out_C0B0, float* CB, int CB_size_tot
									, int m0){
	unsigned int poly_id = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ float lol[];
	int CdBd_size = 2 * d[0] *d[0] + 2* d[0];
	unsigned int sl = poly_id; // we need 1 sl per polytope since we are stroting in its bits (4*8 = 24) rn we can do 2**24 polytopes max with int 
	int beta_i, beta_i_m1, c_i, c_i_m1, poly_decalage = CB_size_tot*poly_id+CdBd_size;
	int i, j, l;

	for(i = 0; i < m0; i++){
		for(j = 0; j < d[0]; j++){
			c_i = (d[0]+1) * i + j;
			CB[c_i+ poly_decalage - CdBd_size] = in_CdBd[c_i]; 
		}
		beta_i = (d[0]+1) * i + d[0];
		CB[beta_i + poly_decalage - CdBd_size] = in_CdBd[beta_i];
	}

	// Initialisation
	for(i = 0; i < d[1]; i++){
		// -diag(s1) W1 and gets s1 through bit shifting
		for( j = 0; j < d[0]; j++){
			c_i = (d[0]+1) * i + j;
			// C0
			CB[c_i + poly_decalage] = -GET_SL(sl, i) * WB[c_i];
		}
		// Beta0
		beta_i = (d[0]+1) * i + d[0];
		CB[beta_i + poly_decalage] = -GET_SL(sl, i) * WB[beta_i];
	}

	int decalage_sl    = 0;
	int decalage_layer = 0;
	int inbetween_two_layers = 0;
	for(l = 2; l < nb_layers - 1; l++){
		decalage_sl    += d[l-1];
		inbetween_two_layers = (d[l-2] + 1) * d[l-1];
		decalage_layer += inbetween_two_layers;
		for(i = 0; i < d[l]; i++){
			for(j = 0; j < d[l-1]; j++){
				c_i   = (d[l-1]+1) * i + j + decalage_layer;
				c_i_m1= c_i - inbetween_two_layers;
				// Cl
				CB[c_i + poly_decalage] = GET_SL(sl, i+decalage_sl) * WB[c_i] * // TODO right multiplication
					leakyReLU(GET_SL(sl, i+decalage_sl-d[l-1])) * CB[c_i_m1 + poly_decalage];
			}
			// Betal
			beta_i    = (d[l-1] + 1) * i + d[l-1] + decalage_layer;
			beta_i_m1 = beta_i - inbetween_two_layers;
			CB[beta_i + poly_decalage] = GET_SL(sl, i+decalage_sl) * 
											(WB[c_i] * leakyReLU(GET_SL(sl, i+decalage_sl-d[l-1])) * 
												CB[beta_i_m1 + poly_decalage] + WB[beta_i]);
		}
	}

	decalage_sl    += d[l-1];
	inbetween_two_layers = (d[l-2] + 1) * d[l-1];
	decalage_layer += inbetween_two_layers;
	// C0 and B0
	for(i = 0; i < d[l]; i++){
		for(j = 0; j < d[l-1]; j++){
			c_i   = (d[l-1]+1) * i + j + decalage_layer;
			c_i_m1= c_i - inbetween_two_layers;
			CB[c_i + poly_decalage] = -WB[c_i] * leakyReLU(GET_SL(sl, i+decalage_sl-d[l-1])) *
													 CB[c_i_m1 + poly_decalage];
		}
		beta_i    = (d[l-1] + 1) * i + d[l-1] + decalage_layer;
		beta_i_m1 = beta_i - inbetween_two_layers;
		CB[beta_i + poly_decalage] = -(WB[c_i] * leakyReLU(GET_SL(sl, i+decalage_sl-d[l-1])) * 
											CB[beta_i_m1 + poly_decalage] + WB[beta_i]);
	}
}

int main(void)
{
	// TOOD from file
	int domains_CPU[NB_LAYERS];
	initializeDomainsArray(NB_LAYERS, INPUT_LAYER_DIM, 1, domains_CPU);
	int* domains_GPU;
	testCUDA(cudaMalloc(&domains_GPU, NB_LAYERS*sizeof(int)));
	testCUDA(cudaMemcpy(domains_GPU, domains_CPU, NB_LAYERS*sizeof(int), cudaMemcpyHostToDevice));

	// nb_connections_layers+biases = nn_i * nn_i+1 + nn_i+1
	int sizeWB = getSizeOfWeightAndBiases(NB_LAYERS, domains_CPU);
	std::cout << sizeWB << std::endl;

	MinMax_t *min_max;
	testCUDA(cudaMallocManaged(&min_max, sizeof(MinMax_t)));
	min_max->min = 5000000;
	min_max->max = -5000000;
	min_max->final_max = 0;

	int nb_subpolytopes = getNbOfPossibleCOmbinationsOfNeurons(NB_LAYERS, domains_CPU); // number of possible combinations of neurons active inactive 2^nb_neurons in the hidden layers

	// Dimensions of CB matrix and more
	int m0 = 2 * domains_CPU[0];
	int deltaSize, sizeB, sizeCB;
	getDimensionsCB(m0, &deltaSize, &sizeB, &sizeCB, NB_LAYERS, domains_CPU);

	// maximum number of vertices by subpolytope: d0 combinations among m-d0 m0 = 2*d0
	int m = m0;
	for (int i = 1; i < NB_LAYERS; i++) 
		m += domains_CPU[i];
	int nb_vertices = nCr(m, domains_CPU[0]);
	std::cout << nb_vertices << std::endl;

	int magic_valuesCPU[5], *magic_values;
	// testCUDA(cudaMallocManaged(&magic_values, 5*sizeof(int)));
	magic_valuesCPU[0] = sizeCB - deltaSize * (1 + domains_CPU[0]); // Values should be stored at the end Algo (4.4) deltaC
	magic_valuesCPU[1] = sizeCB;									// The total size needed for each configuration s DeltaC
	magic_valuesCPU[2] = nb_vertices;
	magic_valuesCPU[3] = sizeB - deltaSize; // Values should be stored at the end Algo (4.4) deltaB
	magic_valuesCPU[4] = sizeB;				// The total size needed for each configuration s DeltaB TODO never used ???

	testCUDA(cudaMalloc(&magic_values, 5 * sizeof(int)));
	testCUDA(cudaMemcpy(magic_values, magic_valuesCPU, 5 * sizeof(int), cudaMemcpyHostToDevice));
	testCUDA(cudaMemcpyToSymbol(dld, domains_CPU, NB_LAYERS * sizeof(int), 0, cudaMemcpyHostToDevice));

	// C is C matrix and beta, levels: list of levels, Ver contains the list of vertices,
	// R contains the isometry matrices, q contains the translation values
	float *C, *Ccpu, *levels, *levelscpu, *Ver, *Vercpu, *R, *q;
	Num_t *num, *numcpu; // contains the true number of vertices and levels

	std::string pathWB = "weights.txt";
	std::vector<float> WB_data;
	if (!readArrayFromFile(sizeWB, pathWB, WB_data))
	{
		return 1;
	}

	testCUDA(cudaMalloc(&C, sizeCB * nb_subpolytopes * sizeof(float)));
	Ccpu = (float *)malloc(sizeCB * nb_subpolytopes * sizeof(float));

	testCUDA(cudaMalloc(&levels, 2 * nb_vertices * nb_subpolytopes * sizeof(float))); // twice the size to be able to have a sorted list
	levelscpu = (float *)malloc(nb_vertices * nb_subpolytopes * sizeof(float));
	testCUDA(cudaMalloc(&Ver, nb_vertices * nb_subpolytopes * domains_CPU[0] * sizeof(float)));
	Vercpu = (float *)malloc(nb_vertices * nb_subpolytopes * domains_CPU[0] * sizeof(float));

	// TODO how those these work
	int siV = (domains_CPU[0] == 2) ? (3 + 2) : (4 + 3 + 2); // binding index for volume V
	int siVD = (domains_CPU[0] == 2) ? (2) : (3 + 2);		 // binding index for volume V TODO what are the differnces between them
	int siR = (domains_CPU[0] == 2) ? (4) : (9 + 4);		 // binding index for volume R
	int siRD = (domains_CPU[0] == 2) ? 0 : (4);				 // binding index for volume R

	if (domains_CPU[0] > 3)
		std::cout << "The code is not made for dimension > 3\n";

	// TODO instead of doing 4 layers do 3 then add it
	testCUDA(cudaMalloc(&R, siR * nb_subpolytopes * sizeof(float)));
	testCUDA(cudaMalloc(&num, nb_subpolytopes * sizeof(Num_t)));
	numcpu = (Num_t *)malloc(nb_subpolytopes * sizeof(Num_t));

	testCUDA(cudaMalloc(&q, nb_subpolytopes * domains_CPU[0] * sizeof(float)));

	float *WBGPU; // Coefficents of matrices and bias vectors
	testCUDA(cudaMalloc(&WBGPU, sizeWB * sizeof(float)));
	testCUDA(cudaMemcpy(WBGPU, WB_data.data(), sizeWB * sizeof(float), cudaMemcpyHostToDevice));

	int low = 0;			  // the starting index
	int up = nb_subpolytopes; // the ending index
	int nbN = 16;			  // Number of neurones TODO  we should have 15 no ? in the 3d we should have 16

	testCUDA(cudaMemset(C, 0, nb_subpolytopes * sizeCB * sizeof(float) ));

	float timer;
	cudaEvent_t start, stop;   // GPU timer instructions
	cudaEventCreate(&start);   // GPU timer instructions
	cudaEventCreate(&stop);	   // GPU timer instructions
	cudaEventRecord(start, 0); // GPU timer instructions

	size_t currentLimit;
	cudaDeviceGetLimit(&currentLimit, cudaLimitStackSize);
	printf("Current CUDA stack size: %zu bytes\n", currentLimit);

	size_t NcurrentLimit = 64 * currentLimit;
	cudaDeviceSetLimit(cudaLimitStackSize, NcurrentLimit);
	cudaDeviceGetLimit(&currentLimit, cudaLimitStackSize);
	printf("Current CUDA stack size: %zu bytes\n", currentLimit);

	testCUDA(cudaMemset(levels, 0, 2 * nb_vertices * nb_subpolytopes * sizeof(float)));
	testCUDA(cudaMemset(Ver, 0, nb_vertices * nb_subpolytopes * domains_CPU[0] * sizeof(float)));
	testCUDA(cudaMemset(num, 0, 2 * nb_subpolytopes * sizeof(float)));

	//        TODO     TODO      why + 1
	// Part_k <<<16 * 8, 16 * 2 * (dCPU[0] + 1), (sizeWB + 16 * 2 * max(nbN * 4, 15 * dCPU[0])) * sizeof(float) >>>
	// levL_k <<<16 * 8, 16 * 2 * (dCPU[0] + 1), (sizeWB + 16 * 2 * max(nbN * 4, 15 * dCPU[0])) * sizeof(float) >>>
	int min_nb_vertices_by_polytope = domains_CPU[0] + 1; // d0 + 1 = minimum number of vertices for a d0 dimension polytope
	
	partitionPolytope_k<<<nb_subpolytopes/256/2, 256>>>
			(domains_GPU, WBGPU, NB_LAYERS, nb_subpolytopes, 0, nb_subpolytopes, 
				C, C, C, sizeCB, m0); // TODO initialise CDBd in the CB directly 
	// partialPart_k <<<16 * 8, 16 * 2 * min_nb_vertices_by_polytope, (sizeWB + 16 * 2 * max(nbN * 4, 15 * domains_CPU[0])) * sizeof(float) >>>
	// 		(WBGPU, C, levels, 2 * domains_CPU[0], sizeWB,
	// 			low, up, R, q, Ver, num, nbN, 4,
	// 			siV, siVD, siR, siRD, min_max, magic_values);

	testCUDA(cudaDeviceSynchronize()); // TODO not sure if i need this sync

	// // TODO maybe there is a better way to calculate and create a new array
	// testCUDA(cudaMemcpy(numcpu, num, nb_subpolytopes * sizeof(Num_t), cudaMemcpyDeviceToHost));
	// int *non_empty_num_indices, *non_empty_num_indicesCPU, counter_non_empty_num = 0;
	// non_empty_num_indicesCPU = (int *)malloc(nb_subpolytopes * sizeof(int));

	// // Non empty num calculation
	// for (int i = 0; i < nb_subpolytopes; i++)
	// {
	// 	if (numcpu[i].ver != 0)
	// 	{
	// 		non_empty_num_indicesCPU[counter_non_empty_num++] = i;
	// 		// std::cout << i << std::endl;
	// 	}
	// }

	// testCUDA(cudaMalloc(&non_empty_num_indices, counter_non_empty_num * sizeof(int)));
	// testCUDA(cudaMemcpy(non_empty_num_indices, non_empty_num_indicesCPU,
	// 					counter_non_empty_num * sizeof(int), cudaMemcpyHostToDevice));

	// // levL_k <<<counter_non_empty_num, 16 * 2 * min_nb_vertices_by_polytope, (sizeWB + 16 * 2 * max(nbN * 4, 15 * domains_CPU[0])) * sizeof(float) >>>
	// // 		(WBGPU, C, levels, 2 * domains_CPU[0], sizeWB,
	// // 			low, up, R, q, Ver, num, nbN, 4,
	// // 			siV, siVD, siR, siRD, min_max, magic_values, non_empty_num_indices);

	// testCUDA(cudaDeviceSynchronize());

	// testCUDA(cudaMemcpy(Ccpu, C, nb_subpolytopes * sizeCB * sizeof(float), cudaMemcpyDeviceToHost));
	// testCUDA(cudaMemcpy(levelscpu, levels, nb_vertices * nb_subpolytopes * sizeof(float), cudaMemcpyDeviceToHost));
	// testCUDA(cudaMemcpy(Vercpu, Ver, nb_vertices * nb_subpolytopes * domains_CPU[0] * sizeof(float), cudaMemcpyDeviceToHost));
	// testCUDA(cudaMemcpy(numcpu, num, 2 * nb_subpolytopes * sizeof(float), cudaMemcpyDeviceToHost));

	// printf("The computed Minimum level %f\n", (float)0.0000001f * min_max->min);
	// printf("The computed Maximum level %f\n", (float)0.0000001f * min_max->max);
	// printf("Number of levels %i\n", min_max->final_max);
	// printf("With notation (index of pol, number of vertices, number of levels), the non-empty polytopes are:\n");
	// int count = 0;
	// for (int k = 0; k < nb_subpolytopes; k++)
	// {
	// 	if (numcpu[k].ver > 0)
	// 	{
	// 		printf("(%d, %d, %d), ", k, numcpu[k].ver, numcpu[k].lvl);
	// 		count++;
	// 	}
	// }
	// printf("\n");
	// printf("The number of non-empty polytopes: %d\n", count);

	// cudaEventRecord(stop, 0);		   // GPU timer instructions
	// cudaEventSynchronize(stop);		   // GPU timer instructions
	// cudaEventElapsedTime(&timer,	   // GPU timer instructions
	// 					 start, stop); // GPU timer instructions
	// cudaEventDestroy(start);		   // GPU timer instructions
	// cudaEventDestroy(stop);			   // GPU timer instructions

	// printf("Execution time %f ms\n", timer);

	// testCUDA(cudaFree(min_max));
	// testCUDA(cudaFree(C));
	// free(Ccpu);
	// testCUDA(cudaFree(levels));
	// free(levelscpu);
	// testCUDA(cudaFree(Ver));
	// free(Vercpu);
	// testCUDA(cudaFree(R));
	// testCUDA(cudaFree(q));
	// testCUDA(cudaFree(num));
	// free(numcpu);
	// testCUDA(cudaFree(WBGPU));
	// testCUDA(cudaFree(magic_values));
	// free(non_empty_num_indicesCPU);

	return 0;
}