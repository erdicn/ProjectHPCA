/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code
the name of the author above.
***************************************************************/
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <curand_kernel.h>

#define MaxDepth 16
#define Sizedld (MaxDepth + 5)


__device__ float WG[16777216];
__constant__ int dld[Sizedld];


// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

/****************************************************************************************************
Algorithm 4.3 in Global Maximization of piecewise linear Feedforward Neural Network
****************************************************************************************************/
// Ver contains the list of vertices
// C contains both the C matrix and beta
// R contains the isometry matrices
// q contains the translation values
// num contains the number of vertices
// Wl shared memory
// tidx thread index in a group of threads working together
// rnm1 number of rows
// eps by default equal to zero
// K = 2^{cell[log2(rnm1)]}, the closest power of two
// flag test whether the computations have to be done
// gbx the polytope index, in case we want to debug
// i and tr indices of the recursionin dimension 3, in case we want to debug
__device__ void oneDimVertices(float* Ver, float* C, float* R, float* q, int *num, float* Wl, 
								int tidx, int rnm1, float eps, int K, int *flag, int gbx, int i, int* tr) {

	float *loc1, *loc2, *MLB, *MGB, *v;
	int j, k, f0, f00, f2, *f1, *Imgb, *Imlb;
	float loc3;
	k = K;

	float* V;

	V = Ver + gbx * dld[MaxDepth + 2] * dld[0]; // When the shared memory is not sufficient

	v = Wl;
	loc2 = v + k;

	loc1 = WG + gbx * 512;
	MLB = loc1 + k;
	MGB = MLB + k;
	f1 = (int*)(MGB + k);
	Imgb = f1 + k;
	Imlb = Imgb + k;


	for (j = tidx; (j < K) && (*flag); j += dld[0] + 1) {
		f0 = j * (j < rnm1);
		loc1[j] = C[f0 * 2 + 1];
		loc2[j] = C[f0 * 2];
		f1[j] = ((loc2[j] > eps) || (loc2[j] < -eps) || (loc1[j] >= 0.0f));
		loc1[j] /= loc2[j];
		MLB[j] = 0.0f;
		MGB[j] = 0.0f;
		Imgb[j] = -1;
		Imlb[j] = -1;
	}
	__syncthreads();

	////////////////////// f0 contains the exit condition ////////////////////////////////////////
	for (j = tidx + dld[0] + 1; j < K; j += (dld[0] + 1)) {
		f1[tidx] *= f1[j];
	}
	__syncthreads();
	for (j = dld[0]; (j > 0) && (tidx == 0); j--) {
		f1[0] *= f1[j];
	}
	__syncthreads();

	f0 = (*flag)*f1[0];


	for (j = tidx; j < K; j += dld[0] + 1) {
		if ((loc2[j] > 0.0f) && f0) {
			MGB[j] = loc1[j];
			Imgb[j] = j;
		}
		if ((loc2[j] < 0.0f) && f0) {
			MLB[j] = loc1[j];
			Imlb[j] = j;
		}
	}

	__syncthreads();

	k = (K + dld[0]) / (dld[0] + 1);
	k *= (dld[0] + 1);

	for (j = tidx + dld[0] + 1; j < k; j += dld[0] + 1) {
		if (j < K) {
			if ((loc2[j] > 0.0f) && f0) {
				MGB[tidx] = (Imgb[tidx] < 0) * MGB[j] +
					(Imgb[tidx] >= 0) * fminf(MGB[tidx], MGB[j]);
				Imgb[tidx] = (MGB[tidx] == MGB[j]) * j +
					(MGB[tidx] < MGB[j]) * Imgb[tidx];
			}
			if ((loc2[j] < 0.0f) && f0) {
				MLB[tidx] = (Imlb[tidx] < 0) * MLB[j] +
					(Imlb[tidx] >= 0) * fmaxf(MLB[tidx], MLB[j]);
				Imlb[tidx] = (MLB[tidx] == MLB[j]) * j +
					(MLB[tidx] > MLB[j]) * Imlb[tidx];
			}
		}
		__syncthreads();
	}

	if (f0) {
		for (j = dld[0]; (j > 0) && (tidx == 0); j--) {
			if (Imgb[j] > -1) {
				MGB[0] = (Imgb[0] < 0) * MGB[j] + (Imgb[0] >= 0) * fminf(MGB[0], MGB[j]);
				Imgb[0] = (MGB[0] == MGB[j]) * Imgb[j] + (MGB[0] < MGB[j]) * Imgb[0];
			}
			if (Imlb[j] > -1) {
				MLB[0] = (Imlb[0] < 0) * MLB[j] + (Imlb[0] >= 0) * fmaxf(MLB[0], MLB[j]);
				Imlb[0] = (MLB[0] == MLB[j]) * Imlb[j] + (MLB[0] > MLB[j]) * Imlb[0];
			}
		}
	}

	__syncthreads();

	f00 = f0;

	///////////////////////////// First possible vertex ////////////////////////////////////////
	f0 = f00 * (MGB[0] >= MLB[0]);

	f2 = 0;
	for (j = 0; j < dld[0]; j++) {
		if ((j == 0) && f0 && (tidx == 0)) {
			v[j] = MGB[0];
		}
		if ((j > 0) && f0 && (tidx == 0)) {
			v[j] = q[(j - 1)];
		}
		__syncthreads();

		loc2[tidx] = 0.0f;
		if ((tidx < j + 1) && f0 && (j > 0)) {
			for (k = 0; k < j + 1; k++) {
				loc2[tidx] += R[f2 + tidx * (j + 1) + k] * v[k];
			}
		}
		__syncthreads();
		
		if ((tidx < j + 1) && f0 && (j > 0)) {
			v[tidx] = loc2[tidx];
		}
		__syncthreads();

		f2 += (j > 0) * (j + 1) * (j + 1);
	}


	///////////////////////////// Is this vertex already in the list? ////////////////////////////////

	f1[tidx] = 1;
	
	for (j = tidx; (j < num[0]) && f0; j += dld[0] + 1) {
		loc3 = 0.0f;
		for (k = 0; k < dld[0]; k++) {
			loc3 += (v[k] - V[j * dld[0] + k]) * (v[k] - V[j * dld[0] + k]);
		}
		f1[tidx] *= (sqrtf(loc3) > eps);
	}
	__syncthreads();

	if (f0) {
		for (j = dld[0]; (j > 0) && (tidx == 0); j--) {
			f1[tidx] *= f1[j];
		}
	}

	__syncthreads();

	if (f0 * f1[0]) {
		if (tidx < dld[0]) {
			V[num[0] * dld[0] + tidx] = v[tidx];
		}
		if (tidx == 0) {
			num[0]++;
		}
	}
	__syncthreads();

	///////////////////////////// Second possible vertex ////////////////////////////////////////

	f2 = 0;
	for (j = 0; j < dld[0]; j++) {
		if ((j == 0) && f0 && (tidx == 0)) {
			v[j] = MLB[0];
		}
		if ((j > 0) && f0 && (tidx == 0)) {
			v[j] = q[(j - 1)];
		}
		__syncthreads();

		loc2[tidx] = 0.0f;
		if ((tidx < j + 1) && f0 && (j > 0)) {
			for (k = 0; k < j + 1; k++) {
				loc2[tidx] += R[f2 + tidx * (j + 1) + k] * v[k];
			}
		}
		__syncthreads();

		if ((tidx < j + 1) && f0 && (j > 0)) {
			v[tidx] = loc2[tidx];
		}
		__syncthreads();

		f2 += (j > 0) * (j + 1) * (j + 1);
	}


	///////////////////////////// Is this vertex already in the list? ////////////////////////////////

	f1[tidx] = 1;

	for (j = tidx; (j < num[0]) && f0; j += dld[0] + 1) {
		loc3 = 0.0f;
		for (k = 0; k < dld[0]; k++) {
			loc3 += (v[k] - V[j * dld[0] + k]) * (v[k] - V[j * dld[0] + k]);
		}
		f1[tidx] *= (sqrtf(loc3) > eps);
	}

	__syncthreads();


	if (f0) {
		for (j = dld[0]; (j > 0) && (tidx == 0); j--) {
			f1[tidx] *= f1[j];
		}
	}

	__syncthreads();

	if (f0 * f1[0]) {
		if (tidx < dld[0]) {
			V[num[0] * dld[0] + tidx] = v[tidx];
		}
		if (tidx == 0) {
			num[0]++;
		}
	}
	__syncthreads();
}

/****************************************************************************************************
Computes the square of the euclidean norm 
****************************************************************************************************/
// C contains both the C matrix
// Wl shared memory
// tidx thread index in a group of threads working together
// r is the dimension
// K = 2^{cell[log2(r)]}, the closest power of two
__device__ float normCri(float* C, int r, float* Wl, int tidx, int K) {

	float loc;
	int k = K; // the closest power of two////////////////////////////////////////////////////////////

	for (int j = tidx; j < k; j += dld[0] + 1) {
		Wl[j] = 1.0f * (tidx < r);
	}
	Wl[tidx] *= C[tidx] * C[tidx];
	__syncthreads();

	k /= 2;
		while (k > 0) {
			if (tidx < k) {
				Wl[tidx] += Wl[tidx + k];
			}
			k /= 2;
			__syncthreads();
		}
	loc = Wl[0];

	return loc;
}

/****************************************************************************************************
Manages colinearity and compute the matrix R as in Lemma 3.2 of "Polynomial Distribution
of Feedforward Neural Network Output"
****************************************************************************************************/
// i the row index given as an input to Decomposition function
// r the dimension given as an input to Decomposition function
// C contains both the C matrix and beta
// R contains the isometry matrices
// Wl shared memory
// tidx thread index in a group of threads working together
// eps by default equal to zero
// flag test whether the computations have to be done
// norm2 the square of the euclidean norm needed to build orthonormal vectors in R 
// k = 2^{cell[log2(r)]}, the closest power of two
// gbx the polytope index, in case we want to debug
// tr index of the recursion, in case we want to debug
__device__ void DecompVer(int i, int r, float* C, float* R, float* Wl, int tidx,
						  float eps, int* flag, float* norm2, int k, int gbx, int* tr) {
	float Ci, Cj;
	int q, j, l;

	//////////////////////////////////// positive colinearity check done by tidx 0 ////////////////////////////////
	for (j = 0; j < i; j++) {
		Wl[tidx] = 0.0f;
		Ci = C[tidx + i * (r + 1)];
		Cj = C[tidx + j * (r + 1)];

		if (*flag) {
			q = ((Ci > eps) || (Ci < -eps));
			if (q) {
				Wl[tidx] = (Cj / Ci); // only positive colinearity counts 
			}
			else {
				Wl[tidx] = - 1.0f; // only positive colinearity counts 
			}
		}
		__syncthreads();


		if ((tidx == 0) && (*flag)) {
			Ci = -1.0f;
			q = 0;
			for (l = 0; l < r + 1; l++) {
				Cj = Wl[l];
				Ci = Cj * (Cj > eps) + Ci * (Cj <= eps);
				q += (Cj > eps);
			}
			Cj = 1;
			for (l = 0; (l < r + 1) && (q > 1); l++) {
				Cj *= ((Ci < Wl[l] + eps) && (Ci > Wl[l] - eps)) * (Wl[l] > -eps) +
					(Wl[l] < -eps);
			}
			Wl[dld[0] + 1] = (q < 2) + (q > 1) * (1 - Cj);
		}
		__syncthreads();

		*flag *= Wl[dld[0] + 1];
	}

	j = 0;
	int p = 0;

	//////////////////////////////////// Computation of matrix R using Gram Schmidt ////////////////////////////////
	while ((p < r)) {
		for (l = tidx; l < k && (*flag); l += (r + 1)) {
			Wl[l] = 1.0f * (tidx == p) - (C[tidx + i * (r + 1)] * C[p + i * (r + 1)] / *norm2) * (l < r);
			Wl[l + k] = Wl[l];
		}
		if (tidx < r) {
			Wl[2 * k + j * r + tidx] = Wl[tidx] * (*flag);
		}
		__syncthreads();

		for (q = 0; q < p; q++) {
			if (tidx < k) {
				Wl[tidx] *= Wl[2 * k + q * r + tidx] * (q < j);
			}
			__syncthreads();
			l = k / 2;
			while (l > 0) {
				if (tidx < l) {
					Wl[tidx] += Wl[tidx + l];
				}
				l /= 2;
				__syncthreads();
			}
			if (tidx < k) {
				Wl[tidx + k] -= Wl[0] * Wl[2 * k + q * r + tidx];
			}
			for (l = tidx; l < k && (*flag); l += (r + 1)) {
				Wl[l] = Wl[2 * k + j * r + tidx] * (*flag) * (l < r);
			}
			__syncthreads();
		}


		if (tidx < k) {
			Wl[tidx] = Wl[tidx + k] * Wl[tidx + k] * (tidx < r);
		}

		__syncthreads();
		l = k / 2;
		while (l > 0) {
			if (tidx < l) {
				Wl[tidx] += Wl[tidx + l];
			}
			l /= 2;
			__syncthreads();
		}

		if (Wl[0] > eps) {
			if (tidx < r) {
				Wl[2 * k + j * r + tidx] = Wl[tidx + k] / sqrtf(Wl[0]);
			}
			j++;
		}
		__syncthreads();
		p++;
	}

	if (tidx < r) {
		Wl[2 * k + (r - 1) * r + tidx] = C[tidx + i * (r + 1)] / sqrtf(*norm2);
	}
	__syncthreads();

	for (j = 0; j < r; j++) {
		if (tidx < r) {
			R[tidx * r + j] = Wl[2 * k + j * r + tidx];
		}
	}
	__syncthreads();
}

/****************************************************************************************************
Computation of new C and new beta as in Lemma 3.2 of "Polynomial Distribution
of Feedforward Neural Network Output"
****************************************************************************************************/
// i the row index given as an input to Decomposition function
// r the dimension given as an input to Decomposition function
// rnm1 number of rows
// C contains both the C matrix and beta
// R contains the isometry matrices
// q contains the translation values
// Wl shared memory
// tidx thread index in a group of threads working together
// eps by default equal to zero
// flag test whether the computations have to be done
// norm2 the square of the euclidean norm needed in Lemma 3.3 
// gbx the polytope index, in case we want to debug
// tr index of the recursion, in case we want to debug
__device__ void CBqVer(int i, int r, int rnm1, float* C, float* R, float* q, float* Wl,
	int tidx, float eps, int* flag, float* norm2, int gbx, int *tr) {

	int j, l;
	float* Cn;
	Cn = C - r * (rnm1 - 1);


	for (j = 0; (j < i) && (*flag); j++) {
		if (tidx < r) {
			Cn[j * r + tidx] = 0.0f;
			for (l = 0; l < r; l++) {
				Cn[j * r + tidx] += C[j * (r + 1) + l] * R[l * r + tidx];
			}
			if (tidx == r - 1) {
				Cn[j * r + tidx] = C[j * (r + 1) + tidx + 1] - (Cn[j * r + tidx] * C[i * (r + 1) + tidx + 1]) / sqrtf(*norm2);
			}
		}
	}

	for (j = i + 1; (j < rnm1) && (*flag); j++) {
		if (tidx < r) {
			Cn[(j - 1) * r + tidx] = 0.0f;
			for (l = 0; l < r; l++) {
				Cn[(j - 1) * r + tidx] += C[j * (r + 1) + l] * R[l * r + tidx];
			}
			if (tidx == r - 1) {
				Cn[(j - 1) * r + tidx] = C[j * (r + 1) + tidx + 1] - (Cn[(j - 1) * r + tidx] * C[i * (r + 1) + tidx + 1]) / sqrtf(*norm2);
			}
		}
	}


	if (tidx == 0) {
		if (*flag) {
			q[r - 2] = C[i * (r + 1) + r] / sqrtf(*norm2);
		}
	}
}


/****************************************************************************************************
Algorithm 4.2 in "Global Maximization of piecewise linear Feedforward Neural Network"
****************************************************************************************************/
// Ver contains the list of vertices
// C contains both the C matrix and beta
// R contains the isometry matrices
// q contains the translation values
// r the original dimension given as an input
// rnm1 number of rows
// WB shared memory
// tidx thread index in a group of threads working together
// eps by default equal to zero
// flag tests whether the computations have to be done
// num contains the number of vertices
// gbx the polytope index, in case we want to debug
// i and tr indices of the recursion in dimension 3, in case we want to debug
__device__ void Vertices(float* Ver, float* C, float* R, float* q, int r, int rnm1,
	float* WB, int tidx, float eps, int* flag, int* num, int gbx, int i, int *tr) {

	int k, f1, f2;

	if (r == 1) {
		f1 = (rnm1 > 2) * (rnm1 != 4); // When f1 is not true, it means that rnm1 is a power of two
		f2 = (rnm1 / 2 < 2); // The closest power of two is 4 otherwise 8
		k = f1 * (f2 * 4 + (1 - f2) * 8) + (1 - f1) * rnm1; // The closest power of two when rnm1 <= 8
		f1 = (rnm1 > 8) && (rnm1 <= 16);
		f2 = (rnm1 > 16) && (rnm1 <= 32);
		k = f1 * 16 + f2 * 32 + (1 - f1) * (1 - f2) * k;
		f1 = (rnm1 > 32) && (rnm1 <= 64);
		k = f1 * 64 + (1 - f1) * k;
		oneDimVertices(Ver, C, R, q, num, WB, tidx, rnm1, eps, k, flag, gbx, i, tr);
	}
	else {
		f1 = (r > 2) * (r != 4); // When f1 is not true, it means that r is a power of two
		f2 = (r / 2 < 2); // The closest power of two is 4 otherwise 8
		k = f1 * (f2 * 4 + (1 - f2) * 8) + (1 - f1) * r; // The closest power of two when r <= 8 
		for (int i = 0; i < rnm1; i++) {
			if (r == 3) {
				*tr = i;
			}
			float norm2 = normCri(C + i * (r + 1), r, WB, tidx, k);
			f1 = (*flag) * (sqrtf(norm2) > eps);
			__syncthreads();
			DecompVer(i, r, C, R, WB, tidx, eps, &f1, &norm2, k, gbx, tr);
			if (f1) {
				CBqVer(i, r, rnm1, C, R, q, WB, tidx, eps, flag, &norm2, gbx, tr);
			}
			__syncthreads();
			Vertices(Ver, C - r * (rnm1 - 1), R - (r - 1) * (r - 1) * (r > 2), q,
						r - 1, rnm1 - 1, WB, tidx, eps, &f1, num, gbx, i, tr);
		}
	}
}


/****************************************************************************************************
Algorithm 4.2 in Global Maximization of piecewise linear Feedforward Neural Network
****************************************************************************************************/
// Ver contains the list of vertices
// C contains both the C matrix and beta
// R contains the isometry matrices
// q contains the translation values
// r the original dimension given as an input
// rnm1 number of rows
// WB shared memory
// tidx thread index in a group of threads working together
// eps by default equal to zero
// flag tests whether the computations have to be done
// num contains the number of vertices as well as the number of levels
// gbx the polytope index, in case we want to debug
// i and tr indices of the recursion in dimension 3, in case we want to debug
__device__ void levL(float* Ver, float* C, float* LL, int* num, float* Wl,
						int tidx, float eps, int r, int gbx) {

	int K, f1, f2, i, j, k, del, numTemp;
	int* Wl2;
	float* V; 
	del = dld[MaxDepth + 2];
	V = Ver + gbx * del * dld[0];

	// The closest power of two when r <= 8 
	K = (r == 1) + 2 * (r == 2) + 4 * (r == 3) + 4 * (r == 4) + 8 * (r > 5);

	//Wl2 = (int*)WG + gbx * 512; // alternative to the use of shared memory
	Wl2 = (int*)Wl + K;

	for (i = 0; i < del; i++) {
		numTemp = num[1];
		f1 = (num[0] > i);

		for (j = tidx; j < K && f1; j += (r + 1)) {
			f2 = (j < r);
			Wl[j] = f2 * C[j * f2] * V[i * r + j * f2] - (j == 0) * C[r];
		}
		__syncthreads();
		k = K / 2;
		while (k > 0) {
			if (tidx < k && f1) {
				Wl[tidx] += Wl[tidx + k];
			}
			k /= 2;
			__syncthreads();
		}
		if ((numTemp == 0) && f1 && (tidx == 0)) {
			LL[0] = Wl[0];
			num[1] += 1;
		}
		if ((numTemp == 1) && f1 && (tidx == 0)) {
			k = (fabsf(LL[((i + 1) % 2) * del] - Wl[0]) > eps);
			if (k) {
				j = (LL[((i + 1) % 2) * del] < Wl[0]);
				LL[(i % 2) * del] = j * LL[((i + 1) % 2) * del] + (1 - j) * Wl[0];
				LL[(i % 2) * del + 1] = j * Wl[0] + (1 - j) * LL[((i + 1) % 2) * del];
				num[1] += 1;
			}
			else { 
				LL[(i % 2) * del] = LL[((i + 1) % 2) * del];
			}
		}
		if ((numTemp > 1) && f1) {
			for (k = tidx; k < K; k += (r + 1)) {
				Wl2[k] = 1;
			}
		}
		__syncthreads();
		if ((numTemp > 1) && f1) {
			for (k = tidx; (k < numTemp); k += (r + 1)) {
				f2 = (k < r);
				Wl2[tidx] *= (fabsf(Wl[0] - LL[((i + 1) % 2) * del + k]) > eps);
			}
		}
		__syncthreads();
		k = K / 2;
		while (k > 0) {
			if (tidx < k && f1 && (numTemp > 1)) {
				Wl2[tidx] *= Wl2[tidx + k];
			}
			k /= 2;
			__syncthreads();
		}
		if ((numTemp > 1) && f1) {
			if (1 - Wl2[0]) {
				for (k = tidx; (k < numTemp); k += (r + 1)) {
					LL[(i % 2) * del + k] = LL[((i + 1) % 2) * del + k];
				}
			}
			else {
				for (k = tidx; (k < numTemp + 1); k += (r + 1)) {
					if ((k > 0) && (k < numTemp)) {
						j = (LL[((i + 1) % 2) * del + k] < Wl[0]);
						f2 = (LL[((i + 1) % 2) * del + k - 1] < Wl[0]);
						LL[(i % 2) * del + k] = LL[((i + 1) % 2) * del + k] * j +
							(1 - j) * f2 * Wl[0] +
							(1 - f2) * LL[((i + 1) % 2) * del + k - 1];
					}
					else {
						if (k == 0) {
							j = (LL[((i + 1) % 2) * del] < Wl[0]);
							LL[(i % 2) * del] = LL[((i + 1) % 2) * del] * j +
								(1 - j) * Wl[0];
						}
						else {
							j = (LL[((i + 1) % 2) * del + numTemp - 1] < Wl[0]);
							LL[(i % 2) * del + numTemp] = Wl[0] * j +
								(1 - j) * LL[((i + 1) % 2) * del + numTemp - 1];
						}

					}
				}
				if (tidx == 0) {
					num[1] += 1;
				}
			}
		}
		__syncthreads();
	}

	for (k = tidx; k < num[1]; k += (1 + r)) {
		LL[(num[0] % 2) * del + k] = LL[((num[0] - 1) % 2) * del + k];
	}
}


/****************************************************************************************************
Partitioning algorithm as in  "Polynomial Distribution of Feedforward Neural Network Output"
****************************************************************************************************/
// WlBl contains coefficients of matrices and bias vectors
// C contains both the C matrix and beta
// Contains the values of levels
// LL list of levels
// m0 number of rows for the definition of the input compact D_0
// size of data needed (coefficients of matrices and bias vectors) to define the NN
// low the starting index, here = 0 by default
// up the ending index, here = 4096 by default 
// V contains the list of volume coefficients 
// R contains the isometry matrices
// q contains the translation values
// Ver contains the list of vertices
// num contains the number of vertices as well as the number of 
// nbN number of neurones
// L number of layers
// siV binding index for volume V
// siVD binding index for volume V
// siR binding index for volume R
// siRD binding index for volume R
// MinMax[0] and MinMax[1] are respectively the minimal and the maximal values of levels   
__global__ void Part_k(float* WlBl, float* C, float* LL,
						int m0, int size, int low, int up, float* R, 
						float* q, float* Ver, int *num, int nbN, int L, 
						int siV, int siVD, int siR, int siRD, int *MinMax) {

	int i, j, l, dMax;
	float loc;
	// The maximum number of involved threads per configuration (s_1,...,s_{L-1})
	dMax = dld[0] + 1;			 // number of needed threads = d_0 + 1

	extern __shared__ float WB[];
	float *sl, *Wl;
	sl = WB + size;
	Wl = sl + nbN*(blockDim.x / dMax);
	// nbN is the total number of neurons and thus the number of ones and -ones

	for (i = threadIdx.x; i < size; i += blockDim.x) {
		WB[i] = WlBl[i];
	}

	int Qt = threadIdx.x / dMax;
	int tidx = threadIdx.x - Qt * dMax;
	//int gbx = low + Qt + blockIdx.x * (blockDim.x / dMax);
	int gbx = Qt + blockIdx.x * (blockDim.x / dMax);
	int deltaC, DeltaC, nbVer, val, lim, dl, dlm1;
	deltaC = dld[MaxDepth]; // Values should be stored at the end Algo (4.4)
	DeltaC = dld[MaxDepth+1]; // The total size needed for each configuration s
	nbVer = dld[MaxDepth + 2]; // The maximum number of vertices in each sub-polytope


	// Translate with L != 0 as long as Part_k has to be executed many times
	if (low + gbx < up) {
		for (i = tidx; i < nbN; i += dMax) {
			val = 2 * ((low + gbx >> i) % 2) - 1;
			sl[i + Qt * nbN] = 1.0f*(val>=0) - 0.01f*(val<0);
		}
	}
	
	// Initialization that depends on the definition of the compact set D
	// As this is the same to all sub-polytops, it has to be computed once, condition (low == 0)
	if (low == 0) {
		int index;
		for (i = 0; i < m0; i++) {
			index = i / 2;
			C[gbx * DeltaC + deltaC + tidx + i * dMax] = 1.0f * (i == 2 * tidx) * (tidx < (dMax - 1)) -
														1.0f * (i == (2 * tidx + 1)) * (tidx < (dMax - 1)) +
														(2 * index == i) * (tidx == (dMax - 1));
		}
	}
	__syncthreads();

	// This part is related to the computation of C_{1}
	lim = m0;
	dl = dld[1];
	for (i = lim; i < lim + dl; i++) { // C starts at the right place at + m0*dMax
		val = (sl[i-lim + Qt * nbN]>=0) - (sl[i - lim + Qt * nbN] < 0);
		C[gbx * DeltaC + deltaC + tidx + i * dMax] = -(tidx < (dMax - 1)) * val * WB[tidx + (i - lim) * dMax] +
														(tidx == (dMax - 1)) * val * WB[tidx + (i - lim) * dMax];
	}
	
	lim = 0;
	int dlmDelta = 0;
	// This part is related to the computation of C_{l}
	for (l = 2; l < L + 1; l++) {
		lim += dld[l - 1] * (dld[l - 2] + 1);
		dl = dld[l];
		dlm1 = dld[l - 1];
		dlmDelta += dlm1;
		for(j=0; j<dl; j++){
			if (l < L) {
				val = (sl[j + Qt * nbN + dlmDelta] >= 0) - (sl[j + Qt * nbN + dlmDelta] < 0);
			}
			else { val = -1; }
			// Starting with diag(s_{l})*W_{l}*diag(a(s_{l-1})) row by row
			for (i = tidx; i < dlm1; i += dMax) {
				Wl[i + Qt * dlm1] = val * WB[lim + j * (dlm1 + 1) + i] * sl[i + dlmDelta - dlm1 + Qt * nbN];
			}
			__syncthreads();
			
			loc = 0.0f;
			for (i = 0; i < dlm1; i++) {
				loc += Wl[i + Qt * dlm1] * C[gbx * DeltaC + deltaC + tidx + (i + m0 + dlmDelta - dlm1) * dMax];
			}
			C[gbx * DeltaC + deltaC + tidx + (j + m0 + dlmDelta) * dMax] = loc + (tidx == (dMax - 1)) * val * WB[lim + j * (dlm1 + 1) + dlm1];
			__syncthreads();
		}
	}

	int flag = 1; // if 1 computations performed, when flag switches to 0 the threads are only involved in synchronization 

	int r = dMax - 1; // r = d0

	float eps = 0.0f; // replaces the true zero : targetted precision

	int rnm1 = m0 + dlmDelta + dl;	 // This is what is called  nrow(C[r-1]) - 1 for r = d0 in Algo 4.4 or m in Algo 4.1

	__syncthreads();

	int tr = 100;

	Vertices(Ver, C + gbx * DeltaC + deltaC, R + gbx * siR + siRD,
		q + gbx * r, r, rnm1 - 1, WB + Qt * 4 * nbN, tidx, eps, &flag,
		num + gbx * 2, gbx + low, 0, &tr);


	levL(Ver, C + gbx * DeltaC + deltaC + (m0 + 12) * dMax, LL + gbx * 2 * nbVer,
		num + gbx * 2, WB + Qt * 4 * nbN, tidx, 0, dMax - 1, gbx);

	int pkint;

	__syncthreads();

	for (l = 0; l < nbVer; l++) {
		flag = (l < num[gbx * 2 + 1]);
		if ((tidx == 0) && flag) {
			pkint = (int)10000000*LL[gbx * 2 * nbVer + l];
			atomicMin(MinMax, pkint);
			atomicMax(MinMax + 1, pkint);
		}
	}

	if ((tidx == 0)) {
		atomicMax(MinMax + 2, num[gbx * 2 + 1]);
	}

}


int main(void) {

	int dCPU[Sizedld];
	dCPU[0] = 2; // d_0
	dCPU[1] = 4; // d_1
	dCPU[2] = 4; // d_2
	dCPU[3] = 4; // d_3
	dCPU[4] = 1; // d_4

	int sizeWB = (dCPU[0] + 1) * dCPU[1] +
				(dCPU[1] + 1) * dCPU[2] +
				(dCPU[2] + 1) * dCPU[3] +
				(dCPU[3] + 1) * dCPU[4];

	int* MinMax;
	cudaMallocManaged(&MinMax, 3*sizeof(int));
	MinMax[0] = 5000000;
	MinMax[1] = -5000000;
	MinMax[2] = 0;

	int nop = 16 * 16 * 16; // Number of subpolytopes

	int deltaSize = (2 * dCPU[0] + dCPU[1] + dCPU[2] + dCPU[3] + dCPU[4]);
	
	int sizeB = (2 * dCPU[0] + dCPU[1] + dCPU[2] + dCPU[3] + dCPU[4]);

	int sizeCB = (2 * dCPU[0] + dCPU[1] + dCPU[2] + dCPU[3] + dCPU[4]) * (1 + dCPU[0]);

	for (int k = 1; k < 1 + dCPU[0]; k++) {
		sizeCB += (deltaSize - k) * (1 + dCPU[0] - k);
		sizeB += (deltaSize - k);
	}

	dCPU[MaxDepth] = sizeCB - deltaSize * (1 + dCPU[0]); // Values should be stored at the end Algo (4.4) deltaC
	dCPU[MaxDepth + 1] = sizeCB; // The total size needed for each configuration s DeltaC
	dCPU[MaxDepth + 3] = sizeB - deltaSize; // Values should be stored at the end Algo (4.4) deltaB
	dCPU[MaxDepth + 4] = sizeB; // The total size needed for each configuration s DeltaB

	// maximum number of vertices by subpolytope: d0 combinations among m-d0
	int NbVertices = 1000;

	dCPU[MaxDepth + 2] = NbVertices; // The total size needed for each configuration s
	
	cudaMemcpyToSymbol(dld, dCPU, Sizedld * sizeof(int), 0, cudaMemcpyHostToDevice);

	float* C, * Ccpu, * LL, * LLcpu, * Ver, * Vercpu, * R, *q;
	int* num, * numcpu; // contains the true number of vertices and levels

	testCUDA(cudaMalloc(&C, sizeCB * nop * sizeof(float)));
	Ccpu = (float*)malloc(sizeCB * nop * sizeof(float));

	testCUDA(cudaMalloc(&LL, 2 * NbVertices * nop * sizeof(float))); // twice the size to be able to have a sorted list
	LLcpu = (float*)malloc(NbVertices * nop * sizeof(float));
	testCUDA(cudaMalloc(&Ver, NbVertices * nop * dCPU[0] * sizeof(float)));
	Vercpu = (float*)malloc(NbVertices * nop * dCPU[0] * sizeof(float));

	int siV = (3 + 2);
	int siVD = (2);
	int siR = (4);
	int siRD = 0;

	testCUDA(cudaMalloc(&R, siR * nop * sizeof(float)));
	testCUDA(cudaMalloc(&num, 2 * nop * sizeof(int)));
	numcpu = (int*)malloc(2 * nop * sizeof(float));

	testCUDA(cudaMalloc(&q, nop * dCPU[0] * sizeof(float)));
	
	float* WBGPU;

	testCUDA(cudaMalloc(&WBGPU, sizeWB * sizeof(float)));

	std::string path = "weights.txt";  // path of the file with one value per row

	// Reading the file
	std::ifstream file(path);
	if (!file.is_open()) {
		std::cerr << "Error: impossible to open " << path << std::endl;
		return 1;
	}

	std::vector<float> WB_data;
	WB_data.reserve(sizeWB);

	float value;
	while (file >> value) {
		WB_data.push_back(value);
	}
	file.close();

	if (WB_data.size() != sizeWB) {
		std::cerr << "Error: " << WB_data.size() << " read values " << sizeWB << std::endl;
		return 1;
	}

	std::cout << "read done of weights: " << WB_data.size() << " floats." << std::endl;

	testCUDA(cudaMemcpy(WBGPU, WB_data.data(), sizeWB * sizeof(float), cudaMemcpyHostToDevice));

	int low = 0;
	int up = nop;
	int nbN = 16; // Number of neurones;
	
	testCUDA(cudaMemset(C, 0, nop * sizeCB * sizeof(float)));

	float Tim;
	cudaEvent_t start, stop;			// GPU timer instructions
	cudaEventCreate(&start);			// GPU timer instructions
	cudaEventCreate(&stop);				// GPU timer instructions
	cudaEventRecord(start, 0);			// GPU timer instructions

	size_t currentLimit;
	cudaDeviceGetLimit(&currentLimit, cudaLimitStackSize);
	printf("Current CUDA stack size: %zu bytes\n", currentLimit);

	size_t NcurrentLimit =  64 * currentLimit;
	cudaDeviceSetLimit(cudaLimitStackSize, NcurrentLimit);
	cudaDeviceGetLimit(&currentLimit, cudaLimitStackSize);
	printf("Current CUDA stack size: %zu bytes\n", currentLimit);

	testCUDA(cudaMemset(LL, 0, 2 * NbVertices * nop * sizeof(float)));
	testCUDA(cudaMemset(Ver, 0, NbVertices * nop * dCPU[0] * sizeof(float)));
	testCUDA(cudaMemset(num, 0, 2 * nop * sizeof(float)));

	Part_k << <16 * 8, 16 * 2 * (dCPU[0] + 1), (sizeWB + 16 * 2 * max(nbN * 4, 15 * dCPU[0])) * sizeof(float) >> > (WBGPU,
																							C, LL, 2 * dCPU[0], sizeWB,
																							low, up, R, q, Ver, num, nbN, 4,
																							siV, siVD, siR, siRD, MinMax);
		
	cudaDeviceSynchronize();
	testCUDA(cudaMemcpy(Ccpu, C, nop * sizeCB * sizeof(float), cudaMemcpyDeviceToHost));
	testCUDA(cudaMemcpy(LLcpu, LL, NbVertices * nop * sizeof(float), cudaMemcpyDeviceToHost));
	testCUDA(cudaMemcpy(Vercpu, Ver, NbVertices * nop * dCPU[0] * sizeof(float), cudaMemcpyDeviceToHost));
	testCUDA(cudaMemcpy(numcpu, num, 2 * nop * sizeof(float), cudaMemcpyDeviceToHost));


	printf("The computed Minimum level %f\n", (float)0.0000001f * MinMax[0]);
	printf("The computed Maximum level %f\n", (float)0.0000001f * MinMax[1]);
	printf("Number of levels %i\n", MinMax[2]);
	printf("With notation (index of pol, number of vertices, number of levels), the non-empty polytopes are:\n");
	int count = 0;
	for (int k = 0; k < nop; k++) {
		if (numcpu[2 * k] > 0) {
			printf("(%d, %d, %d), ", k, numcpu[2 * k], numcpu[2 * k + 1]);
			count++;
		}
	}
	printf("\n");
	printf("The number of non-empty polytopes: %d\n", count);

	cudaEventRecord(stop, 0);			// GPU timer instructions
	cudaEventSynchronize(stop);			// GPU timer instructions
	cudaEventElapsedTime(&Tim,			// GPU timer instructions
		start, stop);					// GPU timer instructions
	cudaEventDestroy(start);			// GPU timer instructions
	cudaEventDestroy(stop);				// GPU timer instructions

	printf("Execution time %f ms\n", Tim);

	testCUDA(cudaFree(MinMax));
	testCUDA(cudaFree(C));
	free(Ccpu);
	testCUDA(cudaFree(LL));
	free(LLcpu);
	testCUDA(cudaFree(Ver));
	free(Vercpu);
	testCUDA(cudaFree(R));
	testCUDA(cudaFree(q));
	testCUDA(cudaFree(num));
	free(numcpu);
	testCUDA(cudaFree(WBGPU));

	return 0;
}