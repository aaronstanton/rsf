/* Copyright (c) University of Alberta, 2013.*/
/* All rights reserved.                       */
/* parts taken from seismicunix's alloc.c*/

/* Allocate and free multi-dimensional arrays

alloc1		allocate a 1-d array
free1		free a 1-d array
alloc2		allocate a 2-d array
free2		free a 2-d array
alloc3		allocate a 3-d array
free3		free a 3-d array
free1int	free a 1-d array of ints
alloc2int	allocate a 2-d array of ints
free2int	free a 2-d array of ints
alloc3int	allocate a 3-d array of ints
free3int	free a 3-d array of ints
alloc1float	allocate a 1-d array of floats
free1float	free a 1-d array of floats
alloc2float	allocate a 2-d array of floats
free2float	free a 2-d array of floats
alloc3float	allocate a 3-d array of floats
free3float	free a 3-d array of floats
alloc1complex	allocate a 1-d array of complexs
free1complex	free a 1-d array of complexs
alloc2complex	allocate a 2-d array of complexs
free2complex	free a 2-d array of complexs
alloc3complex	allocate a 3-d array of complexs
free3complex	free a 3-d array of complexs

******************************************************************************
Function Prototypes:
void *alloc1 (size_t n1, size_t size);
void free1 (void *p);
void **alloc2 (size_t n1, size_t n2, size_t size);
void free2 (void **p);
void ***alloc3 (size_t n1, size_t n2, size_t n3, size_t size);
void free3 (void ***p);
int *alloc1int (size_t n1);
void free1int (int *p);
int **alloc2int (size_t n1, size_t n2);
void free2int (int **p);
int ***alloc3int (size_t n1, size_t n2, size_t n3);
void free3int (int ***p);
float *alloc1float (size_t n1);
void free1float (float *p);
float **alloc2float (size_t n1, size_t n2);
void free2float (float **p);
float ***alloc3float (size_t n1, size_t n2, size_t n3);
void free3float (float ***p);
complex *alloc1complex (size_t n1);
void free1complex (complex *p);
complex **alloc2complex (size_t n1, size_t n2);
void free2complex (complex **p);
complex ***alloc3complex (size_t n1, size_t n2, size_t n3);
void free3complex (complex ***p);

#include <rsf.h>

/* allocate a 1-d array */
void *alloc1 (size_t n1, size_t size)
{
	void *p;

	if ((p=malloc(n1*size))==NULL)
		return NULL;
	return p;
}

/* free a 1-d array */
void free1 (void *p)
{
	free(p);
}

/* allocate a 2-d array */
void **alloc2 (size_t n1, size_t n2, size_t size)
{
	size_t i2;
	void **p;

	if ((p=(void**)malloc(n2*sizeof(void*)))==NULL) 
		return NULL;
	if ((p[0]=(void*)malloc(n2*n1*size))==NULL) {
		free(p);
		return NULL;
	}
	for (i2=0; i2<n2; i2++)
		p[i2] = (char*)p[0]+size*n1*i2;
	return p;
}

/* free a 2-d array */
void free2 (void **p)
{
	free(p[0]);
	free(p);
}

/* allocate a 3-d array */
void ***alloc3 (size_t n1, size_t n2, size_t n3, size_t size)
{
	size_t i3,i2;
	void ***p;

	if ((p=(void***)malloc(n3*sizeof(void**)))==NULL)
		return NULL;
	if ((p[0]=(void**)malloc(n3*n2*sizeof(void*)))==NULL) {
		free(p);
		return NULL;
	}
	if ((p[0][0]=(void*)malloc(n3*n2*n1*size))==NULL) {
		free(p[0]);
		free(p);
		return NULL;
	}

	for (i3=0; i3<n3; i3++) {
		p[i3] = p[0]+n2*i3;
		for (i2=0; i2<n2; i2++)
			p[i3][i2] = (char*)p[0][0]+size*n1*(i2+n2*i3);
	}
	return p;
}

/* free a 3-d array */
void free3 (void ***p)
{
	free(p[0][0]);
	free(p[0]);
	free(p);
}

/* allocate a 1-d array of ints */
int *alloc1int(size_t n1)
{
	return (int*)alloc1(n1,sizeof(int));
}

/* free a 1-d array of ints */
void free1int(int *p)
{
	free1(p);
}

/* allocate a 2-d array of ints */
int **alloc2int(size_t n1, size_t n2)
{
	return (int**)alloc2(n1,n2,sizeof(int));
}

/* free a 2-d array of ints */
void free2int(int **p)
{
	free2((void**)p);
}

/* allocate a 3-d array of ints */
int ***alloc3int(size_t n1, size_t n2, size_t n3)
{
	return (int***)alloc3(n1,n2,n3,sizeof(int));
}

/* free a 3-d array of ints */
void free3int(int ***p)
{
	free3((void***)p);
}

/* allocate a 1-d array of floats */
float *alloc1float(size_t n1)
{
	return (float*)alloc1(n1,sizeof(float));
}

/* free a 1-d array of floats */
void free1float(float *p)
{
	free1(p);
}

/* allocate a 2-d array of floats */
float **alloc2float(size_t n1, size_t n2)
{
	return (float**)alloc2(n1,n2,sizeof(float));
}

/* free a 2-d array of floats */
void free2float(float **p)
{
	free2((void**)p);
}

/* allocate a 3-d array of floats */
float ***alloc3float(size_t n1, size_t n2, size_t n3)
{
	return (float***)alloc3(n1,n2,n3,sizeof(float));
}

/* free a 3-d array of floats */
void free3float(float ***p)
{
	free3((void***)p);
}

/* allocate a 1-d array of complexs */
sf_complex *alloc1complex(size_t n1)
{
	return (sf_complex*)alloc1(n1,sizeof(sf_complex));
}

/* free a 1-d array of complexs */
void free1complex(complex *p)
{
	free1(p);
}

/* allocate a 2-d array of complexs */
sf_complex **alloc2complex(size_t n1, size_t n2)
{
	return (sf_complex**)alloc2(n1,n2,sizeof(sf_complex));
}

/* free a 2-d array of complexs */
void free2complex(sf_complex **p)
{
	free2((void**)p);
}

/* allocate a 3-d array of complexs */
sf_complex ***alloc3complex(size_t n1, size_t n2, size_t n3)
{
	return (sf_complex***)alloc3(n1,n2,n3,sizeof(sf_complex));
}

/* free a 3-d array of complexs */
void free3complex(sf_complex ***p)
{
	free3((void***)p);
}


