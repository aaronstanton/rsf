#include <rsf.h>

#include "myalloc.h"

void *alloc1 (size_t n1, size_t size)
{
	void *p;

	if ((p=malloc(n1*size))==NULL)
		return NULL;
	return p;
}

void free1 (void *p)
{
	free(p);
}

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

void free2 (void **p)
{
	free(p[0]);
	free(p);
}

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

void free3 (void ***p)
{
	free(p[0][0]);
	free(p[0]);
	free(p);
}

int *alloc1int(size_t n1)
{
	return (int*)alloc1(n1,sizeof(int));
}

void free1int(int *p)
{
	free1(p);
}

int **alloc2int(size_t n1, size_t n2)
{
	return (int**)alloc2(n1,n2,sizeof(int));
}

void free2int(int **p)
{
	free2((void**)p);
}

int ***alloc3int(size_t n1, size_t n2, size_t n3)
{
	return (int***)alloc3(n1,n2,n3,sizeof(int));
}

void free3int(int ***p)
{
	free3((void***)p);
}

float *alloc1float(size_t n1)
{
	return (float*)alloc1(n1,sizeof(float));
}

void free1float(float *p)
{
	free1(p);
}

float **alloc2float(size_t n1, size_t n2)
{
	return (float**)alloc2(n1,n2,sizeof(float));
}

void free2float(float **p)
{
	free2((void**)p);
}

float ***alloc3float(size_t n1, size_t n2, size_t n3)
{
	return (float***)alloc3(n1,n2,n3,sizeof(float));
}

void free3float(float ***p)
{
	free3((void***)p);
}

sf_complex *alloc1complex(size_t n1)
{
	return (sf_complex*)alloc1(n1,sizeof(sf_complex));
}

void free1complex(sf_complex *p)
{
	free1(p);
}

sf_complex **alloc2complex(size_t n1, size_t n2)
{
	return (sf_complex**)alloc2(n1,n2,sizeof(sf_complex));
}

void free2complex(sf_complex **p)
{
	free2((void**)p);
}

sf_complex ***alloc3complex(size_t n1, size_t n2, size_t n3)
{
	return (sf_complex***)alloc3(n1,n2,n3,sizeof(sf_complex));
}

void free3complex(sf_complex ***p)
{
	free3((void***)p);
}


