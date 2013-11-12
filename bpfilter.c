void bpfilter(float *trace, float dt, int nt, float a, float b, float c, float d)
{
  int iw,nw,ntfft;
  float *in1, *out2;
  sf_complex *in2,*out1;
  sf_complex czero;
  fftwf_plan p1;
  fftwf_plan p2;

  __real__ czero = 0;
  __imag__ czero = 0;
  ntfft = 4*nt;
  nw=ntfft/2+1;
  if(a>0) ia = trunc(a*dt*ntfft);
  else ia = 0;
  if(b>0) ib = trunc(b*dt*ntfft);
  else ib = 1;
  if(c*dt*ntfft<nw) ic = trunc(c*dt*ntfft);
  else ic = nw-1;
  if(d*dt*ntfft<nw) id = trunc(d*dt*ntfft);
  else id = nw;

  out1 = sf_complexalloc(nw);
  in1  = sf_floatalloc(N);
  p1   = fftwf_plan_dft_r2c_1d(ntfft, in1, (fftwf_complex*)out1, FFTW_ESTIMATE);
  out2 = sf_floatalloc(nzfft);
  in2  = sf_complexalloc(N);
  p2   = fftwf_plan_dft_c2r_1d(ntfft, (fftwf_complex*)in2, out2, FFTW_ESTIMATE);

  for (it=0; it<nt; it++) in1[it]=trace[it];
  for (it=nt; it< ntfft;it++) in1[it] = 0.0;
  fftwf_execute(p1);
  for(iw=0;iw<ia;iw++)  in2[iw] = czero; 
  for(iw=ia;iw<ib;iw++) in2[iw] = out1[iw]*((float) (iw-ia)/(ib-ia)); 
  for(iw=ib;iw<ic;iw++) in2[iw] = out1[iw]; 
  for(iw=ic;iw<id;iw++) in2[iw] = out1[iw]*(1 - (float) (iw-ic)/(id-ic)); 
  for(iw=id;iw<nw;iw++) in2[iw] = czero; 
  fftwf_execute(p2); /* take the FFT along the time dimension */
  for(it=0;it<nt;it++) trace[it] = out2[it]/ntfft; 
  
  fftwf_destroy_plan(p1);
  fftwf_destroy_plan(p2);
  fftwf_free(in1); fftwf_free(out1);
  fftwf_free(in2); fftwf_free(out2);
  return;
}

