// This is core/vnl/vnl_sample.h
#ifndef elx_sample_h_
#define elx_sample_h_

//#ifdef VCL_NEEDS_PRAGMA_INTERFACE
//#pragma interface
//#endif
//:
//  \file
//  \brief easy ways to sample from various probability distributions

//: re-seed the random number generator.
void elx_sample_reseed();

//: re-seed the random number generator given a seed.
void elx_sample_reseed(int seed);

//: uniform on [a, b)
double elx_sample_uniform(double a, double b);

//: two independent samples from a standard normal distribution.
void elx_sample_normal_2(double *x, double *y);

//: Normal distribution with given mean and standard deviation
double elx_sample_normal(double mean, double sigma);

// P(X = k) = [kth term in binomial expansion of (p + (1-p))^n]
//int vnl_sample_binomial(int n, int k, double p);

// ----------------------------------------

//: handy function to fill a range of values.
template <class I>
inline void elx_sample_uniform(I begin, I end, double a, double b)
{
  for (I p=begin; p!=end; ++p)
    (*p) = elx_sample_uniform(a, b);
}

//: handy function to fill a range of values.
template <class I>
inline void elx_sample_normal(I begin, I end, double mean, double sigma)
{
  for (I p=begin; p!=end; ++p)
    (*p) = elx_sample_normal(mean, sigma);
}

#endif // elx_sample_h_
