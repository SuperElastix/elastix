// This is core/vnl/vnl_sample.cxx

//#ifdef VCL_NEEDS_PRAGMA_INTERFACE
//#pragma implementation
//#endif
//:
// \file
// \author fsm

#include "elx_sample.h"
#include <vnl/vnl_math.h>

#include <vcl_cmath.h>
#include <vcl_ctime.h>

//#include <vxl_config.h>
#ifdef VXL_STDLIB_HAS_SRAND48
	#undef VXL_STDLIB_HAS_SRAND48
#endif
#ifdef VXL_STDLIB_HAS_DRAND48
	#undef VXL_STDLIB_HAS_DRAND48
#endif

#if VXL_STDLIB_HAS_DRAND48
# include <stdlib.h> // dont_vxl_filter
#else
// rand() is not always a good random number generator,
// so use a simple congruential random number generator - PVr
static unsigned long elx_sample_seed = 12345;
#endif

void elx_sample_reseed()
{
#if VXL_STDLIB_HAS_SRAND48
  srand48( vcl_time(0) );
#elif !VXL_STDLIB_HAS_DRAND48
  elx_sample_seed = (unsigned long)vcl_time(0);
#endif
}

void elx_sample_reseed(int seed)
{
#if VXL_STDLIB_HAS_SRAND48
  srand48( seed );
#elif !VXL_STDLIB_HAS_DRAND48
  elx_sample_seed = seed;
#endif
}

//: return a random number uniformly drawn on [a, b)
double elx_sample_uniform(double a, double b)
{
#if VXL_STDLIB_HAS_DRAND48
  double u = drand48(); // uniform on [0, 1)
#else
  elx_sample_seed = (elx_sample_seed*16807)%2147483647L;
  double u = double(elx_sample_seed)/2147483711UL;
#endif
  return (1.0 - u)*a + u*b;
}

void elx_sample_normal_2(double *x, double *y)
{
  double u     = elx_sample_uniform(0, 1);
  double theta = elx_sample_uniform(0, 2 * vnl_math::pi);

  double r = vcl_sqrt(-2*vcl_log(u));

  if (x) *x = r * vcl_cos(theta);
  if (y) *y = r * vcl_sin(theta);
}

double elx_sample_normal(double mean, double sigma)
{
  double x;
  elx_sample_normal_2(&x, 0);
  return mean + sigma * x;
}


