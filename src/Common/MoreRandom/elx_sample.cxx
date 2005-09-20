// This is and adapted version of core/vnl/vnl_sample.cxx
// Stefan Klein, february 2005.

//#ifdef VCL_NEEDS_PRAGMA_INTERFACE
//#pragma implementation
//#endif
//:
// \file
// \author fsm

#include "elx_sample.h"
#include <vnl/vnl_math.h>

#include <vcl_cmath.h>
#include <vxl_config.h>
#include <vcl_ctime.h>

#define ELX_MORE_RANDOM
//#define ELX_NOT_SO_RANDOM
//#define ELX_DRAND48


#if defined(ELX_MORE_RANDOM)
  #include "vnl/vnl_random.h"
  static const unsigned long initial_seed = 9667566;
  static vnl_random morerandom_generator(initial_seed);
#elif defined(ELX_NOT_SO_RANDOM)
	static unsigned long elx_sample_seed = 12345;
#elif defined(ELX_DRAND48)
  #include <stdlib.h> // dont_vxl_filte
#endif

void elx_sample_reseed()
{
#if defined(ELX_MORE_RANDOM)
	morerandom_generator.reseed();
#elif defined(ELX_NOT_SO_RANDOM)
	elx_sample_seed = (unsigned long)vcl_time(0);
#elif defined(ELX_DRAND48)
	srand48( vcl_time(0) );
#endif
}

void elx_sample_reseed(int seed)
{
#if defined(ELX_MORE_RANDOM)
	morerandom_generator.reseed(static_cast<unsigned long>(seed));
#elif defined(ELX_NOT_SO_RANDOM)
	elx_sample_seed = seed;
#elif defined(ELX_DRAND48)
	srand48( seed );
#endif
}

//: return a random number uniformly drawn on [a, b)
double elx_sample_uniform(double a, double b)
{
#if defined(ELX_MORE_RANDOM)
	return morerandom_generator.drand64(a,b);
#elif defined(ELX_NOT_SO_RANDOM)
  elx_sample_seed = (elx_sample_seed*16807)%2147483647L;
  double u = double(elx_sample_seed)/2147483711UL;
#elif defined(ELX_DRAND48)
	double u = drand48(); // uniform on [0, 1)
#endif
#ifndef ELX_MORE_RANDOM
	return (1.0 - u)*a + u*b;
#endif
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


