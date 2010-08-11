
#include "itkBSplineSecondOrderDerivativeKernelFunction.h"
#include "itkBSplineSecondOrderDerivativeKernelFunction2.h"

#include <ctime>
#include <iomanip>

//-------------------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  /** Some basic type definitions.
   * NOTE: don't change the dimension or the spline order, since the
   * hard-coded ground truth depends on this.
   */
  const unsigned int SplineOrder = 3;

  //const double distance = 1e-3; // the allowable distance
  //const double allowedTimeDifference = 0.1; // 10% is considered within limits
  /** The number of calls to Evaluate(). This number gives reasonably
   * fast test results in Release mode.
   */
  unsigned int N = static_cast<unsigned int>( 1e8 );

  /** Other typedefs. */
  typedef itk::BSplineSecondOrderDerivativeKernelFunction<SplineOrder> BSplineSODerivativeKernelType;
  typedef itk::BSplineSecondOrderDerivativeKernelFunction2<SplineOrder> BSplineSODerivativeKernelType2;

  /** Create the kernel. */
  BSplineSODerivativeKernelType::Pointer dkernel = BSplineSODerivativeKernelType::New();
  const unsigned int size_u = 15;
  std::vector<double> u( size_u );
  u[ 0 ] = -2.5;
  u[ 1 ] = -2.0;
  u[ 2 ] = -1.9;
  u[ 3 ] = -1.5;
  u[ 4 ] = -1.0;
  u[ 5 ] = -0.8;
  u[ 6 ] = -0.5;
  u[ 7 ] = -0.1;
  u[ 8 ] =  0.0;
  for ( unsigned int i = ( size_u + 3 ) / 2; i < size_u; ++i )
  {
    u[ i ] = -u[ -i + size_u + 1 ];
  }

  /** Time the implementation. */
  clock_t startClock = clock();
  for ( unsigned int i = 0; i < N; ++i )
  {
    dkernel->Evaluate( u[ 3 ] );
  }
  clock_t endClock = clock();
  clock_t clockDiff = endClock - startClock;
  std::cerr << "The elapsed time for ITK implementation is: "
    << clockDiff << std::endl;

  /** Create the kernel. */
  BSplineSODerivativeKernelType2::Pointer dkernel2 = BSplineSODerivativeKernelType2::New();

  /** Time the implementation. */
  startClock = clock();
  for ( unsigned int i = 0; i < N; ++i )
  {
    dkernel2->Evaluate( u[ 3 ] );
  }
  endClock = clock();
  clockDiff = endClock - startClock;
  std::cerr << "The elapsed time for our implementation is: "
    << clockDiff << std::endl;

  /***************************************************************************/

  for ( unsigned int i = 0; i < size_u; ++i )
  {
    double diff = dkernel->Evaluate( u[ i ] ) - dkernel2->Evaluate( u[ i ] );
    if ( diff > 1e-5 )
    {
      std::cerr << "ERROR: our implementation differs from ITK." << std::endl;
      return 1;
    }
  }
  std::cerr << "The results are good." << std::endl;
  //std::cerr << "\nITK output: " << dkernel->Evaluate( u ) << std::endl;
  //std::cerr << "Our output: " << dkernel2->Evaluate( u ) << std::endl;

  /** Return a value. */
  return 0;

} // end main
