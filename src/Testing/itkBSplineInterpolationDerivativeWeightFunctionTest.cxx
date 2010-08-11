#include "itkBSplineInterpolationDerivativeWeightFunction.h"

#include <ctime>
#include <iomanip>

//-------------------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  /** Some basic type definitions.
   * NOTE: don't change the dimension or the spline order, since the
   * hard-coded ground truth depends on this.
   */
  const unsigned int Dimension = 2;
  const unsigned int SplineOrder = 3;
  typedef float CoordinateRepresentationType;
  const double distance = 1e-3; // the allowable distance
  //const double allowedTimeDifference = 0.1; // 10% is considered within limits
  /** The number of calls to Evaluate(). This number gives reasonably
   * fast test results in Release mode.
   */
  unsigned int N = static_cast<unsigned int>( 1e7 );

  /** Other typedefs. */
  typedef itk::BSplineInterpolationDerivativeWeightFunction<
    CoordinateRepresentationType,
    Dimension, SplineOrder >                DerivativeWeightFunctionType;
  typedef DerivativeWeightFunctionType::ContinuousIndexType   ContinuousIndexType;
  typedef DerivativeWeightFunctionType::WeightsType           WeightsType;

  /**
   * *********** TESTING ***********************************************
   */

  std::cerr << "TESTING:\n" << std::endl;

  /** Construct several weight functions. */
  DerivativeWeightFunctionType::Pointer foWeightFunction
    = DerivativeWeightFunctionType::New();

  /** Create and fill a continuous index.
   * NOTE: don't change this, since the hard-coded ground truth depends on this.
   */
  ContinuousIndexType cindex;
  cindex[ 0 ] =  3.1;
  cindex[ 1 ] = -2.2;
  foWeightFunction->SetDerivativeDirection( 0 );

  /** Run evaluate for the first order derivative. */
  WeightsType foWeights = foWeightFunction->Evaluate( cindex );
  std::cerr << "weights (1st order) " << foWeights << std::endl;

  /** Hard-code the ground truth. You should change this if you change the
   * spline order.
   *
   * x1 =  3.1  ->  support y1 =  2  3  4  5  ->  x1 - y1 = 1.1 0.1 -0.9 -1.9
   * x2 = -2.2  ->  support y2 = -4 -3 -3 -1  ->  x2 - y2 = 1.8 0.8 -0.2 -1.2
   *
   * B3 is the third order B-spline. ?etc means repeat ? for ever.
   * The coefficients are [ B2(x1-y1i+1/2)-B2(x1-y1i-1/2) ] * B3(x2-y2j):
   *
   * B3d(  1.1 ) = -0.405
   * B3d(  0.1 ) = -0.185
   * B3d( -0.9 ) =  0.585
   * B3d( -1.9 ) =  0.005
   * B3 (  1.8 ) =  0.0013etc
   * B3 (  0.8 ) =  0.2826etc
   * B3 ( -0.2 ) =  0.6306etc
   * B3 ( -1.2 ) =  0.0853etc
   *
   *                         -> i
   *      -5.4e-4    -2.46/e-4     7.8e-4    6.6/e-6
   *  |   -0.11448   -5.2293/e-2   0.16536   1.413/e-3
   *  j   -0.25542   -0.116673/    0.36894   3.153/e-3
   *      -0.03456   -0.015786/    0.04992   4.26/e-4
   *
   * These numbers are created by a small Matlab program. So, if this appears
   * to be not a valid check, then we made the same bug twice.
   */
  WeightsType trueFOWeights( 16 );
  trueFOWeights.Fill( 0.0 );
  trueFOWeights[  0 ] = -5.400000000000e-4;
  trueFOWeights[  1 ] = -2.466666666666e-4;
  trueFOWeights[  2 ] =  7.800000000000e-4;
  trueFOWeights[  3 ] =  6.666666666666e-6;
  trueFOWeights[  4 ] = -1.144800000000e-1;
  trueFOWeights[  5 ] = -5.229333333333e-2;
  trueFOWeights[  6 ] =  1.653600000000e-1;
  trueFOWeights[  7 ] =  1.413333333333e-3;
  trueFOWeights[  8 ] = -2.554200000000e-1;
  trueFOWeights[  9 ] = -1.166733333333e-1;
  trueFOWeights[ 10 ] =  3.689400000000e-1;
  trueFOWeights[ 11 ] =  3.153333333333e-3;
  trueFOWeights[ 12 ] = -3.456000000000e-2;
  trueFOWeights[ 13 ] = -1.578666666666e-2;
  trueFOWeights[ 14 ] =  4.992000000000e-2;
  trueFOWeights[ 15 ] =  4.266666666666e-4;

  /** Compute the distance between the two vectors. */
  double error = 0.0;
  for ( unsigned int i = 0; i < foWeights.Size(); ++i )
  {
    error += vnl_math_sqr( foWeights[ i ] - trueFOWeights[ i ] );
  }
  error = vcl_sqrt( error );

  /** TEST: Compare the two qualitatively. */
  if ( error > distance )
  {
    std::cerr << "ERROR: the first order weights differs more than "
      << distance << " from the truth." << std::endl;
    return 1;
  }
  std::cerr << std::showpoint;
  std::cerr << std::scientific;
  std::cerr << std::setprecision( 4 );
  std::cerr << "The distance is: " << error << std::endl;

  /** Time the fo implementation. */
  clock_t startClock = clock();
  for ( unsigned int i = 0; i < N; ++i )
  {
    foWeightFunction->Evaluate( cindex );
  }
  clock_t endClock = clock();
  clock_t clockITK = endClock - startClock;
  std::cerr << "The elapsed time for the 1st order derivative is: "
    << clockITK << std::endl;

  /**
   * *********** Function TESTING ****************************************
   */

  std::cerr << "\n--------------------------------------------------------";
  std::cerr << "\nFunction TESTING:\n" << std::endl;

  /** Just call all available public functions. */
  DerivativeWeightFunctionType::IndexType startIndex;
  DerivativeWeightFunctionType::IndexType trueStartIndex;
  trueStartIndex[ 0 ] =  2;
  trueStartIndex[ 1 ] = -4;
  foWeightFunction->ComputeStartIndex( cindex, startIndex );
  if ( startIndex != trueStartIndex )
  {
    std::cerr << "ERROR: wrong start index was computed." << std::endl;
    return 1;
  }

  DerivativeWeightFunctionType::SizeType trueSize;
  trueSize.Fill( SplineOrder + 1 );
  if ( foWeightFunction->GetSupportSize() != trueSize )
  {
    std::cerr << "ERROR: wrong support size was computed." << std::endl;
    return 1;
  }

  if ( foWeightFunction->GetNumberOfWeights()
    != static_cast<unsigned long>( vcl_pow(
    static_cast<float>( SplineOrder + 1 ), 2.0f ) ) )
  {
    std::cerr << "ERROR: wrong number of weights was computed." << std::endl;
    return 1;
  }

  std::cerr << "All public functions returned valid output." << std::endl;

  /**
   * *********** PrintSelf TESTING ****************************************
   */

  std::cerr << "\n--------------------------------------------------------";
  std::cerr << "\nPrintSelf() TESTING:\n" << std::endl;

  foWeightFunction->Print( std::cerr, 0 );

  /** Return a value. */
  return 0;

} // end main
