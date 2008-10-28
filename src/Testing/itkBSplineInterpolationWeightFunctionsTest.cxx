
#include "itkBSplineInterpolationWeightFunction.h"
#include "itkBSplineInterpolationWeightFunction2.h"
#include "itkBSplineInterpolationDerivativeWeightFunction.h"
#include "itkBSplineInterpolationSecondOrderDerivativeWeightFunction.h"

#include <ctime>

//-------------------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  /** Some basic type definitions. */
	const unsigned int Dimension = 2;
  const unsigned int SplineOrder = 3;
  typedef float CoordinateRepresentationType;

  /** Other typedefs. */
  typedef itk::BSplineInterpolationWeightFunction<
    CoordinateRepresentationType, Dimension, SplineOrder >    WeightFunctionType;
  typedef itk::BSplineInterpolationWeightFunction2<
    CoordinateRepresentationType, Dimension, SplineOrder >    WeightFunction2Type;
  typedef itk::BSplineInterpolationDerivativeWeightFunction<
    CoordinateRepresentationType, Dimension, SplineOrder >    DerivativeWeightFunctionType;
  typedef itk::BSplineInterpolationSecondOrderDerivativeWeightFunction<
    CoordinateRepresentationType, Dimension, SplineOrder >    SODerivativeWeightFunctionType;

  typedef WeightFunctionType::ContinuousIndexType   ContinuousIndexType;
  typedef WeightFunctionType::WeightsType           WeightsType;
  
  /** Construct several weight functions. */
  WeightFunctionType::Pointer weight = WeightFunctionType::New();
  WeightFunction2Type::Pointer weight2 = WeightFunction2Type::New();
  DerivativeWeightFunctionType::Pointer foDweight = DerivativeWeightFunctionType::New();
  SODerivativeWeightFunctionType::Pointer soDweight = SODerivativeWeightFunctionType::New();

  /** Create and fill a continuous index. */
  ContinuousIndexType cindex;
  cindex.Fill( 0.1 );

  /** TESTING. */

  /** Run evaluate for the original ITK implementation. */
  WeightsType weights = weight->Evaluate( cindex );
  //weight->Print( std::cerr, 0 );
  std::cerr << "weights (Weight) " << weights << std::endl;

  std::cerr << "\n-------------------------------\n" << std::endl;

  /** Run evaluate for our modified implementation. */
  WeightsType weights2 = weight2->Evaluate( cindex );
  //weight2->Print( std::cerr, 0 );
  std::cerr << "weights (Weight2) " << weights2 << std::endl;

  /** Compare the two qualitatively. */
  double error = 0.0;
  for ( unsigned int i = 0; i < weights.Size(); ++i )
  {
    error += vnl_math_sqr( weights[ i ] - weights2[ i ] );
  }

  /** TEST. */
  const double epsilon = 1e-1;
  if ( error > epsilon )
  {
    std::cerr << "ERROR: the ITK implementation differs from our "
      << "implementation with more than "
      << static_cast<unsigned int>( epsilon * 100.0 )
      << "%." << std::endl;
    return 1;
  }
  std::cerr << "The difference is: " << vcl_sqrt( error ) << std::endl;

  /** Compare the two performance wise. */
  unsigned int N = 1e7;
  if ( Dimension == 3 ) N = 5e6;

  /** The ITK implementation. */
  clock_t startClock = clock();
  for ( unsigned int i = 0; i < N; ++i )
  {
    weight->Evaluate( cindex );
  }
  clock_t endClock = clock();
  clock_t clockITK = endClock - startClock;
  std::cerr << "The elapsed time for the ITK implementation is: "
    << clockITK << std::endl;

  /** Our own implementation, which is essentially the same, but created
   * a little more general, so that higher order derivatives are also easily
   * implemented.
   */
  startClock = clock();
  for ( unsigned int i = 0; i < N; ++i )
  {
    weight2->Evaluate( cindex );
  }
  endClock = clock();
  clock_t clockOur = endClock - startClock;
  std::cerr << "The elapsed time for our own implementation is: "
    << clockOur << std::endl;
  
  /** TEST. */
  //std::cerr << static_cast<double>( clockITK ) / static_cast<double>( clockOur ) << std::endl;
  const double epsilon2 = 0.2;
  if ( ( static_cast<double>( clockITK ) / static_cast<double>( clockOur ) )
    > ( 1.0 + epsilon2 ) )
  {
    std::cerr << "ERROR: the ITK implementation is more than "
      << static_cast<unsigned int>( epsilon2 * 100.0 )
      << "% faster than our implementation." << std::endl;
    return 1;
  }
  
  /***/
  foDweight->SetDerivativeDirection( 0 );
  WeightsType foWeights = foDweight->Evaluate( cindex );

  soDweight->SetDerivativeDirections( 0, 0 );
  WeightsType soWeights = soDweight->Evaluate( cindex );
  
  /** Return a value. */
  return 0;

} // end main
