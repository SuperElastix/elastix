/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#include "itkBSplineInterpolationWeightFunction.h"
#include "itkBSplineInterpolationWeightFunction2.h"

#include <ctime>
#include <iomanip>

//-------------------------------------------------------------------------------------
// This test tests the itkBSplineInterpolationWeightFunction2 and compares
// it with the ITK implementation. It should give equal results and comparable
// performance. The test is performed in 2D and 3D, with spline order 3.
// Also the PrintSelf()-functions are called.

int
main( int argc, char * argv[] )
{
  /** Some basic type definitions. */
  const unsigned int SplineOrder = 3;
  typedef float CoordinateRepresentationType;
  const double distance              = 1e-3; // the allowable distance
  const double allowedTimeDifference = 0.2;  // 20% is considered within limits
  /** The number of calls to Evaluate() in 2D. This number gives reasonably
   * fast test results in Release mode. In 3D half of this number calls are
   * made.
   */
  unsigned int N = static_cast< unsigned int >( 1e6 );

  /** Other typedefs. */
  typedef itk::BSplineInterpolationWeightFunction<
    CoordinateRepresentationType, 2, SplineOrder >    WeightFunctionType2D;
  typedef itk::BSplineInterpolationWeightFunction2<
    CoordinateRepresentationType, 2, SplineOrder >    WeightFunction2Type2D;
  typedef itk::BSplineInterpolationWeightFunction<
    CoordinateRepresentationType, 3, SplineOrder >    WeightFunctionType3D;
  typedef itk::BSplineInterpolationWeightFunction2<
    CoordinateRepresentationType, 3, SplineOrder >    WeightFunction2Type3D;
  typedef WeightFunctionType2D::ContinuousIndexType ContinuousIndexType2D;
  typedef WeightFunctionType2D::WeightsType         WeightsType2D;
  typedef WeightFunctionType3D::ContinuousIndexType ContinuousIndexType3D;
  typedef WeightFunctionType3D::WeightsType         WeightsType3D;

  /**
   * *********** 2D TESTING ***********************************************
   */

  std::cerr << "2D TESTING:\n" << std::endl;

  /** Construct several weight functions. */
  WeightFunctionType2D::Pointer  weightFunction2D  = WeightFunctionType2D::New();
  WeightFunction2Type2D::Pointer weight2Function2D = WeightFunction2Type2D::New();

  /** Create and fill a continuous index. */
  ContinuousIndexType2D cindex;
  cindex.Fill( 0.1 );

  /** Run evaluate for the original ITK implementation. */
  WeightsType2D weights2D = weightFunction2D->Evaluate( cindex );
  //std::cerr << "weights (ITK) " << weights2D << std::endl;
  unsigned int weightsSize = weights2D.Size();
  std::cerr << "weights (ITK): ["
            << weights2D[ 0 ] << ", " << weights2D[ 1 ] << ", ..., "
            << weights2D[ weightsSize - 2 ] << ", " << weights2D[ weightsSize - 1 ]
            << "]" << std::endl;

  /** Run evaluate for our modified implementation. */
  WeightsType2D weights2_2D = weight2Function2D->Evaluate( cindex );
  //std::cerr << "weights (our) " << weights2_2D << std::endl;
  std::cerr << "weights (our): ["
            << weights2_2D[ 0 ] << ", " << weights2_2D[ 1 ] << ", ..., "
            << weights2_2D[ weightsSize - 2 ] << ", " << weights2_2D[ weightsSize - 1 ]
            << "]" << std::endl;

  /** Compute the distance between the two vectors. */
  double error = 0.0;
  for( unsigned int i = 0; i < weights2D.Size(); ++i )
  {
    error += vnl_math_sqr( weights2D[ i ] - weights2_2D[ i ] );
  }
  error = vcl_sqrt( error );

  /** TEST: Compare the two qualitatively. */
  if( error > distance )
  {
    std::cerr << "ERROR: the ITK implementation differs from our "
              << "implementation with more than "
              << static_cast< unsigned int >( distance * 100.0 )
              << "%." << std::endl;
    return EXIT_FAILURE;
  }
  std::cerr << std::showpoint;
  std::cerr << std::scientific;
  std::cerr << std::setprecision( 4 );
  std::cerr << "The distance is: " << error << std::endl;

  /** Time the ITK implementation. */
  clock_t startClock = clock();
  for( unsigned int i = 0; i < N; ++i )
  {
    weightFunction2D->Evaluate( cindex );
  }
  clock_t endClock = clock();
  clock_t clockITK = endClock - startClock;
  std::cerr << "The elapsed time for the ITK implementation is: "
            << clockITK << std::endl;

  /** Time our own implementation, which is essentially the same, but created
   * a little more general, so that higher order derivatives are also easily
   * implemented.
   */
  startClock = clock();
  for( unsigned int i = 0; i < N; ++i )
  {
    weight2Function2D->Evaluate( cindex );
  }
  endClock = clock();
  clock_t clockOur = endClock - startClock;
  std::cerr << "The elapsed time for our own implementation is: "
            << clockOur << std::endl;

  /** TEST: Compare the two performance wise. */
  double timeDifference = static_cast< double >( clockITK )
    / static_cast< double >( clockOur );
  std::cerr << std::fixed;
  std::cerr << std::setprecision( 1 );
  std::cerr << "The time difference is " << ( timeDifference - 1.0 ) * 100.0
            << "% in favor of "
            << ( timeDifference > 1.0 ? "our " : "the ITK " )
            << "implementation." << std::endl;
  if( timeDifference < ( 1.0 - allowedTimeDifference ) )
  {
    std::cerr << "ERROR: the ITK implementation is more than "
              << static_cast< unsigned int >( allowedTimeDifference * 100.0 )
              << "% faster than our implementation." << std::endl;
#if _ELASTIX_TEST_TIMING
    return EXIT_FAILURE;
#endif
  }

  /**
   * *********** 3D TESTING ***********************************************
   */

  std::cerr << "\n--------------------------------------------------------";
  std::cerr << "\n3D TESTING:\n" << std::endl;

  /** Construct several weight functions. */
  WeightFunctionType3D::Pointer  weightFunction3D  = WeightFunctionType3D::New();
  WeightFunction2Type3D::Pointer weight2Function3D = WeightFunction2Type3D::New();

  /** Create and fill a continuous index. */
  ContinuousIndexType3D cindex3D;
  cindex3D.Fill( 0.1 );

  /** Run evaluate for the original ITK implementation. */
  WeightsType3D weights3D = weightFunction3D->Evaluate( cindex3D );
  std::cerr << std::setprecision( 6 );
  //std::cerr << "weights (ITK) " << weights3D << std::endl;
  weightsSize = weights3D.Size();
  std::cerr << "weights (ITK): ["
            << weights3D[ 0 ] << ", " << weights3D[ 1 ] << ", ..., "
            << weights3D[ weightsSize - 2 ] << ", " << weights3D[ weightsSize - 1 ]
            << "]" << std::endl;

  /** Run evaluate for our modified implementation. */
  WeightsType3D weights2_3D = weight2Function3D->Evaluate( cindex3D );
  //std::cerr << "weights (our) " << weights2_3D << std::endl;
  std::cerr << "weights (our): ["
            << weights2_3D[ 0 ] << ", " << weights2_3D[ 1 ] << ", ..., "
            << weights2_3D[ weightsSize - 2 ] << ", " << weights2_3D[ weightsSize - 1 ]
            << "]" << std::endl;

  /** Compute the distance between the two vectors. */
  error = 0.0;
  for( unsigned int i = 0; i < weights3D.Size(); ++i )
  {
    error += vnl_math_sqr( weights3D[ i ] - weights2_3D[ i ] );
  }
  error = vcl_sqrt( error );

  /** TEST: Compare the two qualitatively. */
  if( error > distance )
  {
    std::cerr << "ERROR: the ITK implementation differs from our "
              << "implementation with more than "
              << static_cast< unsigned int >( distance * 100.0 )
              << "%." << std::endl;
    return EXIT_FAILURE;
  }
  std::cerr << std::scientific;
  std::cerr << std::setprecision( 4 );
  std::cerr << "The distance is: " << error << std::endl;

  /** TEST: Compare the two performance wise. */
  N /= 2;

  /** Time the ITK implementation. */
  startClock = clock();
  for( unsigned int i = 0; i < N; ++i )
  {
    weightFunction3D->Evaluate( cindex3D );
  }
  endClock = clock();
  clockITK = endClock - startClock;
  std::cerr << "The elapsed time for the ITK implementation is: "
            << clockITK << std::endl;

  /** Time our own implementation, which is essentially the same, but created
   * a little more general, so that higher order derivatives are also easily
   * implemented.
   */
  startClock = clock();
  for( unsigned int i = 0; i < N; ++i )
  {
    weight2Function3D->Evaluate( cindex3D );
  }
  endClock = clock();
  clockOur = endClock - startClock;
  std::cerr << "The elapsed time for our own implementation is: "
            << clockOur << std::endl;

  /** TEST: Compare the two performance wise. */
  timeDifference = static_cast< double >( clockITK )
    / static_cast< double >( clockOur );
  std::cerr << std::fixed;
  std::cerr << std::setprecision( 1 );
  std::cerr << "The time difference is " << ( timeDifference - 1.0 ) * 100.0
            << "% in favor of "
            << ( timeDifference > 1.0 ? "our " : "the ITK " )
            << "implementation." << std::endl;
  if( timeDifference < ( 1.0 - allowedTimeDifference ) )
  {
    std::cerr << "ERROR: the ITK implementation is more than "
              << static_cast< unsigned int >( allowedTimeDifference * 100.0 )
              << "% faster than our implementation." << std::endl;
#if _ELASTIX_TEST_TIMING
    return EXIT_FAILURE;
#endif
  }

  /**
   * *********** Function TESTING ****************************************
   */

  std::cerr << "\n--------------------------------------------------------";
  std::cerr << "\nFunction TESTING:\n" << std::endl;

  /** Just call all available public functions. */
  WeightFunction2Type2D::IndexType startIndex;
  WeightFunction2Type2D::IndexType trueStartIndex;
  trueStartIndex.Fill( -1 );
  weight2Function2D->ComputeStartIndex( cindex, startIndex );
  if( startIndex != trueStartIndex )
  {
    std::cerr << "ERROR: wrong start index was computed." << std::endl;
    return EXIT_FAILURE;
  }

  WeightFunction2Type2D::SizeType trueSize;
  trueSize.Fill( SplineOrder + 1 );
  if( weight2Function2D->GetSupportSize() != trueSize )
  {
    std::cerr << "ERROR: wrong support size was computed." << std::endl;
    return EXIT_FAILURE;
  }

  if( weight2Function2D->GetNumberOfWeights()
    != static_cast< unsigned long >( vcl_pow(
    static_cast< float >( SplineOrder + 1 ), 2.0f ) ) )
  {
    std::cerr << "ERROR: wrong number of weights was computed." << std::endl;
    return EXIT_FAILURE;
  }

  std::cerr << "All public functions returned valid output." << std::endl;

  /**
   * *********** PrintSelf TESTING ****************************************
   */

  std::cerr << "\n--------------------------------------------------------";
  std::cerr << "\nPrintSelf() TESTING:\n" << std::endl;

  weightFunction2D->Print( std::cerr, 0 );
  std::cerr << "\n--------------------------------------------------------\n";
  weight2Function2D->Print( std::cerr, 0 );
  std::cerr << "\n--------------------------------------------------------\n";
  weightFunction3D->Print( std::cerr, 0 );
  std::cerr << "\n--------------------------------------------------------\n";
  weight2Function3D->Print( std::cerr, 0 );

  /** Return a value. */
  return EXIT_SUCCESS;

} // end main
