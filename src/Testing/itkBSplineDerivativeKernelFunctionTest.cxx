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
#include "itkBSplineDerivativeKernelFunction.h"
#include "itkBSplineDerivativeKernelFunction2.h"
#include "itkKernelFunctionBase.h"

#include <ctime>
#include <iomanip>

//-------------------------------------------------------------------------------------

int
main( int argc, char * argv[] )
{
  /** Some basic type definitions. */
  std::vector< unsigned int > splineOrders;
  splineOrders.push_back( 1 );
  splineOrders.push_back( 2 );
  splineOrders.push_back( 3 );

  /** The number of calls to Evaluate(). This number gives reasonably
   * fast test results in Release mode.
   * Increase it for real time testing.
   */
  unsigned int N                  = static_cast< unsigned int >( 1e7 );
  const double maxAllowedDistance = 1e-5; // the allowable distance

  /** Other typedefs. */
  typedef itk::KernelFunctionBase< double >          BaseKernelType;
  typedef itk::BSplineDerivativeKernelFunction< 1 >  KernelType_ITK_1;
  typedef itk::BSplineDerivativeKernelFunction2< 1 > KernelType_elx_1;
  typedef itk::BSplineDerivativeKernelFunction< 2 >  KernelType_ITK_2;
  typedef itk::BSplineDerivativeKernelFunction2< 2 > KernelType_elx_2;
  typedef itk::BSplineDerivativeKernelFunction< 3 >  KernelType_ITK_3;
  typedef itk::BSplineDerivativeKernelFunction2< 3 > KernelType_elx_3;

  /** Create the evaluation points. */
  //const unsigned int size_u = 17;
  std::vector< double > u;
  u.push_back( -2.5 );
  u.push_back( -2.0 );
  u.push_back( -1.9 );
  u.push_back( -1.5 );
  u.push_back( -1.0 );
  u.push_back( -0.8 );
  u.push_back( -0.5 );
  u.push_back( -0.1 );
  u.push_back(  0.0 );
  for( int i = static_cast< int >( u.size() ) - 2; i > -1; --i )
  {
    u.push_back( -u[ i ] );
  }

  /** For all spline orders. */
  for( unsigned int so = 0; so < splineOrders.size(); so++ )
  {
    std::cerr << "Evaluating spline order " << splineOrders[ so ] << std::endl;

    /** Create the kernel. */
    BaseKernelType::Pointer kernel_ITK, kernel_elx;
    if( splineOrders[ so ] == 1 )
    {
      kernel_ITK = KernelType_ITK_1::New();
      kernel_elx = KernelType_elx_1::New();
    }
    else if( splineOrders[ so ] == 2 )
    {
      kernel_ITK = KernelType_ITK_2::New();
      kernel_elx = KernelType_elx_2::New();
    }
    else if( splineOrders[ so ] == 3 )
    {
      kernel_ITK = KernelType_ITK_3::New();
      kernel_elx = KernelType_elx_3::New();
    }
    else
    {
      std::cerr << "ERROR: spline order " << splineOrders[ so ]
                << " not defined." << std::endl;
      return 1;
    }

    /** Print header. */
    std::cerr << "eval at:";
    for( unsigned int j = 0; j < u.size(); j++ )
    {
      std::cerr << " " << u[ j ];
    }
    std::cerr << " average" << std::endl;

    /** Time the ITK implementation. */
    std::cerr << "ITK new:";
    clock_t startClock = clock();
    for( unsigned int j = 0; j < u.size(); j++ )
    {
      clock_t startClockRegion = clock();
      for( unsigned int i = 0; i < N; ++i )
      {
        kernel_ITK->Evaluate( u[ j ] );
      }
      std::cerr << " " << ( clock() - startClockRegion ) * 1000.0 / CLOCKS_PER_SEC;
    }
    clock_t endClock  = clock();
    clock_t clockDiff = endClock - startClock;
    std::cerr << " " << clockDiff * 1000.0 / CLOCKS_PER_SEC << " ms" << std::endl;

    /** Time the elx implementation. */
    std::cerr << "elastix:";
    startClock = clock();
    for( unsigned int j = 0; j < u.size(); j++ )
    {
      clock_t startClockRegion = clock();
      for( unsigned int i = 0; i < N; ++i )
      {
        kernel_elx->Evaluate( u[ j ] );
      }
      std::cerr << " " << ( clock() - startClockRegion ) * 1000.0 / CLOCKS_PER_SEC;
    }
    endClock  = clock();
    clockDiff = endClock - startClock;
    std::cerr << " " << clockDiff * 1000.0 / CLOCKS_PER_SEC << " ms" << std::endl;

    /** Compare the results. */
    for( unsigned int i = 0; i < u.size(); ++i )
    {
      double diff = kernel_ITK->Evaluate( u[ i ] ) - kernel_elx->Evaluate( u[ i ] );
      if( diff > maxAllowedDistance )
      {
        std::cerr << "ERROR: our implementation differs from ITK." << std::endl;
        return 1;
      }
    }
    std::cerr << "The results are good.\n" << std::endl;

  } // end for all spline orders

  /** Return a value. */
  return 0;

} // end main
