/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#include "itkBSplineDerivativeKernelFunction.h"
#include "itkBSplineDerivativeKernelFunction2.h"
#include "itkKernelFunction.h"

#include <ctime>
#include <iomanip>

//-------------------------------------------------------------------------------------

int main( int argc, char *argv[] )
{
  /** Some basic type definitions. */
  std::vector<unsigned int> splineOrders;
  splineOrders.push_back( 1 );
  splineOrders.push_back( 2 );
  splineOrders.push_back( 3 );

  /** The number of calls to Evaluate(). This number gives reasonably
   * fast test results in Release mode.
   */
  unsigned int N = static_cast<unsigned int>( 1e7 );
  const double maxAllowedDistance = 1e-5; // the allowable distance

  /** Other typedefs. */
  typedef itk::KernelFunction                       BaseKernelType;
  typedef itk::BSplineDerivativeKernelFunction<1>   KernelType_ITK_1;
  typedef itk::BSplineDerivativeKernelFunction2<1>  KernelType_elx_1;
  typedef itk::BSplineDerivativeKernelFunction<2>   KernelType_ITK_2;
  typedef itk::BSplineDerivativeKernelFunction2<2>  KernelType_elx_2;
  typedef itk::BSplineDerivativeKernelFunction<3>   KernelType_ITK_3;
  typedef itk::BSplineDerivativeKernelFunction2<3>  KernelType_elx_3;

  /** Create the evaluation points. */
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

  /** For all spline orders. */
  for ( unsigned int so = 0; so < splineOrders.size(); so++ )
  {
    std::cerr << "Evaluating spline order " << splineOrders[ so ] << std::endl;

    /** Create the kernel. */
    BaseKernelType::Pointer kernel_ITK, kernel_elx;
    if ( splineOrders[ so ] == 1 )
    {
      kernel_ITK = KernelType_ITK_1::New();
      kernel_elx = KernelType_elx_1::New();
    }
    else if ( splineOrders[ so ] == 2 )
    {
      kernel_ITK = KernelType_ITK_2::New();
      kernel_elx = KernelType_elx_2::New();
    }
    else if ( splineOrders[ so ] == 3 )
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

    /** Time the ITK implementation. */
    clock_t startClock = clock();
    for ( unsigned int i = 0; i < N; ++i )
    {
      for ( unsigned int j = 0; j < size_u; j++ )
      {
        kernel_ITK->Evaluate( u[ j ] );
      }
    }
    clock_t endClock = clock();
    clock_t clockDiff = endClock - startClock;
    std::cerr << "The elapsed time for ITK implementation is: "
      << clockDiff << " ms" << std::endl;

    /** Time the elx implementation. */
    startClock = clock();
    for ( unsigned int i = 0; i < N; ++i )
    {
      for ( unsigned int j = 0; j < size_u; j++ )
      {
        kernel_elx->Evaluate( u[ j ] );
      }
    }
    endClock = clock();
    clockDiff = endClock - startClock;
    std::cerr << "The elapsed time for the elastix implementation is: "
      << clockDiff << " ms"  << std::endl;

    /** Compare the results. */
    for ( unsigned int i = 0; i < size_u; ++i )
    {
      double diff = kernel_ITK->Evaluate( u[ i ] ) - kernel_elx->Evaluate( u[ i ] );
      if ( diff > maxAllowedDistance )
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
