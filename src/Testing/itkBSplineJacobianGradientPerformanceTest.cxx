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

#include "itkAdvancedBSplineDeformableTransform.h" // original elastix
#include "itkRecursiveBSplineTransform.h"          // recursive version

// Report timings
#include "itkTimeProbe.h"
#include "itkTimeProbesCollectorBase.h"

#include <fstream>
#include <iomanip>

//-------------------------------------------------------------------------------------

int
main( int argc, char * argv[] )
{
  /** Some basic type definitions.
   * NOTE: don't change the dimension or the spline order, since the
   * hard-coded ground truth depends on this.
   */
  const unsigned int Dimension   = 3;
  const unsigned int SplineOrder = 3;
  typedef double CoordinateRepresentationType;

  /** The number of calls to Evaluate(). Distinguish between
   * Debug and Release mode.
   */
#ifndef NDEBUG
  unsigned int N = static_cast< unsigned int >( 1e3 );
#else
  unsigned int N = static_cast< unsigned int >( 1e5 );
#endif
  std::cerr << "N = " << N << std::endl;

  /** Check. */
  if( argc != 2 )
  {
    std::cerr << "ERROR: You should specify a text file with the B-spline "
              << "transformation parameters." << std::endl;
    return 1;
  }

  /** Typedefs. */
  typedef itk::AdvancedBSplineDeformableTransform<
    CoordinateRepresentationType, Dimension, SplineOrder >    TransformType;
  typedef itk::RecursiveBSplineTransform<
    CoordinateRepresentationType, Dimension, SplineOrder >    RecursiveTransformType;

  typedef TransformType::NumberOfParametersType     NumberOfParametersType;
  typedef TransformType::InputPointType             InputPointType;
  typedef TransformType::ParametersType             ParametersType;
  typedef TransformType::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;
  typedef TransformType::DerivativeType             DerivativeType;
  typedef TransformType::JacobianType               JacobianType;
  typedef TransformType::MovingImageGradientType    MovingImageGradientType;

  typedef itk::Image< CoordinateRepresentationType,
    Dimension >                                         InputImageType;
  typedef InputImageType::RegionType    RegionType;
  typedef InputImageType::SizeType      SizeType;
  typedef InputImageType::IndexType     IndexType;
  typedef InputImageType::SpacingType   SpacingType;
  typedef InputImageType::PointType     OriginType;
  typedef InputImageType::DirectionType DirectionType;

  /** Create the transform. */
  TransformType::Pointer          transform          = TransformType::New();
  RecursiveTransformType::Pointer recursiveTransform = RecursiveTransformType::New();

  /** Setup the B-spline transform:
   * (GridSize 44 43 35)
   * (GridIndex 0 0 0)
   * (GridSpacing 10.7832773148 11.2116431394 11.8648235177)
   * (GridOrigin -237.6759555555 -239.9488431747 -344.2315805162)
   */
  SizeType gridSize;
  gridSize[ 0 ] = 44; gridSize[ 1 ] = 43; gridSize[ 2 ] = 35;
  IndexType gridIndex;
  gridIndex.Fill( 0 );
  RegionType gridRegion;
  gridRegion.SetSize( gridSize );
  gridRegion.SetIndex( gridIndex );
  SpacingType gridSpacing;
  gridSpacing[ 0 ] = 10.7832773148;
  gridSpacing[ 1 ] = 11.2116431394;
  gridSpacing[ 2 ] = 11.8648235177;
  OriginType gridOrigin;
  gridOrigin[ 0 ] = -237.6759555555;
  gridOrigin[ 1 ] = -239.9488431747;
  gridOrigin[ 2 ] = -344.2315805162;
  DirectionType gridDirection;
  gridDirection.SetIdentity();

  transform->SetGridOrigin( gridOrigin );
  transform->SetGridSpacing( gridSpacing );
  transform->SetGridRegion( gridRegion );
  transform->SetGridDirection( gridDirection );

  recursiveTransform->SetGridOrigin( gridOrigin );
  recursiveTransform->SetGridSpacing( gridSpacing );
  recursiveTransform->SetGridRegion( gridRegion );
  recursiveTransform->SetGridDirection( gridDirection );

  /** Now read the parameters as defined in the file par.txt. */
  ParametersType parameters( transform->GetNumberOfParameters() );
  std::ifstream  input( argv[ 1 ] );
  if( input.is_open() )
  {
    for( unsigned int i = 0; i < parameters.GetSize(); ++i )
    {
      input >> parameters[ i ];
    }
  }
  else
  {
    std::cerr << "ERROR: could not open the text file containing the "
              << "parameter values." << std::endl;
    return 1;
  }
  transform->SetParameters( parameters );
  recursiveTransform->SetParameters( parameters );

  /** Declare variables. */
  InputPointType          inputPoint; inputPoint.Fill( 4.1 );
  MovingImageGradientType movingImageGradient;
  movingImageGradient[ 0 ] = 29.43; movingImageGradient[ 1 ] = 18.21; movingImageGradient[ 2 ] = 1.7;
  const NumberOfParametersType nnzji = transform->GetNumberOfNonZeroJacobianIndices();
  JacobianType                 jacobian( Dimension, nnzji );
  DerivativeType               imageJacobian_old( nnzji );
  DerivativeType               imageJacobian_new( nnzji );
  DerivativeType               imageJacobian_recursive( nnzji );
  NonZeroJacobianIndicesType   nzji( nnzji );
  itk::TimeProbesCollectorBase timeCollector;
  double                       sum = 0.0;

  /** Time the plain old way. */
  timeCollector.Start( "JacobianGradient plain old" );
  for( unsigned int i = 0; i < N; ++i )
  {
    /** Get the TransformJacobian dT/dmu. */
    transform->GetJacobian( inputPoint, jacobian, nzji );

    /** Compute the inner products (dM/dx)^T (dT/dmu). */
    const unsigned int numberOfParametersPerDimension = nnzji / Dimension;
    unsigned int       counter                        = 0;
    for( unsigned int dim = 0; dim < Dimension; ++dim )
    {
      const double imDeriv = movingImageGradient[ dim ];
      for( unsigned int mu = 0; mu < numberOfParametersPerDimension; ++mu )
      {
        imageJacobian_old( counter ) = jacobian( dim, counter ) * imDeriv;
        ++counter;
      }
    }

    sum += imageJacobian_old( 0 ); // just to avoid compiler to optimize away
  }
  timeCollector.Stop( "JacobianGradient plain old" );

  /** Time the plain new way. */
  timeCollector.Start( "JacobianGradient plain new" );
  for( unsigned int i = 0; i < N; ++i )
  {
    /** Compute the inner product of the transform Jacobian dT/dmu and the moving image gradient dM/dx. */
    transform->EvaluateJacobianWithImageGradientProduct(
      inputPoint, movingImageGradient,
      imageJacobian_new, nzji );

    sum += imageJacobian_new( 0 ); // just to avoid compiler to optimize away
  }
  timeCollector.Stop( "JacobianGradient plain new" );

  /** Time the recursive old way. */
  timeCollector.Start( "JacobianGradient recursive old" );
  for( unsigned int i = 0; i < N; ++i )
  {
    /** Compute the inner product of the transform Jacobian dT/dmu and the moving image gradient dM/dx. */
    recursiveTransform->GetJacobian( inputPoint, jacobian, nzji );

    /** Compute the inner products (dM/dx)^T (dT/dmu). */
    const unsigned int numberOfParametersPerDimension = nnzji / Dimension;
    unsigned int       counter                        = 0;
    for( unsigned int dim = 0; dim < Dimension; ++dim )
    {
      const double imDeriv = movingImageGradient[ dim ];
      for( unsigned int mu = 0; mu < numberOfParametersPerDimension; ++mu )
      {
        imageJacobian_old( counter ) = jacobian( dim, counter ) * imDeriv;
        ++counter;
      }
    }

    sum += imageJacobian_old( 0 ); // just to avoid compiler to optimize away
  }
  timeCollector.Stop( "JacobianGradient recursive old" );

  /** Time the recursive new way. */
  timeCollector.Start( "JacobianGradient recursive new" );
  for( unsigned int i = 0; i < N; ++i )
  {
    /** Compute the inner product of the transform Jacobian dT/dmu and the moving image gradient dM/dx. */
    recursiveTransform->EvaluateJacobianWithImageGradientProduct(
      inputPoint, movingImageGradient,
      imageJacobian_new, nzji );

    sum += imageJacobian_new( 0 ); // just to avoid compiler to optimize away
  }
  timeCollector.Stop( "JacobianGradient recursive new" );

  /** Report timings. */
  timeCollector.Report();

  // Avoid compiler optimizations, so use sum
  std::cerr << sum << std::endl; // works but ugly on screen

  /**
   *
   * Test accuracy
   *
   */

  transform->EvaluateJacobianWithImageGradientProduct(
    inputPoint, movingImageGradient,
    imageJacobian_old, nzji );

  recursiveTransform->EvaluateJacobianWithImageGradientProduct(
    inputPoint, movingImageGradient,
    imageJacobian_new, nzji );

  double diffNorm = ( imageJacobian_old - imageJacobian_new ).magnitude();
  std::cerr << "Recursive B-spline MSD with previous: " << diffNorm << std::endl;
  if( diffNorm > 1e-5 )
  {
    std::cerr << "ERROR: Recursive B-spline EvaluateJacobianWithImageGradientProduct() returning incorrect result." << std::endl;
    return EXIT_FAILURE;
  }

  /** Return a value. */
  return EXIT_SUCCESS;

} // end main
