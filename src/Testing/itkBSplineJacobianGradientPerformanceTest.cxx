/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#include "itkAdvancedBSplineDeformableTransform.h"

// Report timings
#include "itkTimeProbe.h"

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

  typedef TransformType::NumberOfParametersType NumberOfParametersType;
  typedef TransformType::InputPointType         InputPointType;
  typedef TransformType::OutputPointType        OutputPointType;
  typedef TransformType::ParametersType         ParametersType;
  typedef TransformType::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;
  typedef TransformType::DerivativeType         DerivativeType;
  typedef TransformType::JacobianType           JacobianType;
  typedef TransformType::MovingImageGradientType  MovingImageGradientType;

  typedef itk::Image< CoordinateRepresentationType,
    Dimension >                                         InputImageType;
  typedef InputImageType::RegionType    RegionType;
  typedef InputImageType::SizeType      SizeType;
  typedef InputImageType::IndexType     IndexType;
  typedef InputImageType::SpacingType   SpacingType;
  typedef InputImageType::PointType     OriginType;
  typedef InputImageType::DirectionType DirectionType;

  /** Create the transform. */
  TransformType::Pointer transform = TransformType::New();

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

  /** Declare variables. */
  InputPointType  inputPoint; inputPoint.Fill( 4.1 );
  MovingImageGradientType movingImageGradient;
  movingImageGradient[0] = 29.43; movingImageGradient[0] = 18.21; movingImageGradient[0] = 1.7;
  const NumberOfParametersType nnzji = transform->GetNumberOfNonZeroJacobianIndices();
  JacobianType   jacobian( Dimension, nnzji );
  DerivativeType imageJacobian_old( nnzji );
  DerivativeType imageJacobian_new( nnzji );
  NonZeroJacobianIndicesType nzji( nnzji );
  itk::TimeProbe timeProbeOLD, timeProbeNEW;
  double sum = 0.0;

  /** Time the old way. */
  timeProbeOLD.Start();
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

    sum += imageJacobian_old(0); // just to avoid compiler to optimize away
  }
  timeProbeOLD.Stop();
  const double oldTime = timeProbeOLD.GetMean();

  /** Time the new way. */
  timeProbeNEW.Start();
  for( unsigned int i = 0; i < N; ++i )
  {
    /** Compute the inner product of the transform Jacobian dT/dmu and the moving image gradient dM/dx. */
    transform->EvaluateJacobianWithImageGradientProduct(
      inputPoint, movingImageGradient,
      imageJacobian_new, nzji );

    sum += imageJacobian_new(0); // just to avoid compiler to optimize away
  }
  timeProbeNEW.Stop();
  const double newTime = timeProbeNEW.GetMean();

  /** Report timings. */
  std::cerr << std::setprecision( 4 );
  std::cerr << "Time OLD = " << oldTime << " " << timeProbeOLD.GetUnit() << std::endl;
  std::cerr << "Time NEW = " << newTime << " " << timeProbeNEW.GetUnit() << std::endl;
  std::cerr << "Speedup factor = " << oldTime / newTime << std::endl;

  // Avoid compiler optimizations, so use sum
  std::cerr << sum << std::endl; // works but ugly on screen

  /** Return a value. */
  return 0;

} // end main
