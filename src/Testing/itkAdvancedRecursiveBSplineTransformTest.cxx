/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#include "itkBSplineDeformableTransform.h"         // original ITK
#include "itkAdvancedBSplineDeformableTransform.h" // original elastix
#include "itkRecursiveBSplineTransform.h"          // recursive version
//#include "itkBSplineTransform.h" // new ITK4

#include "itkGridScheduleComputer.h"

#include "itkTimeProbe.h"
#include "itkTimeProbesCollectorBase.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
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
  typedef float CoordinateRepresentationType;
  //const double distance = 1e-3; // the allowable distance
  //const double allowedTimeDifference = 0.1; // 10% is considered within limits

  /** The number of calls to Evaluate(). Distinguish between
   * Debug and Release mode.
   */
#ifndef NDEBUG
  //unsigned int N = static_cast< unsigned int >( 1e3 );
  unsigned int N = static_cast< unsigned int >( 0 );
#else
  unsigned int N = static_cast< unsigned int >( 1e6 );
#endif
  std::cerr << "N = " << N << std::endl;

  /** Check. */
  if( argc != 2 )
  {
    std::cerr << "ERROR: You should specify a text file with the B-spline "
              << "transformation parameters." << std::endl;
    return 1;
  }

  /** Other typedefs. */
  typedef itk::BSplineDeformableTransform<
    CoordinateRepresentationType, Dimension, SplineOrder >    ITKTransformType;
  typedef itk::AdvancedBSplineDeformableTransform<
    CoordinateRepresentationType, Dimension, SplineOrder >    TransformType;
  typedef itk::RecursiveBSplineTransform<
    CoordinateRepresentationType, Dimension, SplineOrder >    RecursiveTransformType;
  typedef TransformType::JacobianType                  JacobianType;
  typedef TransformType::SpatialJacobianType           SpatialJacobianType;
  typedef TransformType::SpatialHessianType            SpatialHessianType;
  typedef TransformType::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
  typedef TransformType::JacobianOfSpatialHessianType  JacobianOfSpatialHessianType;
  typedef TransformType::NonZeroJacobianIndicesType    NonZeroJacobianIndicesType;
  typedef TransformType::NumberOfParametersType        NumberOfParametersType;
  typedef TransformType::InputPointType                InputPointType;
  typedef TransformType::OutputPointType               OutputPointType;
  typedef TransformType::ParametersType                ParametersType;
  typedef TransformType::ImagePointer     CoefficientImagePointer;
  typedef itk::Image< CoordinateRepresentationType,
    Dimension >                                         InputImageType;
  typedef InputImageType::RegionType    RegionType;
  typedef InputImageType::SizeType      SizeType;
  typedef InputImageType::IndexType     IndexType;
  typedef InputImageType::SpacingType   SpacingType;
  typedef InputImageType::PointType     OriginType;
  typedef InputImageType::DirectionType DirectionType;
  typedef itk::Statistics::MersenneTwisterRandomVariateGenerator MersenneTwisterType;

  /** Create the transforms. */
  ITKTransformType::Pointer transformITK = ITKTransformType::New();
  TransformType::Pointer    transform    = TransformType::New();
  RecursiveTransformType::Pointer recursiveTransform    = RecursiveTransformType::New();

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

  transformITK->SetGridOrigin( gridOrigin );
  transformITK->SetGridSpacing( gridSpacing );
  transformITK->SetGridRegion( gridRegion );
  transformITK->SetGridDirection( gridDirection );

  transform->SetGridOrigin( gridOrigin );
  transform->SetGridSpacing( gridSpacing );
  transform->SetGridRegion( gridRegion );
  transform->SetGridDirection( gridDirection );

  recursiveTransform->SetGridOrigin( gridOrigin );
  recursiveTransform->SetGridSpacing( gridSpacing );
  recursiveTransform->SetGridRegion( gridRegion );
  recursiveTransform->SetGridDirection( gridDirection );

  //ParametersType fixPar( Dimension * ( 3 + Dimension ) );
  //fixPar[ 0 ] = gridSize[ 0 ]; fixPar[ 1 ] = gridSize[ 1 ]; fixPar[ 2 ] = gridSize[ 2 ];
  //fixPar[ 3 ] = gridOrigin[ 0 ]; fixPar[ 4 ] = gridOrigin[ 1 ]; fixPar[ 5 ] = gridOrigin[ 2 ];
  //fixPar[ 6 ] = gridSpacing[ 0 ]; fixPar[ 7 ] = gridSpacing[ 1 ]; fixPar[ 8 ] = gridSpacing[ 2 ];
  //fixPar[ 9 ] = gridDirection[ 0 ][ 0 ]; fixPar[ 10 ] = gridDirection[ 0 ][ 1 ]; fixPar[ 11 ] = gridDirection[ 0 ][ 2 ];
  //fixPar[ 12 ] = gridDirection[ 1 ][ 0 ]; fixPar[ 13 ] = gridDirection[ 1 ][ 1 ]; fixPar[ 14 ] = gridDirection[ 1 ][ 2 ];
  //fixPar[ 15 ] = gridDirection[ 2 ][ 0 ]; fixPar[ 16 ] = gridDirection[ 2 ][ 1 ]; fixPar[ 17 ] = gridDirection[ 2 ][ 2 ];
  //transformITK->SetFixedParameters( fixPar );

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
  transformITK->SetParameters( parameters );
  transform->SetParameters( parameters );
  recursiveTransform->SetParameters( parameters );  

  /** Get the number of nonzero Jacobian indices. */
  const NumberOfParametersType nonzji = transform->GetNumberOfNonZeroJacobianIndices();

  /** Declare variables. */
  InputPointType inputPoint;
  inputPoint.Fill( 4.1 );
  JacobianType                  jacobian;
  SpatialJacobianType           spatialJacobian;
  SpatialHessianType            spatialHessian;
  JacobianOfSpatialJacobianType jacobianOfSpatialJacobian;
  JacobianOfSpatialHessianType  jacobianOfSpatialHessian;
  NonZeroJacobianIndicesType    nzji;

  /** Resize some of the variables. */
  nzji.resize( nonzji );
  jacobian.SetSize( Dimension, nonzji );
  jacobianOfSpatialJacobian.resize( nonzji );
  jacobianOfSpatialHessian.resize( nonzji );
  jacobian.Fill( 0.0 );

  /**
   *
   * Call functions for testing that they don't crash.
   *
   */

  /** The transform point. */
  recursiveTransform->TransformPoint( inputPoint );

  /** The Jacobian. *
  recursiveTransform->GetJacobian( inputPoint, jacobian, nzji );

  /** The spatial Jacobian. */
  //recursiveTransform->GetSpatialJacobian( inputPoint, spatialJacobian );
  // crashes

  /** The spatial Hessian. *
  recursiveTransform->GetSpatialHessian( inputPoint, spatialHessian );

  /** The Jacobian of the spatial Jacobian. *
  recursiveTransform->GetJacobianOfSpatialJacobian( inputPoint,
    jacobianOfSpatialJacobian, nzji );

  recursiveTransform->GetJacobianOfSpatialJacobian( inputPoint,
    spatialJacobian, jacobianOfSpatialJacobian, nzji );

  /** The Jacobian of the spatial Hessian. *
  recursiveTransform->GetJacobianOfSpatialHessian( inputPoint,
    jacobianOfSpatialHessian, nzji );

  recursiveTransform->GetJacobianOfSpatialHessian( inputPoint,
    spatialHessian, jacobianOfSpatialHessian, nzji );

  /**
   *
   * Test timing
   *
   */

  itk::TimeProbesCollectorBase timeCollector;
  OutputPointType opp;
  TransformType::WeightsType				        weights;
  RecursiveTransformType::WeightsType				weights2;
  TransformType::ParameterIndexArrayType	  indices;
  RecursiveTransformType::ParameterIndexArrayType	indices2;

  const unsigned int dummyNum = vcl_pow( static_cast< double >( SplineOrder + 1 ),static_cast< double >( Dimension ) );
  weights.SetSize( dummyNum );
  indices.SetSize( dummyNum );
  weights2.SetSize( dummyNum );
  indices2.SetSize( dummyNum );
  
  bool isInside = true;

  // Generate a list of random points
  MersenneTwisterType::Pointer mersenneTwister = MersenneTwisterType::New();
  mersenneTwister->Initialize( 140377 );
  std::vector< InputPointType > pointList( N );
  IndexType dummyIndex;
  CoefficientImagePointer coefficientImage = transform->GetCoefficientImages()[0];
  for( unsigned int i = 0; i < N; ++i )
  {
    for( unsigned int j = 0; j < Dimension; ++j )
    {
      dummyIndex[ j ] = mersenneTwister->GetUniformVariate( 1, gridSize[ j ] - 2 );
    }
    coefficientImage->TransformIndexToPhysicalPoint( dummyIndex, pointList[ i ] );
  }

  /** Time the implementation of the TransformPoint. */
  timeCollector.Start( "TransformPoint elastix" );
  for( unsigned int i = 0; i < N; ++i )
  {
    //OutputPointType opp1 = transform->TransformPoint( inputPoint );
    transform->TransformPoint( pointList[ i ], opp, weights, indices, isInside );

    if( isInside == false ){ printf("error point is not in the image"); } // Just to make sure the previous is not optimized away
  }
  timeCollector.Stop( "TransformPoint elastix" );

  timeCollector.Start( "TransformPoint recursive" );
  for( unsigned int i = 0; i < N; ++i )
  {
    //OutputPointType opp2 = recursiveTransform->TransformPoint( inputPoint );
    recursiveTransform->TransformPoint( pointList[ i ], opp, weights2, indices2, isInside );

    if( isInside == false ){ printf("error point is not in the image"); } // Just to make sure the previous is not optimized away
  }
  timeCollector.Stop( "TransformPoint recursive" );

  timeCollector.Start( "TransformPoint recursive vector" );
  for( unsigned int i = 0; i < N; ++i )
  {
    //OutputPointType opp3 = recursiveTransform->TransformPointVector( inputPoint );
    recursiveTransform->TransformPointVector( pointList[ i ], opp, weights2, indices2, isInside );

    if( isInside == false ){ printf("error point is not in the image"); } // Just to make sure the previous is not optimized away
  }
  timeCollector.Stop( "TransformPoint recursive vector" );

  /** Time the implementation of the Jacobian. *
  timeCollector.Start( "Jacobian elastix" );
  for( unsigned int i = 0; i < N; ++i )
  {
    transform->GetJacobian( inputPoint, jacobian, nzji );
  }
  timeCollector.Stop( "Jacobian elastix" );

  timeCollector.Start( "Jacobian recursive" );
  for( unsigned int i = 0; i < N; ++i )
  {
    recursiveTransform->GetJacobian( inputPoint, jacobian, nzji );
  }
  timeCollector.Stop( "Jacobian recursive" );*/

  timeCollector.Report();

  /**
   *
   * Test accuracy
   *
   */

  /** These should return the same values as the original ITK functions. */

  /** TransformPoint. */
  OutputPointType opp1, opp2, opp3;
  double differenceNorm1 = 0.0;
  double differenceNorm2 = 0.0;
  double differenceNorm3 = 0.0;
  for( unsigned int i = 0; i < N; ++i )
  {
    transform->TransformPoint( pointList[ i ], opp1, weights, indices, isInside );
    recursiveTransform->TransformPoint( pointList[ i ], opp2, weights, indices, isInside );
    recursiveTransform->TransformPointVector( pointList[ i ], opp3, weights, indices, isInside );

    for( unsigned int i = 0; i < Dimension; ++i )
    {
      differenceNorm1 += ( opp1[ i ] - opp2[ i ] ) * ( opp1[ i ] - opp2[ i ] );
      differenceNorm2 += ( opp1[ i ] - opp3[ i ] ) * ( opp1[ i ] - opp3[ i ] );
      differenceNorm3 += ( opp2[ i ] - opp3[ i ] ) * ( opp2[ i ] - opp3[ i ] );
    }
    differenceNorm1 = vcl_sqrt( differenceNorm1 );
    differenceNorm2 = vcl_sqrt( differenceNorm2 );
    differenceNorm3 = vcl_sqrt( differenceNorm3 );
  }
  differenceNorm1 /= N; differenceNorm2 /= N; differenceNorm3 /= N;
  std::cerr << "Recursive B-spline TransformPoint() MSD with ITK: " << differenceNorm1 << std::endl;
  std::cerr << "Recursive B-spline TransformPointVector() MSD with ITK: " << differenceNorm2 << std::endl;
  std::cerr << "Recursive B-spline MSD with itself: " << differenceNorm3 << std::endl;
  if( differenceNorm1 > 1e-5 )
  {
    std::cerr << "ERROR: Recursive B-spline TransformPoint() returning incorrect result." << std::endl;
    return EXIT_FAILURE;
  }
  if( differenceNorm2 > 1e-5 )
  {
    std::cerr << "ERROR: Recursive B-spline TransformPointVector() returning incorrect result." << std::endl;
    return EXIT_FAILURE;
  }

  /** Jacobian */
  //JacobianType jacobianITK; jacobianITK.Fill( 0.0 );
  //transformITK->ComputeJacobianWithRespectToParameters( inputPoint, jacobianITK );
#if 1
  JacobianType jacobianElastix; jacobianElastix.SetSize( Dimension, nzji.size() ); jacobianElastix.Fill( 0.0 );
  transform->GetJacobian( inputPoint, jacobianElastix, nzji );
  for( unsigned int i = 0; i < 64; ++i ){ std::cout << jacobianElastix[ 0 ][ i ] << " "; }
  std::cout << "\n" << std::endl;

  JacobianType jacobianRecursive; jacobianRecursive.SetSize( Dimension, nzji.size() ); jacobianRecursive.Fill( 0.0 );
  recursiveTransform->GetJacobian( inputPoint, jacobianRecursive, nzji );
  for( unsigned int i = 0; i < 64; ++i ){ std::cout << jacobianRecursive[ 0 ][ i ] << " "; }
  std::cout << "\n" << std::endl;

  // 
  JacobianType jacobianDifferenceMatrix = jacobianElastix - jacobianRecursive;
  if( jacobianDifferenceMatrix.frobenius_norm() > 1e-10 )
  {
    std::cerr << "ERROR: Recursive B-spline GetJacobian() returning incorrect result." << std::endl;
    return EXIT_FAILURE;
  }
#endif
  /** Exercise PrintSelf(). */
  //recursiveTransform->Print( std::cerr );

  /** Return a value. */
  return 0;

} // end main
