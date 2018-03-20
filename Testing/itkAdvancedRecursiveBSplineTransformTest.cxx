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
#include "itkBSplineDeformableTransform.h"         // original ITK
#include "itkAdvancedBSplineDeformableTransform.h" // original elastix
#include "itkRecursiveBSplineTransform.h"          // recursive version
//#include "itkBSplineTransform.h"                   // new ITK4

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
  typedef double CoordinateRepresentationType;
  //const double distance = 1e-3; // the allowable distance
  //const double allowedTimeDifference = 0.1; // 10% is considered within limits

  /** The number of calls to Evaluate(). Distinguish between
   * Debug and Release mode.
   */
#ifndef NDEBUG
  unsigned int N = static_cast< unsigned int >( 1 );
#else
  unsigned int N = static_cast< unsigned int >( 1e5 );
#endif
  std::cerr << "N = " << N << std::endl;

  /** Check. */
  if( argc != 2 )
  {
    std::cerr << "ERROR: You should specify a text file with the B-spline "
              << "transformation parameters." << std::endl;
    return EXIT_FAILURE;
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
  typedef TransformType::ImagePointer                  CoefficientImagePointer;
  typedef itk::Image< CoordinateRepresentationType,
    Dimension >                                         InputImageType;
  typedef InputImageType::RegionType                             RegionType;
  typedef InputImageType::SizeType                               SizeType;
  typedef InputImageType::IndexType                              IndexType;
  typedef InputImageType::SpacingType                            SpacingType;
  typedef InputImageType::PointType                              OriginType;
  typedef InputImageType::DirectionType                          DirectionType;
  typedef itk::Statistics::MersenneTwisterRandomVariateGenerator MersenneTwisterType;

  /** Create the transforms. */
  ITKTransformType::Pointer       transformITK = ITKTransformType::New();
  TransformType::Pointer          transform    = TransformType::New();
  RecursiveTransformType::Pointer recursiveTransform
    = RecursiveTransformType::New();

  /** Setup the B-spline transform:
   * (GridSize 44 43 35)
   * (GridIndex 0 0 0)
   * (GridSpacing 10.7832773148 11.2116431394 11.8648235177)
   * (GridOrigin -237.6759555555 -239.9488431747 -344.2315805162)
   */
  std::ifstream input( argv[ 1 ] );
  if( !input.is_open() )
  {
    std::cerr << "ERROR: could not open the text file containing the "
              << "parameter values." << std::endl;
    return EXIT_FAILURE;
  }
  int dimsInPar1;
  input >> dimsInPar1;
  if( dimsInPar1 != Dimension )
  {
    std::cerr << "ERROR: The file containing the parameters specifies "
              << dimsInPar1 << " dimensions, while this test is compiled for "
              << Dimension << " dimensions." << std::endl;
    return EXIT_FAILURE;
  }

  SizeType gridSize;
  for( unsigned int i = 0; i < Dimension; ++i )
  {
    input >> gridSize[ i ];
    std::cerr << "Gridsize dimension " << i << " = " << gridSize[ i ] << std::endl;
  }
  //gridSize[ 0 ] = 44; gridSize[ 1 ] = 43; gridSize[ 2 ] = 35;
  //gridSize[ 0 ] = 68; gridSize[ 1 ] = 69; gridSize[ 2 ] = 64;

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
  gridDirection( 0, 1 ) = 0.02; gridDirection( 0, 2 ) = 0.06;
  gridDirection( 1, 0 ) = 0.03; gridDirection( 1, 2 ) = 0.07;
  gridDirection( 2, 0 ) = 0.09; gridDirection( 2, 1 ) = 0.01;

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
  std::cerr << "Loading parameters from file 1";
  ParametersType parameters( transform->GetNumberOfParameters() );

  for( unsigned int i = 0; i < parameters.GetSize(); ++i )
  {
    input >> parameters[ i ];
  }
  transformITK->SetParameters( parameters );
  transform->SetParameters( parameters );
  recursiveTransform->SetParameters( parameters );
  std::cerr <<  " - done.\n" << std::endl;

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
  NonZeroJacobianIndicesType    nzji, nzjiElastix, nzjiRecursive;

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

  /** The Jacobian. */
  recursiveTransform->GetJacobian( inputPoint, jacobian, nzji );

  /** The spatial Jacobian. */
  recursiveTransform->GetSpatialJacobian( inputPoint, spatialJacobian );

  /** The spatial Hessian. */
  recursiveTransform->GetSpatialHessian( inputPoint, spatialHessian );

  /** The Jacobian of the spatial Jacobian. */
  recursiveTransform->GetJacobianOfSpatialJacobian( inputPoint,
    jacobianOfSpatialJacobian, nzji );

  recursiveTransform->GetJacobianOfSpatialJacobian( inputPoint,
    spatialJacobian, jacobianOfSpatialJacobian, nzji );

  /** The Jacobian of the spatial Hessian. */
  recursiveTransform->GetJacobianOfSpatialHessian( inputPoint,
    jacobianOfSpatialHessian, nzji );

  recursiveTransform->GetJacobianOfSpatialHessian( inputPoint,
    spatialHessian, jacobianOfSpatialHessian, nzji );

  /**
   *
   * Test timing
   *
   */

  itk::TimeProbesCollectorBase                    timeCollector;
  OutputPointType                                 opp;
  TransformType::WeightsType                      weights;
  RecursiveTransformType::WeightsType             weights2;
  TransformType::ParameterIndexArrayType          indices;
  RecursiveTransformType::ParameterIndexArrayType indices2;

  const unsigned int dummyNum = vcl_pow( static_cast< double >( SplineOrder + 1 ), static_cast< double >( Dimension ) );
  weights.SetSize( dummyNum );
  indices.SetSize( dummyNum );
  weights2.SetSize( dummyNum );
  indices2.SetSize( dummyNum );

  // Generate a list of random points
  MersenneTwisterType::Pointer mersenneTwister = MersenneTwisterType::New();
  mersenneTwister->Initialize( 140377 );
  std::vector< InputPointType >  pointList( N );
  std::vector< OutputPointType > transformedPointList1( N );
  std::vector< OutputPointType > transformedPointList2( N );

  IndexType               dummyIndex;
  CoefficientImagePointer coefficientImage = transform->GetCoefficientImages()[ 0 ];
  for( unsigned int i = 0; i < N; ++i )
  {
    for( unsigned int j = 0; j < Dimension; ++j )
    {
      dummyIndex[ j ] = mersenneTwister->GetUniformVariate( 2, gridSize[ j ] - 3 );
    }
    coefficientImage->TransformIndexToPhysicalPoint( dummyIndex, pointList[ i ] );
  }

  /** Time the implementation of the TransformPoint. */
  timeCollector.Start( "TransformPoint elastix           " );
  for( unsigned int i = 0; i < N; ++i )
  {
    transformedPointList1[ i ] = transform->TransformPoint( pointList[ i ] );
  }
  timeCollector.Stop(  "TransformPoint elastix           " );

  timeCollector.Start( "TransformPoint recursive         " );
  for( unsigned int i = 0; i < N; ++i )
  {
    transformedPointList2[ i ] = recursiveTransform->TransformPoint( pointList[ i ] );
  }
  timeCollector.Stop(  "TransformPoint recursive         " );

  /** Time the implementation of the Jacobian. */
  timeCollector.Start( "Jacobian elastix                 " );
  for( unsigned int i = 0; i < N; ++i )
  {
    transform->GetJacobian( pointList[ i ], jacobian, nzji );
  }
  timeCollector.Stop(  "Jacobian elastix                 " );

  timeCollector.Start( "Jacobian recursive               " );
  for( unsigned int i = 0; i < N; ++i )
  {
    recursiveTransform->GetJacobian( pointList[ i ], jacobian, nzji );
  }
  timeCollector.Stop(  "Jacobian recursive               " );

  /** Time the implementation of the spatial Jacobian. */
  SpatialJacobianType sj, sjRecursive;
  timeCollector.Start( "SpatialJacobian elastix          " );
  for( unsigned int i = 0; i < N; ++i )
  {
    transform->GetSpatialJacobian( pointList[ i ], sj );
  }
  timeCollector.Stop(  "SpatialJacobian elastix          " );

  timeCollector.Start( "SpatialJacobian recursive        " );
  for( unsigned int i = 0; i < N; ++i )
  {
    recursiveTransform->GetSpatialJacobian( pointList[ i ], sjRecursive );
  }
  timeCollector.Stop(  "SpatialJacobian recursive        " );

  /** Time the implementation of the spatial Hessian. */
  SpatialHessianType sh, shRecursive;
  timeCollector.Start( "SpatialHessian elastix           " );
  for( unsigned int i = 0; i < N; ++i )
  {
    transform->GetSpatialHessian( pointList[ i ], sh );
  }
  timeCollector.Stop(  "SpatialHessian elastix           " );

  timeCollector.Start( "SpatialHessian recursive         " );
  for( unsigned int i = 0; i < N; ++i )
  {
    recursiveTransform->GetSpatialHessian( pointList[ i ], shRecursive );
  }
  timeCollector.Stop(  "SpatialHessian recursive         " );

  /** Time the implementation of the Jacobian of the spatial Jacobian. */
  JacobianOfSpatialJacobianType jsj, jsjRecursive;
  timeCollector.Start( "JacobianSpatialJacobian elastix  " );
  for( unsigned int i = 0; i < N; ++i )
  {
    transform->GetJacobianOfSpatialJacobian( pointList[ i ], jsj, nzji );
  }
  timeCollector.Stop(  "JacobianSpatialJacobian elastix  " );

  timeCollector.Start( "JacobianSpatialJacobian recursive " );
  for( unsigned int i = 0; i < N; ++i )
  {
    recursiveTransform->GetJacobianOfSpatialJacobian( pointList[ i ], jsjRecursive, nzji );
  }
  timeCollector.Stop(  "JacobianSpatialJacobian recursive " );

  /** Time the implementation of the Jacobian of the spatial Hessian. */
  JacobianOfSpatialHessianType jsh, jshRecursive;
  timeCollector.Start( "JacobianSpatialHessian elastix   " );
  for( unsigned int i = 0; i < N; ++i )
  {
    transform->GetJacobianOfSpatialHessian( pointList[ i ], jsh, nzji );
  }
  timeCollector.Stop(  "JacobianSpatialHessian elastix   " );

  timeCollector.Start( "JacobianSpatialHessian recursive " );
  for( unsigned int i = 0; i < N; ++i )
  {
    recursiveTransform->GetJacobianOfSpatialHessian( pointList[ i ], jshRecursive, nzji );
  }
  timeCollector.Stop(  "JacobianSpatialHessian recursive " );

  /** Time the implementation of the NonZeroJacobianIndices. */
  // Not directly possible, as these are protected functions.

  /** Report. */
  timeCollector.Report();

  /**
   *
   * Test accuracy
   *
   */

  /** These should return the same values as the original ITK functions. */

  /** TransformPoint. */
  OutputPointType opp1, opp2;
  double          differenceNorm1 = 0.0;
  for( unsigned int i = 0; i < N; ++i )
  {
    opp1 = transformedPointList1[ i ]; // transform->TransformPoint();
    opp2 = transformedPointList2[ i ]; // recursiveTransform->TransformPoint();

    for( unsigned int j = 0; j < Dimension; ++j )
    {
      differenceNorm1 += ( opp1[ j ] - opp2[ j ] ) * ( opp1[ j ] - opp2[ j ] );
    }
  }
  differenceNorm1 = vcl_sqrt( differenceNorm1 ) / N;

  std::cerr << "\n" << std::endl;
  std::cerr << "Recursive B-spline TransformPoint() MSD with ITK: " << differenceNorm1 << std::endl;

  if( differenceNorm1 > 1e-5 )
  {
    std::cerr << "ERROR: Recursive B-spline TransformPoint() returning incorrect result." << std::endl;
    return EXIT_FAILURE;
  }

  /** Jacobian. */
  JacobianType jacobianElastix; jacobianElastix.SetSize( Dimension, nzji.size() ); jacobianElastix.Fill( 0.0 );
  transform->GetJacobian( inputPoint, jacobianElastix, nzjiElastix );

  JacobianType jacobianRecursive; jacobianRecursive.SetSize( Dimension, nzji.size() ); jacobianRecursive.Fill( 0.0 );
  recursiveTransform->GetJacobian( inputPoint, jacobianRecursive, nzjiRecursive );

  JacobianType jacobianDifferenceMatrix = jacobianElastix - jacobianRecursive;
  double       jacobianDifference       = jacobianDifferenceMatrix.frobenius_norm();
  std::cerr << "The Recursive B-spline GetJacobian() difference is " << jacobianDifference << std::endl;
  if( jacobianDifference > 1e-10 )
  {
    std::cerr << "ERROR: Recursive B-spline GetJacobian() returning incorrect result." << std::endl;
    return EXIT_FAILURE;
  }

  /** NonZeroJacobianIndices. */
  double nzjiDifference = 0.0;
  for( unsigned int i = 0; i < nzjiElastix.size(); ++i )
  {
    nzjiDifference += ( nzjiElastix[ i ] - nzjiRecursive[ i ] ) * ( nzjiElastix[ i ] - nzjiRecursive[ i ] );
  }
  nzjiDifference = std::sqrt( nzjiDifference );
  std::cerr << "The Recursive B-spline ComputeNonZeroJacobianIndices() difference is " << nzjiDifference << std::endl;
  if( nzjiDifference > 1e-10 )
  {
    std::cerr << "ERROR: Recursive B-spline ComputeNonZeroJacobianIndices() returning incorrect result." << std::endl;
    return EXIT_FAILURE;
  }

  /** Spatial Jacobian. */
  transform->GetSpatialJacobian( inputPoint, sj );
  recursiveTransform->GetSpatialJacobian( inputPoint, sjRecursive );

  SpatialJacobianType sjDifferenceMatrix = sj - sjRecursive;
  double              sjDifference       = sjDifferenceMatrix.GetVnlMatrix().frobenius_norm();
  std::cerr << "The Recursive B-spline GetSpatialJacobian() difference is " << sjDifference << std::endl;
  if( sjDifference > 1e-8 )
  {
    std::cerr << "ERROR: Recursive B-spline GetSpatialJacobian() returning incorrect result." << std::endl;
    return EXIT_FAILURE;
  }

  /** Spatial Hessian. */
  transform->GetSpatialHessian( inputPoint, sh );
  recursiveTransform->GetSpatialHessian( inputPoint, shRecursive );

  double shDifference = 0.0;
  for( unsigned int i = 0; i < Dimension; ++i )
  {
    shDifference += ( sh[ i ] - shRecursive[ i ] ).GetVnlMatrix().frobenius_norm();
  }
  std::cerr << "The Recursive B-spline GetSpatialHessian() difference is " << shDifference << std::endl;
  if( shDifference > 1e-8 )
  {
    std::cerr << "ERROR: Recursive B-spline GetSpatialHessian() returning incorrect result." << std::endl;
    return EXIT_FAILURE;
  }

  /** Jacobian of the spatial Jacobian. */
  transform->GetJacobianOfSpatialJacobian( inputPoint, jsj, nzji );
  recursiveTransform->GetJacobianOfSpatialJacobian( inputPoint, jsjRecursive, nzji );

  double jsjDifference = 0.0;
  for( unsigned int i = 0; i < jsj.size(); ++i )
  {
    jsjDifference += ( jsj[ i ] - jsjRecursive[ i ] ).GetVnlMatrix().frobenius_norm();
  }
  std::cerr << "The Recursive B-spline GetJacobianOfSpatialJacobian() difference is " << jsjDifference << std::endl;
  if( jsjDifference > 1e-8 )
  {
    std::cerr << "ERROR: Recursive B-spline GetJacobianOfSpatialJacobian() returning incorrect result." << std::endl;
    return EXIT_FAILURE;
  }

  /** Jacobian of the spatial Hessian. */
  transform->GetJacobianOfSpatialHessian( inputPoint, jsh, nzji );
  recursiveTransform->GetJacobianOfSpatialHessian( inputPoint, jshRecursive, nzji );

  double jshDifference = 0.0;
  for( unsigned int i = 0; i < jsh.size() / Dimension; ++i ) // only test first part
  {
    for( unsigned int j = 0; j < Dimension; ++j )
    {
      jshDifference += ( jsh[ i ][ j ] - jshRecursive[ i ][ j ] ).GetVnlMatrix().frobenius_norm();
    }
  }
  std::cerr << "The Recursive B-spline GetJacobianOfSpatialHessian() difference is " << jshDifference << std::endl;
  if( jshDifference > 1e-8 )
  {
    std::cerr << "ERROR: Recursive B-spline GetJacobianOfSpatialHessian() returning incorrect result." << std::endl;
    return EXIT_FAILURE;
  }

  /** Exercise PrintSelf(). */
  std::cerr << std::endl;
  recursiveTransform->Print( std::cerr );

  /** Return a value. */
  return EXIT_SUCCESS;

} // end main
