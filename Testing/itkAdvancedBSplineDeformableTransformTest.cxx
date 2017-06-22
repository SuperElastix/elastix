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
#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkBSplineDeformableTransform.h" // original ITK
//#include "itkBSplineTransform.h" // new ITK4
#include "itkGridScheduleComputer.h"

#include <ctime>
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
  unsigned int N = static_cast< unsigned int >( 1e3 );
#else
  unsigned int N = static_cast< unsigned int >( 0.5e5 );
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
  typedef itk::AdvancedBSplineDeformableTransform<
    CoordinateRepresentationType, Dimension, SplineOrder >    TransformType;
  typedef itk::BSplineDeformableTransform<
    //typedef itk::BSplineTransform<
    CoordinateRepresentationType, Dimension, SplineOrder >    ITKTransformType;
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
  typedef itk::Image< CoordinateRepresentationType,
    Dimension >                                         InputImageType;
  typedef InputImageType::RegionType    RegionType;
  typedef InputImageType::SizeType      SizeType;
  typedef InputImageType::IndexType     IndexType;
  typedef InputImageType::SpacingType   SpacingType;
  typedef InputImageType::PointType     OriginType;
  typedef InputImageType::DirectionType DirectionType;

  /** Create the transform. */
  TransformType::Pointer    transform    = TransformType::New();
  ITKTransformType::Pointer transformITK = ITKTransformType::New();

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

  transformITK->SetGridOrigin( gridOrigin );
  transformITK->SetGridSpacing( gridSpacing );
  transformITK->SetGridRegion( gridRegion );
  transformITK->SetGridDirection( gridDirection );

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
  transform->SetParameters( parameters );
  transformITK->SetParameters( parameters );

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

  /** The Jacobian. */
  transform->GetJacobian( inputPoint, jacobian, nzji );

  /** The spatial Jacobian. */
  transform->GetSpatialJacobian( inputPoint, spatialJacobian );

  /** The spatial Hessian. */
  transform->GetSpatialHessian( inputPoint, spatialHessian );

  /** The Jacobian of the spatial Jacobian. */
  transform->GetJacobianOfSpatialJacobian( inputPoint,
    jacobianOfSpatialJacobian, nzji );

  transform->GetJacobianOfSpatialJacobian( inputPoint,
    spatialJacobian, jacobianOfSpatialJacobian, nzji );

  /** The Jacobian of the spatial Hessian. */
  transform->GetJacobianOfSpatialHessian( inputPoint,
    jacobianOfSpatialHessian, nzji );

  transform->GetJacobianOfSpatialHessian( inputPoint,
    spatialHessian, jacobianOfSpatialHessian, nzji );

//   /***/
//   transform->GetSpatialHessian( inputPoint, spatialHessian );
//   for ( unsigned int i = 0; i < Dimension; ++i )
//   {
//     std::cerr << spatialHessian[ i ] << std::endl;
//   }
//
//   /***/
//   transform->GetJacobianOfSpatialHessian( inputPoint,
//     spatialHessian, jacobianOfSpatialHessian, nzji );
//   for ( unsigned int mu = 0; mu < 2; ++mu )
//   {
//     for ( unsigned int i = 0; i < Dimension; ++i )
//     {
//       std::cerr << jacobianOfSpatialHessian[ mu ][ i ] << std::endl;
//     }
//   }

  /**
   *
   * Call functions for timing.
   *
   */

  clock_t startClock, endClock, clockITK;

  /** Time the implementation of the Jacobian. */
  startClock = clock();
  for( unsigned int i = 0; i < N * 10; ++i )
  {
    transform->GetJacobian( inputPoint, jacobian, nzji );
  }
  endClock = clock();
  clockITK = endClock - startClock;
  std::cerr << "The elapsed time for the Jacobian is: "
            << clockITK / 1000.0 << " s." << std::endl;

  /** Time the implementation of the spatial Jacobian. */
  startClock = clock();
  for( unsigned int i = 0; i < N; ++i )
  {
    transform->GetSpatialJacobian( inputPoint, spatialJacobian );
  }
  endClock = clock();
  clockITK = endClock - startClock;
  std::cerr << "The elapsed time for the spatial Jacobian is: "
            << clockITK / 1000.0 << " s." << std::endl;

  /** Time the implementation of the spatial Hessian. */
  startClock = clock();
  for( unsigned int i = 0; i < N; ++i )
  {
    transform->GetSpatialHessian( inputPoint, spatialHessian );
  }
  endClock = clock();
  clockITK = endClock - startClock;
  std::cerr << "The elapsed time for the spatial Hessian is: "
            << clockITK / 1000.0 << " s." << std::endl;

  /** Time the implementation of the Jacobian of the spatial Jacobian. */
  startClock = clock();
  for( unsigned int i = 0; i < N; ++i )
  {
    transform->GetJacobianOfSpatialJacobian( inputPoint,
      jacobianOfSpatialJacobian, nzji );
  }
  endClock = clock();
  clockITK = endClock - startClock;
  std::cerr << "The elapsed time for the Jacobian of the spatial Jacobian is: "
            << clockITK / 1000.0 << " s." << std::endl;

  /** Time the implementation of the Jacobian of the spatial Hessian. */
  startClock = clock();
  for( unsigned int i = 0; i < N; ++i )
  {
    transform->GetJacobianOfSpatialHessian( inputPoint,
      jacobianOfSpatialHessian, nzji );
  }
  endClock = clock();
  clockITK = endClock - startClock;
  std::cerr << "The elapsed time for the Jacobian of the spatial Hessian is: "
            << clockITK / 1000.0 << " s." << std::endl;

  /** Time the implementation of the spatial Jacobian and its Jacobian. */
  startClock = clock();
  for( unsigned int i = 0; i < N; ++i )
  {
    transform->GetSpatialJacobian( inputPoint, spatialJacobian );
    transform->GetJacobianOfSpatialJacobian( inputPoint,
      jacobianOfSpatialJacobian, nzji );
  }
  endClock = clock();
  clockITK = endClock - startClock;
  std::cerr << "The elapsed time for the (Jacobian of the) spatial Jacobian (2 func) is: "
            << clockITK / 1000.0 << " s." << std::endl;

  /** Time the implementation of the spatial Jacobian and its Jacobian. */
  startClock = clock();
  for( unsigned int i = 0; i < N; ++i )
  {
    transform->GetJacobianOfSpatialJacobian( inputPoint,
      spatialJacobian, jacobianOfSpatialJacobian, nzji );
  }
  endClock = clock();
  clockITK = endClock - startClock;
  std::cerr << "The elapsed time for the (Jacobian of the) spatial Jacobian (1 func) is: "
            << clockITK / 1000.0 << " s." << std::endl;

  /** Time the implementation of the spatial Hessian. */
  startClock = clock();
  for( unsigned int i = 0; i < N; ++i )
  {
    transform->GetSpatialHessian( inputPoint, spatialHessian );
    transform->GetJacobianOfSpatialHessian( inputPoint,
      jacobianOfSpatialHessian, nzji );
  }
  endClock = clock();
  clockITK = endClock - startClock;
  std::cerr << "The elapsed time for the (Jacobian of the) spatial Hessian (2 func) is: "
            << clockITK / 1000.0 << " s." << std::endl;

  /** Additional checks. */
  if( !transform->GetHasNonZeroSpatialHessian() )
  {
    std::cerr << "ERROR: GetHasNonZeroSpatialHessian() should return true." << std::endl;
    return 1;
  }
  if( !transform->GetHasNonZeroJacobianOfSpatialHessian() )
  {
    std::cerr << "ERROR: GetHasNonZeroJacobianOfSpatialHessian() should return true." << std::endl;
    return 1;
  }

  /** These should return the same values as the original ITK functions. */
  OutputPointType opp1           = transform->TransformPoint( inputPoint );
  OutputPointType opp2           = transformITK->TransformPoint( inputPoint );
  double          differenceNorm = 0.0;
  for( unsigned int i = 0; i < Dimension; ++i )
  {
    differenceNorm += ( opp1[ i ] - opp2[ i ] ) * ( opp1[ i ] - opp2[ i ] );
  }
  if( vcl_sqrt( differenceNorm ) > 1e-10 )
  {
    std::cerr << "ERROR: Advanced B-spline TransformPoint() returning incorrect result." << std::endl;
    return 1;
  }

  JacobianType jacobianITK; jacobianITK.Fill( 0.0 );
  transformITK->ComputeJacobianWithRespectToParameters( inputPoint, jacobianITK );
  JacobianType jacobianElastix; jacobianElastix.SetSize( Dimension, nzji.size() ); jacobianElastix.Fill( 0.0 );
  transform->GetJacobian( inputPoint, jacobianElastix, nzji );
  // ITK4 B-spline is non-local and returns a full matrix. Cannot compute diff like this anymore:
  //JacobianType jacobianDifferenceMatrix = jacobianElastix - jacobianITK;
  // I believe there are future plans to re-enable local support B-splines
  double jacDiff = 0.0;
  for( unsigned int i = 0; i < nzji.size(); i++ )
  {
    for( unsigned int j = 0; j < Dimension; j++ )
    {
      jacDiff += vnl_math_sqr( jacobianElastix[ j ][ i ] - jacobianITK[ j ][ nzji[ i ] ] );
    }
  }
  //if ( jacobianDifferenceMatrix.frobenius_norm() > 1e-10 )
  if( vcl_sqrt( jacDiff ) > 1e-10 )
  {
    std::cerr << "ERROR: Advanced B-spline GetJacobian() returning incorrect result." << std::endl;
    return 1;
  }

  //// Check
  //JacobianType jacobian1, jacobian2;
  //jacobian1.SetSize( Dimension, nzji.size() ); jacobian2.SetSize( Dimension, nzji.size() );
  //jacobian1.Fill( 0.0 ); jacobian2.Fill( 0.0 );
  //transform->GetJacobian( inputPoint, jacobian1, nzji );
  //transform->GetJacobian_opt( inputPoint, jacobian2, nzji );
  //JacobianType jacobianDifferenceMatrix = jacobian1 - jacobian2;
  //if ( jacobianDifferenceMatrix.frobenius_norm () > 1e-10 )
  //{
  //  std::cerr << "ERROR: Advanced B-spline GetJacobian_opt() returning incorrect result." << std::endl;
  //  return 1;
  //}

  //// Check
  //SpatialHessianType spatialHessian1, spatialHessian2;
  //JacobianOfSpatialHessianType jacobianOfSpatialHessian1, jacobianOfSpatialHessian2;
  //NonZeroJacobianIndicesType nzji1, nzji2;
  //transform->GetJacobianOfSpatialHessian( inputPoint,
  //  spatialHessian1, jacobianOfSpatialHessian1, nzji1 );
  //transform->GetJacobianOfSpatialHessian_opt( inputPoint,
  //  spatialHessian2, jacobianOfSpatialHessian2, nzji2 );
  //double shDiff = 0.0;
  //for( unsigned int i = 0; i < Dimension; i++ ) {
  //  for( unsigned int j = 0; j < Dimension; j++ ) {
  //    for( unsigned int k = 0; k < Dimension; k++ ) {
  //      shDiff += vnl_math_sqr( spatialHessian1[i][j][k] - spatialHessian2[i][j][k] );
  //    } } }
  //double jshDiff = 0.0;
  //for( unsigned int mu = 0; mu < nzji.size(); mu++ ) {
  //  for( unsigned int i = 0; i < Dimension; i++ ) {
  //    for( unsigned int j = 0; j < Dimension; j++ ) {
  //      for( unsigned int k = 0; k < Dimension; k++ ) {
  //        jshDiff += vnl_math_sqr( jacobianOfSpatialHessian1[mu][i][j][k] - jacobianOfSpatialHessian2[mu][i][j][k] );
  //      } } } }
  //if ( vcl_sqrt( shDiff ) > 1e-8 )
  //{
  //  std::cerr << "ERROR: Advanced B-spline GetJacobianOfSpatialHessian_opt() "
  //    << "returning incorrect spatial Hessian result: MSD = "
  //    << vcl_sqrt( shDiff ) << std::endl;
  //  return 1;
  //}
  //if ( vcl_sqrt( jshDiff ) > 1e-8 )
  //{
  //  std::cerr << "ERROR: Advanced B-spline GetJacobianOfSpatialHessian_opt() "
  //    << "returning incorrect Jacobian of spatial Hessian result: MSD = "
  //    << vcl_sqrt( jshDiff ) << std::endl;
  //  return 1;
  //}

  if( ( transform->GetParameters() - transformITK->GetParameters() ).two_norm() > 1e-10 )
  {
    std::cerr << "ERROR: Advanced B-spline GetParameters() returning incorrect result." << std::endl;
    return 1;
  }

  if( ( transform->GetFixedParameters() - transformITK->GetFixedParameters() ).two_norm() > 1e-10 )
  {
    std::cerr << "ERROR: Advanced B-spline GetFixedParameters() returning incorrect result." << std::endl;
    return 1;
  }

  /** Exercise PrintSelf(). */
  transform->Print( std::cerr );

  /** Return a value. */
  return 0;

} // end main
