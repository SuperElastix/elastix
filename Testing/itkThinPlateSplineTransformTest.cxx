/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#include "SplineKernelTransform/itkThinPlateSplineKernelTransform2.h"
#include "itkTransformixInputPointFileReader.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"

#include <ctime>
#include <fstream>
#include <iomanip>

//-------------------------------------------------------------------------------------

int
main( int argc, char * argv[] )
{
  /** Some basic type definitions. */
  const unsigned int Dimension = 3;
  typedef double ScalarType;   // ScalarType double used in elastix

  /** Only perform the test with usedNumberOfLandmarks. */
  const unsigned long usedNumberOfLandmarks = 100;
  std::cerr << "Number of used landmarks: "
            << usedNumberOfLandmarks << std::endl;

  /** Check. */
  if( argc != 2 )
  {
    std::cerr << "ERROR: You should specify a text file with the thin plate spline "
              << "source (fixed image) landmarks." << std::endl;
    return 1;
  }

  /** Other typedefs. */
  typedef itk::ThinPlateSplineKernelTransform2<
    ScalarType, Dimension >                             TransformType;
  typedef TransformType::JacobianType               JacobianType;
  typedef TransformType::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;
  typedef TransformType::PointSetType               PointSetType;
  typedef TransformType::InputPointType             InputPointType;
  typedef TransformType::OutputPointType            OutputPointType;

  typedef itk::TransformixInputPointFileReader<
    PointSetType >                                      IPPReaderType;
  typedef PointSetType::PointsContainer PointsContainerType;
  typedef PointsContainerType::Pointer  PointsContainerPointer;
  typedef PointSetType::PointType       PointType;

  typedef itk::Statistics::MersenneTwisterRandomVariateGenerator MersenneTwisterType;

  /** Create the kernel transform. */
  TransformType::Pointer kernelTransform = TransformType::New();

  /** Read landmarks. */
  IPPReaderType::Pointer landmarkReader = IPPReaderType::New();
  landmarkReader->SetFileName( argv[ 1 ] );
  try
  {
    landmarkReader->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << "  Error while opening landmark file." << std::endl;
    std::cerr << excp << std::endl;
    return 1;
  }
  PointSetType::Pointer sourceLandmarks = landmarkReader->GetOutput();

  /** Check: Expect points, not indices. */
  if( landmarkReader->GetPointsAreIndices() )
  {
    std::cerr << "ERROR: landmarks should be specified as points (not indices)"
              << std::endl;
    return 1;
  }

  /** Get subset. */
  PointsContainerPointer usedLandmarkPoints  = PointsContainerType::New();
  PointSetType::Pointer  usedSourceLandmarks = PointSetType::New();
  for( unsigned long j = 0; j < usedNumberOfLandmarks; j++ )
  {
    PointType tmp = ( *sourceLandmarks->GetPoints() )[ j ];
    usedLandmarkPoints->push_back( tmp );
  }
  usedSourceLandmarks->SetPoints( usedLandmarkPoints );

  /** Test 1: Time setting the source landmarks. */
  clock_t startClock = clock();
  kernelTransform->SetSourceLandmarks( usedSourceLandmarks );
  std::cerr << "Setting source landmarks took "
            << clock() - startClock << " ms." << std::endl;

  /** Further setup the kernel transform. */
  kernelTransform->SetStiffness( 0.0 );              // interpolating
  kernelTransform->SetMatrixInversionMethod( "QR" ); // faster
  kernelTransform->SetIdentity();                    // target landmarks = source landmarks

  /** Create new target landmarks by adding a random vector to it. */
  PointSetType::Pointer        targetLandmarks         = kernelTransform->GetTargetLandmarks();
  PointsContainerPointer       newTargetLandmarkPoints = PointsContainerType::New();
  MersenneTwisterType::Pointer mersenneTwister         = MersenneTwisterType::New();
  mersenneTwister->Initialize( 140377 );
  for( unsigned long j = 0; j < targetLandmarks->GetNumberOfPoints(); j++ )
  {
    PointType tmp = ( *targetLandmarks->GetPoints() )[ j ];
    PointType randomPoint;
    for( unsigned int dim = 0; dim < Dimension; dim++ )
    {
      randomPoint[ dim ] = tmp[ dim ] + mersenneTwister->GetNormalVariate( 1.0, 5.0 );
    }
    newTargetLandmarkPoints->push_back( randomPoint );
  }
  PointSetType::Pointer newTargetLandmarks = PointSetType::New();
  newTargetLandmarks->SetPoints( newTargetLandmarkPoints );

  /** Test 2: Time setting the target landmarks. */
  startClock = clock();
  kernelTransform->SetTargetLandmarks( newTargetLandmarks );
  std::cerr << "Setting source landmarks took "
            << clock() - startClock << " ms." << std::endl;

  InputPointType ipp; ipp[ 0 ] = 10.0; ipp[ 0 ] = 20.0; ipp[ 0 ] = 30.0;

  /** Test TransformPoint(). */
  startClock = clock();
  kernelTransform->TransformPoint( ipp );
  std::cerr << "TransformPoint() computation took: "
            << clock() - startClock << " ms." << std::endl;

  /** Test GetJacobian(). */
  startClock = clock();
  JacobianType jac; NonZeroJacobianIndicesType nzji;
  kernelTransform->GetJacobian( ipp, jac, nzji );
  std::cerr << "GetJacobian() computation took: "
            << clock() - startClock << " ms." << std::endl;

  /** Additional checks. */
  if( !kernelTransform->GetHasNonZeroSpatialHessian() )
  {
    std::cerr << "ERROR: GetHasNonZeroSpatialHessian() should return true." << std::endl;
    return 1;
  }
  if( !kernelTransform->GetHasNonZeroJacobianOfSpatialHessian() )
  {
    std::cerr << "ERROR: GetHasNonZeroJacobianOfSpatialHessian() should return true." << std::endl;
    return 1;
  }
  if( kernelTransform->GetMatrixInversionMethod() != "QR" )
  {
    std::cerr << "ERROR: GetMatrixInversionMethod() should return \"QR\"." << std::endl;
    return 1;
  }

  /** Exercise PrintSelf() method. */
  kernelTransform->Print( std::cerr );

  /** Return a value. */
  return 0;

} // end main
