/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxCorrespondingPointsEuclideanDistanceMetric_HXX__
#define __elxCorrespondingPointsEuclideanDistanceMetric_HXX__

#include "elxCorrespondingPointsEuclideanDistanceMetric.h"
#include "itkTransformixInputPointFileReader.h"
#include "itkTimeProbe.h"


namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template< class TElastix >
void
CorrespondingPointsEuclideanDistanceMetric< TElastix >
::Initialize( void ) throw ( itk::ExceptionObject )
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  elxout << "Initialization of CorrespondingPointsEuclideanDistance metric took: "
    << static_cast< long >( timer.GetMean() * 1000 ) << " ms." << std::endl;

} // end Initialize()


/**
 * ***************** BeforeAllBase ***********************
 */

template< class TElastix >
int
CorrespondingPointsEuclideanDistanceMetric< TElastix >
::BeforeAllBase( void )
{
  this->Superclass2::BeforeAllBase();

  /** Check if the current configuration uses this metric. */
  unsigned int count = 0;
  for( unsigned int i = 0; i < this->m_Configuration
    ->CountNumberOfParameterEntries( "Metric" ); ++i )
  {
    std::string metricName = "";
    this->m_Configuration->ReadParameter( metricName, "Metric", i );
    if( metricName == "CorrespondingPointsEuclideanDistanceMetric" ) { count++; }
  }
  if( count == 0 ) { return 0; }

  /** Check Command line options and print them to the log file. */
  elxout << "Command line options from CorrespondingPointsEuclideanDistanceMetric:" << std::endl;
  std::string check( "" );

  /** Check for appearance of "-fp". */
  check = this->m_Configuration->GetCommandLineArgument( "-fp" );
  if( check.empty() )
  {
    elxout << "-fp       unspecified" << std::endl;
  }
  else
  {
    elxout << "-fp       " << check << std::endl;
  }

  /** Check for appearance of "-mp". */
  check = this->m_Configuration->GetCommandLineArgument( "-mp" );
  if( check.empty() )
  {
    elxout << "-mp       unspecified" << std::endl;
  }
  else
  {
    elxout << "-mp       " << check << std::endl;
  }

  /** Return a value. */
  return 0;

} // end BeforeAllBase()


/**
 * ***************** BeforeRegistration ***********************
 */

template< class TElastix >
void
CorrespondingPointsEuclideanDistanceMetric< TElastix >
::BeforeRegistration( void )
{
  /** Read and set the fixed pointset. */
  std::string fixedName = this->GetConfiguration()->GetCommandLineArgument( "-fp" );
  typename PointSetType::Pointer fixedPointSet      = 0;
  const typename ImageType::ConstPointer fixedImage = this->GetElastix()->GetFixedImage();
  const unsigned int nrOfFixedPoints = this->ReadLandmarks(
    fixedName, fixedPointSet, fixedImage );
  this->SetFixedPointSet( fixedPointSet );

  /** Read and set the moving pointset. */
  std::string movingName = this->GetConfiguration()->GetCommandLineArgument( "-mp" );
  typename PointSetType::Pointer movingPointSet      = 0;
  const typename ImageType::ConstPointer movingImage = this->GetElastix()->GetMovingImage();
  const unsigned int nrOfMovingPoints = this->ReadLandmarks(
    movingName, movingPointSet, movingImage );
  this->SetMovingPointSet( movingPointSet );

  /** Check. */
  if( nrOfFixedPoints != nrOfMovingPoints )
  {
    itkExceptionMacro( << "ERROR: the number of points in the fixed pointset ("
                       << nrOfFixedPoints << ") does not match that of the moving pointset ("
                       << nrOfMovingPoints << "). The points do not correspond. " );
  }

} // end BeforeRegistration()


/**
 * ***************** ReadLandmarks ***********************
 */

template< class TElastix >
unsigned int
CorrespondingPointsEuclideanDistanceMetric< TElastix >
::ReadLandmarks(
  const std::string & landmarkFileName,
  typename PointSetType::Pointer & pointSet,
  const typename ImageType::ConstPointer image )
{
  /** Typedefs. */
  typedef typename ImageType::IndexType      IndexType;
  typedef typename ImageType::IndexValueType IndexValueType;
  typedef typename ImageType::PointType      PointType;
  typedef itk::TransformixInputPointFileReader<
    PointSetType >                            PointSetReaderType;

  elxout << "Loading landmarks for " << this->GetComponentLabel()
         << ":" << this->elxGetClassName() << "." << std::endl;

  /** Read the landmarks. */
  typename PointSetReaderType::Pointer reader = PointSetReaderType::New();
  reader->SetFileName( landmarkFileName.c_str() );
  elxout << "  Reading landmark file: " << landmarkFileName << std::endl;
  try
  {
    reader->Update();
  }
  catch( itk::ExceptionObject & err )
  {
    xl::xout[ "error" ] << "  Error while opening " << landmarkFileName << std::endl;
    xl::xout[ "error" ] << err << std::endl;
    itkExceptionMacro( << "ERROR: unable to configure " << this->GetComponentLabel() );
  }

  /** Some user-feedback. */
  const unsigned int nrofpoints = reader->GetNumberOfPoints();
  if( reader->GetPointsAreIndices() )
  {
    elxout << "  Landmarks are specified as image indices." << std::endl;
  }
  else
  {
    elxout << "  Landmarks are specified in world coordinates." << std::endl;
  }
  elxout << "  Number of specified points: " << nrofpoints << std::endl;

  /** Get the pointset. */
  pointSet = reader->GetOutput();

  /** Convert from index to point if necessary */
  pointSet->DisconnectPipeline();
  if( reader->GetPointsAreIndices() )
  {
    /** Convert to world coordinates */
    for( unsigned int j = 0; j < nrofpoints; ++j )
    {
      /** The landmarks from the pointSet are indices. We first cast to the
       * proper type, and then convert it to world coordinates.
       */
      PointType point; IndexType index;
      pointSet->GetPoint( j, &point );
      for( unsigned int d = 0; d < FixedImageDimension; ++d )
      {
        index[ d ] = static_cast< IndexValueType >( itk::Math::Round< double >( point[ d ] ) );
      }

      /** Compute the input point in physical coordinates. */
      image->TransformIndexToPhysicalPoint( index, point );
      pointSet->SetPoint( j, point );

    } // end for all points
  }   // end for points are indices

  return nrofpoints;

} // end ReadLandmarks()


} // end namespace elastix

#endif // end #ifndef __elxCorrespondingPointsEuclideanDistanceMetric_HXX__
