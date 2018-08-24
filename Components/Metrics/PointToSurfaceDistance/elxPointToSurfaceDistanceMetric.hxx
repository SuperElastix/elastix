/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxPointToSurfaceDistanceMetric_HXX__
#define __elxPointToSurfaceDistanceMetric_HXX__

#include "elxPointToSurfaceDistanceMetric.h"
#include "itkTransformixInputPointFileReader.h"

namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template< class TElastix >
void
PointToSurfaceDistanceMetric< TElastix >
::Initialize( void ) throw ( itk::ExceptionObject )
{
} // end Initialize()

/**
 * ***************** BeforeAllBase ***********************
 */

template< class TElastix >
int
PointToSurfaceDistanceMetric< TElastix >
::BeforeAllBase( void )
{
  this->Superclass2::BeforeAllBase();
  /** Check if the current configuration uses this metric. */
  unsigned int count = 0;
  for( unsigned int i = 0; i < this->m_Configuration->CountNumberOfParameterEntries( "Metric" ); ++i )
  {
    std::string metricName = "";
    this->m_Configuration->ReadParameter( metricName, "Metric", i );
    if( metricName == "PointToSurfaceDistance" ) { count++; }
  }
  if( count == 0 ) { return 0; }

  /** Check Command line options and print them to the log file. */
  elxout << "Command line options from PointToSurfaceDistanceMetric:" << std::endl;
  std::string _fp( "" ),_dt( "" ),_seg( "" ),_dtout( "" );

  /** Check for appearance of parameter "PointToSurfaceDistanceAverage". */
  std::string PointToSurfaceDistanceAverageStr = "true";
  this->m_AvPointWeigh=true;
  if (this->m_Configuration->CountNumberOfParameterEntries( "PointToSurfaceDistanceAverage" )==1)
  {
    this->m_Configuration->ReadParameter( PointToSurfaceDistanceAverageStr, "PointToSurfaceDistanceAverage", 0 );
    if (PointToSurfaceDistanceAverageStr=="false") this->m_AvPointWeigh=false;
  }

  elxout << "\nAverage of points in annotation set : "
         << PointToSurfaceDistanceAverageStr<<"\n"
         <<  std::endl;

  /** Check for appearance of "-fp". */
  _fp = this->m_Configuration->GetCommandLineArgument( "-fp" );
  if( _fp.empty() )
  {
    elxout << "-fp       unspecified" << std::endl;
  }
  else
  {
    elxout << "-fp       " << _fp << std::endl;
  }
  /** Check for appearance of "-dt". */
  _dt = this->m_Configuration->GetCommandLineArgument( "-dt" );
  if( _dt.empty() )
  {
    elxout << "-dt       unspecified" << std::endl;
  }
  else
  {
    elxout << "-dt       " << _dt << std::endl;
    this->Superclass1::SetDTImageIn(_dt);
  }

  /** Check for appearance of "-seg". */
  _seg = this->m_Configuration->GetCommandLineArgument( "-seg" );
  if( _seg.empty() )
  {
    elxout << "-seg      unspecified" << std::endl;
  }
  else
  {
    elxout << "-seg      " << _seg << std::endl;
    this->Superclass1::SetSegImageIn(_seg);
  }

  _dtout = this->m_Configuration->GetCommandLineArgument( "-dtout" );
  if( _dtout.empty() )
  {
    elxout << "-dtout      unspecified" << std::endl;
  }
  else
  {
    elxout << "-dtout    " << _dtout << std::endl;
    this->Superclass1::SetDTImageOut(_dtout);
  }

  this->Superclass1::Initialize();
  /** Return a value. */
  return 0;

} // end BeforeAllBase()

/**
 * ***************** BeforeRegistration ***********************
 */

template< class TElastix >
void
PointToSurfaceDistanceMetric< TElastix >
::BeforeRegistration( void )
{
  /** Read and set the fixed pointset. */
  std::string fixedName = this->GetConfiguration()->GetCommandLineArgument( "-fp" );
  typename PointSetType::Pointer fixedPointSet      = 0;
  const typename FixedImageType::ConstPointer Image = this->GetElastix()->GetFixedImage();
  const unsigned int nrOfFixedPoints = this->ReadLandmarks( fixedName, fixedPointSet, Image );

  this->SetFixedPointSet( fixedPointSet );////this is pointset interface for the layer a code
	
  /*Dummy Setting for the moving point set*/
  //this->SetMovingPointSet( movingPointSet );

} // end BeforeRegistration()

/**
 * ***************** ReadLandmarks ***********************
 */

template< class TElastix >
unsigned int
PointToSurfaceDistanceMetric< TElastix >
::ReadLandmarks( const std::string & landmarkFileName,typename PointSetType::Pointer & pointSet, const typename FixedImageType::ConstPointer image )
{
  /** Typedefs. */
  typedef typename ImageType::IndexType      IndexType;
  typedef typename ImageType::IndexValueType IndexValueType;
  typedef typename ImageType::PointType      PointType;
  typedef itk::TransformixInputPointFileReader<PointSetType > PointSetReaderType;

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

} // end ReadLandmarks()*/

} // end namespace elastix

#endif // end #ifndef __elxPointToSurfaceDistanceMetric_HXX__
