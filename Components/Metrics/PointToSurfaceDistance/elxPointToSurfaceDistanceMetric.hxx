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
::Initialize( void )
{
}

/**
 * ***************** BeforeAllBase ***********************
 */

template< class TElastix >
int
PointToSurfaceDistanceMetric< TElastix >
::BeforeAllBase()
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

  /** Check for appearance of parameter "PointToSurfaceDistanceAverage". */
  std::string PointToSurfaceDistanceAverageStr = "true";
  this->m_AvPointWeigh = true;
  if (this->m_Configuration->CountNumberOfParameterEntries( "PointToSurfaceDistanceAverage" ) == 1)
  {
    this->m_Configuration->ReadParameter( PointToSurfaceDistanceAverageStr, "PointToSurfaceDistanceAverage", 0 );
    if (PointToSurfaceDistanceAverageStr == "false") this->m_AvPointWeigh = false;
  }

  elxout << "\nAverage of points in annotation set : "
         << PointToSurfaceDistanceAverageStr <<"\n"
         <<  std::endl;

  /** Check for appearance of "-fp". */
  auto _fp = this->m_Configuration->GetCommandLineArgument( "-fp" );
  if( _fp.empty() )
  {
    elxout << "-fp       unspecified" << std::endl;
  }
  else
  {
    elxout << "-fp       " << _fp << std::endl;
  }
  /** Check for appearance of "-dt". */
  auto _dt = this->m_Configuration->GetCommandLineArgument( "-dt" );
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
  auto _seg = this->m_Configuration->GetCommandLineArgument( "-seg" );
  if( _seg.empty() )
  {
    elxout << "-seg      unspecified" << std::endl;
  }
  else
  {
    elxout << "-seg      " << _seg << std::endl;
    this->Superclass1::SetSegImageIn(_seg);
  }

  /** Check for appearance of "-dtout". */
  auto _dtout = this->m_Configuration->GetCommandLineArgument( "-dtout" );
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
  return 0;
}

/**
 * ***************** BeforeRegistration ***********************
 */

template< class TElastix >
void
PointToSurfaceDistanceMetric< TElastix >
::BeforeRegistration()
{
  /** Read and set the fixed pointset. */
  auto fixedPointsetFileName = this->GetConfiguration()->GetCommandLineArgument( "-fp" );
  typename PointSetType::Pointer fixedPointSet;
  const typename FixedImageType::ConstPointer Image = this->GetElastix()->GetFixedImage();
  ReadLandmarks( fixedPointsetFileName, fixedPointSet, Image );

  this->SetFixedPointSet( fixedPointSet );////this is pointset interface for the layer a code
} 

/**
 * ***************** ReadLandmarks ***********************
 */

template< class TElastix >
unsigned int
PointToSurfaceDistanceMetric< TElastix >
::ReadLandmarks( const std::string & landmarkFileName, typename PointSetType::Pointer & pointSet, const typename FixedImageType::ConstPointer image )
{
  using  PointSetReaderType = itk::TransformixInputPointFileReader<PointSetType >;

  elxout << "Loading landmarks for " << this->GetComponentLabel()
         << ":" << this->elxGetClassName() << "." << std::endl;

  /** Read the landmarks. */
  auto reader = PointSetReaderType::New();

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
  const auto nrofpoints = reader->GetNumberOfPoints();
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
    for( auto j = 0u; j < nrofpoints; ++j )
    {
      /** The landmarks from the pointSet are indices. We first cast to the
       * proper type, and then convert it to world coordinates.
       */
      typename ImageType::PointType point;
      typename ImageType::IndexType index;
      pointSet->GetPoint( j, &point );
      for( auto d = 0u; d < FixedImageDimension; ++d )
      {
        index[ d ] = static_cast<typename ImageType::IndexValueType>( itk::Math::Round< double >( point[ d ] ) );
      }

      /** Compute the input point in physical coordinates. */
      image->TransformIndexToPhysicalPoint( index, point );
      pointSet->SetPoint( j, point );
    } // end for all points
  }   // end for points are indices

  return nrofpoints;
}

} // end namespace elastix

#endif // end #ifndef __elxPointToSurfaceDistanceMetric_HXX__
