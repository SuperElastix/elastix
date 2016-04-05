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
#ifndef elxParameterObject_cxx
#define elxParameterObject_cxx

#include "elxParameterObject.h"

#include "itkFileTools.h"
#include <fstream>
#include <iostream>


namespace elastix
{

void
ParameterObject
::SetParameterMap( const ParameterMapType parameterMap )
{
  ParameterMapVectorType parameterMapVector;
  parameterMapVector.push_back( parameterMap );
  this->SetParameterMap( parameterMapVector );
}

void
ParameterObject
::SetParameterMap( const ParameterMapVectorType parameterMapVector )
{
  this->Modified();
  this->m_ParameterMapVector = parameterMapVector;
}

void
ParameterObject
::AddParameterMap( const ParameterMapType parameterMap )
{
  this->Modified();
  this->m_ParameterMapVector.push_back( parameterMap );
}

ParameterObject::ParameterMapType&
ParameterObject
::GetParameterMap( unsigned int index )
{
  this->Modified();
  return this->m_ParameterMapVector[ index ];
}

ParameterObject::ParameterMapVectorType&
ParameterObject
::GetParameterMap( void )
{
  this->Modified();
  return this->m_ParameterMapVector;
}

const ParameterObject::ParameterMapVectorType&
ParameterObject
::GetParameterMap( void ) const
{
  return this->m_ParameterMapVector;
}

void
ParameterObject
::ReadParameterFile( const ParameterFileNameType parameterFileName )
{
  ParameterFileParserPointer parameterFileParser = ParameterFileParserType::New();
  parameterFileParser->SetParameterFileName( parameterFileName );
  parameterFileParser->ReadParameterFile();
  this->SetParameterMap( ParameterMapVectorType( 1, parameterFileParser->GetParameterMap() ) );
}

void
ParameterObject
::ReadParameterFile( const ParameterFileNameVectorType parameterFileNameVector )
{
  if( parameterFileNameVector.size() == 0 )
  {
    itkExceptionMacro( "Parameter filename container is empty." );
  }

  this->m_ParameterMapVector.clear();

  for( unsigned int i = 0; i < parameterFileNameVector.size(); ++i )
  {
    if( !itksys::SystemTools::FileExists( parameterFileNameVector[ i ] ) )
    {
      itkExceptionMacro( "Parameter file \"" << parameterFileNameVector[ i ] << "\" does not exist." )
    }

    this->AddParameterFile( parameterFileNameVector[ i ] );
  }
}

void
ParameterObject
::AddParameterFile( const ParameterFileNameType parameterFileName )
{
  ParameterFileParserPointer parameterFileParser = ParameterFileParserType::New();
  parameterFileParser->SetParameterFileName( parameterFileName );
  parameterFileParser->ReadParameterFile();
  this->m_ParameterMapVector.push_back( parameterFileParser->GetParameterMap() );
}

void
ParameterObject
::WriteParameterFile( const ParameterMapType parameterMap, const ParameterFileNameType parameterFileName )
{
  std::ofstream parameterFile;
  parameterFile.exceptions( std::ofstream::failbit | std::ofstream::badbit );
  parameterFile << std::fixed;

  try
  {
    parameterFile.open( parameterFileName.c_str(), std::ofstream::out );
  }
  catch( std::ofstream::failure e )
  {
    itkExceptionMacro( "Error opening parameter file: " << e.what() );
  }

  try
  {
    ParameterMapConstIterator parameterMapIterator = parameterMap.begin();
    ParameterMapConstIterator parameterMapIteratorEnd = parameterMap.end();
    while( parameterMapIterator != parameterMapIteratorEnd )
    {
      parameterFile << "(" << parameterMapIterator->first;

      ParameterValueVectorType parameterMapValueVector = parameterMapIterator->second;
      for( unsigned int i = 0; i < parameterMapValueVector.size(); ++i )
      {
        std::stringstream stream( parameterMapValueVector[ i ] );
        float number;
        stream >> number;
        if( stream.fail() || stream.bad() )
        {
          parameterFile << " \"" << parameterMapValueVector[ i ] << "\"";
        }
        else
        {
          parameterFile << " " << number;
        }
      }

      parameterFile << ")" << std::endl;
      parameterMapIterator++;
    }
  }
  catch( std::stringstream::failure e )
  {
    itkExceptionMacro( "Error writing to paramter file: " << e.what() );
  }

  try
  {
    parameterFile.close();
  }
  catch( std::ofstream::failure e )
  {
    itkExceptionMacro( "Error closing parameter file:" << e.what() );
  }
}

void
ParameterObject
::WriteParameterFile( const ParameterFileNameType parameterFileName )
{
  if( this->m_ParameterMapVector.size() == 0 )
  {
    itkExceptionMacro( "Error writing parameter map to disk: The parameter object is empty." );
  }

  if( this->m_ParameterMapVector.size() > 1 )
  {
    itkExceptionMacro( "Error writing to disk: The number of parameter maps (" << this->m_ParameterMapVector.size() << ")"
                    << " does not match the number of provided filenames (1). Please provide a vector of filenames." );
  }

  this->WriteParameterFile( this->m_ParameterMapVector[ 0 ], parameterFileName );
}

void
ParameterObject
::WriteParameterFile( const ParameterFileNameVectorType parameterFileNameVector )
{
  if( this->m_ParameterMapVector.size() != parameterFileNameVector.size() )
  {
    itkExceptionMacro( "Error writing to disk: The number of parameter maps (" << this->m_ParameterMapVector.size() << ")"
                    << " does not match the number of provided filenames (" << parameterFileNameVector.size() << ")." );
  }

  for( unsigned int i = 0; i < this->m_ParameterMapVector.size(); ++i )
  {
    this->WriteParameterFile( this->m_ParameterMapVector[ i ], parameterFileNameVector[ i ] );
  }
}

void
ParameterObject
::SetParameterMap( const std::string transformName, const unsigned int numberOfResolutions, const double finalGridSpacingInPhysicalUnits )
{
  this->m_ParameterMapVector = ParameterMapVectorType( 1, this->GetParameterMap( transformName, numberOfResolutions, finalGridSpacingInPhysicalUnits ) );
}

void
ParameterObject
::AddParameterMap( const std::string transformName, const unsigned int numberOfResolutions, const double finalGridSpacingInPhysicalUnits )
{
  this->m_ParameterMapVector.push_back( this->GetParameterMap( transformName, numberOfResolutions, finalGridSpacingInPhysicalUnits ) );
}

ParameterObject::ParameterMapType
ParameterObject
::GetParameterMap( const std::string transformName, const unsigned int numberOfResolutions, const double finalGridSpacingInPhysicalUnits )
{
  // Parameters that depend on size and number of resolutions
  ParameterMapType parameterMap                        = ParameterMapType();

  // Common Components
  parameterMap[ "FixedImagePyramid" ]                  = ParameterValueVectorType( 1, "FixedSmoothingImagePyramid" );
  parameterMap[ "MovingImagePyramid" ]                 = ParameterValueVectorType( 1, "MovingSmoothingImagePyramid" );
  parameterMap[ "Interpolator"]                        = ParameterValueVectorType( 1, "LinearInterpolator" );
  parameterMap[ "Optimizer" ]                          = ParameterValueVectorType( 1, "AdaptiveStochasticGradientDescent" );
  parameterMap[ "Resampler" ]                          = ParameterValueVectorType( 1, "DefaultResampler" );
  parameterMap[ "ResampleInterpolator" ]               = ParameterValueVectorType( 1, "FinalBSplineInterpolator" );
  parameterMap[ "FinalBSplineInterpolationOrder" ]     = ParameterValueVectorType( 1, "3" );
  parameterMap[ "NumberOfResolutions" ]                = ParameterValueVectorType( 1, ToString( numberOfResolutions ) );

  // Image Sampler
  parameterMap[ "ImageSampler" ]                       = ParameterValueVectorType( 1, "RandomCoordinate" );
  parameterMap[ "NumberOfSpatialSamples"]              = ParameterValueVectorType( 1, "2048" );
  parameterMap[ "CheckNumberOfSamples" ]               = ParameterValueVectorType( 1, "true" );
  parameterMap[ "MaximumNumberOfSamplingAttempts" ]    = ParameterValueVectorType( 1, "8" );
  parameterMap[ "NewSamplesEveryIteration" ]           = ParameterValueVectorType( 1, "true" );

  // Optimizer
  parameterMap[ "NumberOfSamplesForExactGradient" ]    = ParameterValueVectorType( 1, "4096" );
  parameterMap[ "DefaultPixelValue" ]                  = ParameterValueVectorType( 1, "0.0" );
  parameterMap[ "AutomaticParameterEstimation" ]       = ParameterValueVectorType( 1, "true" );

  // Output
  parameterMap[ "WriteResultImage" ]                   = ParameterValueVectorType( 1, "true" );
  parameterMap[ "ResultImageFormat" ]                  = ParameterValueVectorType( 1, "nii" );

  // transformNames
  if( transformName == "translation" )
  {
    parameterMap[ "Registration" ]                     = ParameterValueVectorType( 1, "MultiResolutionRegistration" );
    parameterMap[ "Transform" ]                        = ParameterValueVectorType( 1, "TranslationTransform" );
    parameterMap[ "Metric" ]                           = ParameterValueVectorType( 1, "AdvancedMattesMutualInformation" );
    parameterMap[ "MaximumNumberOfIterations" ]        = ParameterValueVectorType( 1, "256" );
  }
  else if( transformName == "rigid" )
  {
    parameterMap[ "Registration" ]                     = ParameterValueVectorType( 1, "MultiResolutionRegistration" );
    parameterMap[ "Transform" ]                        = ParameterValueVectorType( 1, "EulerTransform" );
    parameterMap[ "Metric" ]                           = ParameterValueVectorType( 1, "AdvancedMattesMutualInformation" );
    parameterMap[ "MaximumNumberOfIterations" ]        = ParameterValueVectorType( 1, "256" );
  }
  else if( transformName == "affine" )
  {
    parameterMap[ "Registration" ]                     = ParameterValueVectorType( 1, "MultiResolutionRegistration" );
    parameterMap[ "Transform" ]                        = ParameterValueVectorType( 1, "AffineTransform" );
    parameterMap[ "Metric" ]                           = ParameterValueVectorType( 1, "AdvancedMattesMutualInformation" );
    parameterMap[ "MaximumNumberOfIterations" ]        = ParameterValueVectorType( 1, "256" );
  }
  else if( transformName == "bspline" || transformName == "nonrigid" ) // <-- nonrigid for backwards compatibility
  {
    parameterMap[ "Registration" ]                     = ParameterValueVectorType( 1, "MultiMetricMultiResolutionRegistration" );
    parameterMap[ "Transform" ]                        = ParameterValueVectorType( 1, "BSplineTransform" );
    parameterMap[ "Metric" ]                           = ParameterValueVectorType( 1, "AdvancedMattesMutualInformation" );
    parameterMap[ "Metric" ].push_back( "TransformBendingEnergyPenalty" );
    parameterMap[ "Metric0Weight" ]                    = ParameterValueVectorType( 1, "1.0" );
    parameterMap[ "Metric1Weight" ]                    = ParameterValueVectorType( 1, "10000.0" );
    parameterMap[ "MaximumNumberOfIterations" ]        = ParameterValueVectorType( 1, "256" );
  }
  else if( transformName == "spline")   
  {   
    parameterMap[ "Registration" ]                     = ParameterValueVectorType( 1, "MultiResolutionRegistration" );    
    parameterMap[ "Transform" ]                        = ParameterValueVectorType( 1, "SplineKernelTransform" );    
    parameterMap[ "Metric" ]                           = ParameterValueVectorType( 1, "AdvancedMattesMutualInformation" );    
    parameterMap[ "MaximumNumberOfIterations" ]        = ParameterValueVectorType( 1, "256" );
  }
  else if( transformName == "groupwise" )
  {
    parameterMap[ "Registration" ]                     = ParameterValueVectorType( 1, "MultiResolutionRegistration" );
    parameterMap[ "Transform" ]                        = ParameterValueVectorType( 1, "BSplineStackTransform" );
    parameterMap[ "Metric" ]                           = ParameterValueVectorType( 1, "VarianceOverLastDimensionMetric" );
    parameterMap[ "MaximumNumberOfIterations" ]        = ParameterValueVectorType( 1, "256" );
    parameterMap[ "Interpolator"]                      = ParameterValueVectorType( 1, "ReducedDimensionBSplineInterpolator" );
    parameterMap[ "ResampleInterpolator" ]             = ParameterValueVectorType( 1, "FinalReducedDimensionBSplineInterpolator" );
  }
  else
  {
    itkExceptionMacro( "No default parameter map \"" << transformName << "\"." );
  }

  // B-spline transform settings
  if( transformName == "bspline" || transformName == "nonrigid" || transformName == "groupwise" ) // <-- nonrigid for backwards compatibility
  {
    ParameterValueVectorType gridSpacingSchedule = ParameterValueVectorType();
    for( double resolution = 0; resolution < numberOfResolutions; ++resolution )
    {
      gridSpacingSchedule.insert( gridSpacingSchedule.begin(), ToString( pow( 1.41, resolution ) ) );
    }

    parameterMap[ "GridSpacingSchedule" ] = gridSpacingSchedule;
    parameterMap[ "FinalGridSpacingInPhysicalUnits" ] = ParameterValueVectorType( 1, ToString( finalGridSpacingInPhysicalUnits ) );
  }

  return parameterMap;
}

} // namespace elastix

#endif // elxParameterObject_cxx
