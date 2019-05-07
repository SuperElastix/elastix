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
#ifndef __elxMultiResolutionRegistration_HXX__
#define __elxMultiResolutionRegistration_HXX__

#include "elxMultiResolutionRegistration.h"
#include "vnl/vnl_math.h"
#include "itkTimeProbe.h"

namespace elastix
{

/**
 * ******************* BeforeRegistration ***********************
 */

template< class TElastix >
void
MultiResolutionRegistration< TElastix >
::BeforeRegistration( void )
{
  /** Check for a common mistake that people make in their parameter
   * file: using MultiResolutionRegistration in combination with
   * more than 1 metric.
   */
  const unsigned int nrOfMetrics = this->GetElastix()->GetNumberOfMetrics();
  if( nrOfMetrics > 1 )
  {
    itkExceptionMacro( "\nERROR: the parameter file specifies \n"
        << "  (Registration \"MultiResolutionRegistration\")\n"
        << "  in combination with " << nrOfMetrics << " metrics.\n"
        << "  This registration only allows for 1 metric.\n"
        << "  You probably mean to use:\n"
        << "  (Registration \"MultiMetricMultiResolutionRegistration\")" );
  }

  /** Get the components from this->m_Elastix and set them. */
  this->SetComponents();

  /** Set the number of resolutions. */
  unsigned int numberOfResolutions = 3;
  this->m_Configuration->ReadParameter( numberOfResolutions, "NumberOfResolutions", 0 );
  this->SetNumberOfLevels( numberOfResolutions );

  /** Set the FixedImageRegion. */

  /** Make sure the fixed image is up to date. */
  try
  {
    this->GetElastix()->GetFixedImage()->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    /** Add information to the exception. */
    excp.SetLocation( "MultiResolutionRegistration - BeforeRegistration()" );
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while updating region info of the fixed image.\n";
    excp.SetDescription( err_str );
    /** Pass the exception to an higher level. */
    throw excp;
  }

  /** Set the fixedImageRegion. */
  this->SetFixedImageRegion( this->GetElastix()->GetFixedImage()->GetBufferedRegion() );

} // end BeforeRegistration()


/**
 * ******************* BeforeEachResolution ***********************
 */

template< class TElastix >
void
MultiResolutionRegistration< TElastix >
::BeforeEachResolution( void )
{
  /** Get the current resolution level. */
  unsigned int level = this->GetCurrentLevel();

  /** Do erosion, or just reset the original masks in the metric, or
   * do nothing when no masks are used.
   */
  this->UpdateMasks( level );

} // end BeforeEachResolution()


/**
 * *********************** SetComponents ************************
 */

template< class TElastix >
void
MultiResolutionRegistration< TElastix >
::SetComponents( void )
{
  /** Get the component from this-GetElastix() (as elx::...BaseType *),
   * cast it to the appropriate type and set it in 'this'. */

  this->SetFixedImage( this->GetElastix()->GetFixedImage() );
  this->SetMovingImage( this->GetElastix()->GetMovingImage() );

  this->SetFixedImagePyramid( this->GetElastix()->
    GetElxFixedImagePyramidBase()->GetAsITKBaseType() );

  this->SetMovingImagePyramid( this->GetElastix()->
    GetElxMovingImagePyramidBase()->GetAsITKBaseType() );

  this->SetInterpolator( this->GetElastix()->
    GetElxInterpolatorBase()->GetAsITKBaseType() );

  MetricType * testPtr = dynamic_cast< MetricType * >(
    this->GetElastix()->GetElxMetricBase()->GetAsITKBaseType() );
  if( testPtr )
  {
    this->SetMetric( testPtr );
  }
  else
  {
    itkExceptionMacro( << "ERROR: MultiResolutionRegistration expects the "
                       << "metric to be of type AdvancedImageToImageMetric!" );
  }

  this->SetOptimizer( dynamic_cast< OptimizerType * >(
      this->GetElastix()->GetElxOptimizerBase()->GetAsITKBaseType() ) );

  this->SetTransform( this->GetElastix()->
    GetElxTransformBase()->GetAsITKBaseType() );

  /** Samplers are not always needed: */
  if( this->GetElastix()->GetElxMetricBase()->GetAdvancedMetricUseImageSampler() )
  {
    if( this->GetElastix()->GetElxImageSamplerBase() )
    {
      this->GetElastix()->GetElxMetricBase()->SetAdvancedMetricImageSampler(
        this->GetElastix()->GetElxImageSamplerBase()->GetAsITKBaseType() );
    }
    else
    {
      xl::xout[ "error" ] << "No ImageSampler has been specified." << std::endl;
      itkExceptionMacro( << "The metric requires an ImageSampler, but it is not available!" );
    }
  }

} // end SetComponents()


/**
 * ************************* UpdateMasks ************************
 **/

template< class TElastix >
void
MultiResolutionRegistration< TElastix >
::UpdateMasks( unsigned int level )
{
  /** some shortcuts */
  const unsigned int nrOfFixedMasks    = this->GetElastix()->GetNumberOfFixedMasks();
  const unsigned int nrOfMovingMasks   = this->GetElastix()->GetNumberOfMovingMasks();
  const unsigned int oneOrNoFixedMasks = vnl_math_min( static_cast< unsigned int >( 1 ),
    nrOfFixedMasks );
  const unsigned int oneOrNoMovingMasks = vnl_math_min( static_cast< unsigned int >( 1 ),
    nrOfMovingMasks );

  /** Array of bools, that remembers for each mask if erosion is wanted.
   * dummy, we will not use it.
   */
  UseMaskErosionArrayType useMaskErosionArray;

  /** Bool that remembers if mask erosion is wanted in any of the masks
   * remains false when no masks are used.
   */
  bool useFixedMaskErosion;
  bool useMovingMaskErosion;

  /** Read whether mask erosion is wanted, if any masks were supplied
   * Only one mask is taken into account.
   */
  useFixedMaskErosion = this->ReadMaskParameters( useMaskErosionArray,
    oneOrNoFixedMasks, "Fixed", level );
  useMovingMaskErosion = this->ReadMaskParameters( useMaskErosionArray,
    oneOrNoMovingMasks, "Moving", level );

  /** Create and start timer, to time the whole fixed mask configuration procedure. */
  itk::TimeProbe timer;
  timer.Start();

  FixedMaskSpatialObjectPointer fixedMask = this->GenerateFixedMaskSpatialObject(
    this->GetElastix()->GetFixedMask(), useFixedMaskErosion,
    this->GetFixedImagePyramid(), level );

  if (fixedMask != nullptr)
  {
    fixedMask->Update();
  }
  this->GetModifiableMetric()->SetFixedImageMask( fixedMask );

  /** Stop timer and print the elapsed time. */
  timer.Stop();
  elxout << "Setting the fixed masks took: "
         << static_cast< long >( timer.GetMean() * 1000 )
         << " ms." << std::endl;

  /** Start timer, to time the whole moving mask configuration procedure. */
  timer.Reset();
  timer.Start();

  MovingMaskSpatialObjectPointer movingMask = this->GenerateMovingMaskSpatialObject(
    this->GetElastix()->GetMovingMask(), useMovingMaskErosion,
    this->GetMovingImagePyramid(), level );

  if (movingMask != nullptr)
  {
    movingMask->Update();
  }
  this->GetModifiableMetric()->SetMovingImageMask(movingMask);

  /** Stop timer and print the elapsed time. */
  timer.Stop();
  elxout << "Setting the moving masks took: "
         << static_cast< long >( timer.GetMean() * 1000 )
         << " ms." << std::endl;

} // end UpdateMasks()


} // end namespace elastix

#endif // end #ifndef __elxMultiResolutionRegistration_HXX__
