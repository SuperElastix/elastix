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
#ifndef __elxDistancePreservingRigidityPenaltyTerm_HXX__
#define __elxDistancePreservingRigidityPenaltyTerm_HXX__

#include "elxDistancePreservingRigidityPenaltyTerm.h"

#include "itkChangeInformationImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkTimeProbe.h"

namespace elastix
{

/**
 * ******************* BeforeRegistration ***********************
 */

template< class TElastix >
void
DistancePreservingRigidityPenalty< TElastix >
::BeforeRegistration( void )
{
  /** Read the fixed rigidity image. */
  string segmentedImageName = "";
  this->GetConfiguration()->ReadParameter( segmentedImageName,
    "SegmentedImageName", this->GetComponentLabel(), 0, -1, false );

  typedef typename Superclass1::SegmentedImageType                SegmentedImageType;
  typedef itk::ImageFileReader< SegmentedImageType >              SegmentedImageReaderType;
  typedef itk::ChangeInformationImageFilter< SegmentedImageType > ChangeInfoFilterType;
  typedef typename ChangeInfoFilterType::Pointer                  ChangeInfoFilterPointer;
  typedef typename SegmentedImageType::DirectionType              DirectionType;
  typedef typename SegmentedImageType::SizeType::SizeValueType    SizeValueType;

  /** Create the reader and set the filename. */
  typename SegmentedImageReaderType::Pointer segmentedImageReader = SegmentedImageReaderType::New();
  segmentedImageReader->SetFileName( segmentedImageName.c_str() );
  segmentedImageReader->Update();

  /** Possibly overrule the direction cosines. */
  ChangeInfoFilterPointer infoChanger = ChangeInfoFilterType::New();
  DirectionType           direction;
  direction.SetIdentity();
  infoChanger->SetOutputDirection( direction );
  infoChanger->SetChangeDirection( !this->GetElastix()->GetUseDirectionCosines() );
  infoChanger->SetInput( segmentedImageReader->GetOutput() );

  /** Do the reading. */
  try
  {
    infoChanger->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    /** Add information to the exception. */
    excp.SetLocation( "MattesMutualInformationWithRigidityPenalty - BeforeRegistration()" );
    string err_str = excp.GetDescription();
    err_str += "\nError occurred while reading the segmented image.\n";
    excp.SetDescription( err_str );
    /** Pass the exception to an higher level. */
    throw excp;
  }

  this->SetSegmentedImage( infoChanger->GetOutput() );

  /** Get information from the segmented image. */
  typename SegmentedImageType::SizeType segmentedImageSize       = this->GetSegmentedImage()->GetBufferedRegion().GetSize();
  typename SegmentedImageType::PointType segmentedImageOrigin    = this->GetSegmentedImage()->GetOrigin();
  typename SegmentedImageType::SpacingType segmentedImageSpacing = this->GetSegmentedImage()->GetSpacing();

  /** Get the grid sampling spacing for calculation of the rigidity penalty term. */
  typename SegmentedImageType::SpacingType penaltyGridSpacingInVoxels;
  for( unsigned int dim = 0; dim < ImageDimension; ++dim )
  {
    this->m_Configuration->ReadParameter(
      penaltyGridSpacingInVoxels[ dim ], "PenaltyGridSpacingInVoxels",
      this->GetComponentLabel(), dim, 0 );
  }

  /** Compute resampled spacing and size. */
  typename SegmentedImageType::SpacingType resampledImageSpacing;
  typename SegmentedImageType::SizeType resampledImageSize;
  for( unsigned int dim = 0; dim < ImageDimension; ++dim )
  {
    resampledImageSpacing[ dim ] = segmentedImageSpacing[ dim ] * penaltyGridSpacingInVoxels[ dim ];
    resampledImageSize[ dim ] = static_cast< SizeValueType >(
      segmentedImageSize[ dim ] / penaltyGridSpacingInVoxels[ dim ] );
  }

  /** Create resampler, identity transform and linear interpolator. */
  typedef itk::ResampleImageFilter< SegmentedImageType, SegmentedImageType > ResampleFilterType;
  typename ResampleFilterType::Pointer resampler = ResampleFilterType::New();

  typedef itk::IdentityTransform< double, Superclass1::ImageDimension > IdentityTransformType;
  typename IdentityTransformType::Pointer identityTransform = IdentityTransformType::New();
  identityTransform->SetIdentity();

  typedef itk::LinearInterpolateImageFunction< SegmentedImageType, double > LinearInterpolatorType;
  typename LinearInterpolatorType::Pointer linearInterpolator = LinearInterpolatorType::New();

  /** Configure the resampler and run it. */
  resampler->SetInterpolator( linearInterpolator );
  resampler->SetTransform( identityTransform );
  resampler->SetOutputSpacing( resampledImageSpacing );
  resampler->SetOutputOrigin( segmentedImageOrigin );
  resampler->SetSize( resampledImageSize );
  resampler->SetInput( this->GetSegmentedImage() );
  resampler->Update();

  this->SetSampledSegmentedImage( resampler->GetOutput() );

} // end BeforeRegistration()


/**
 * ******************* Initialize ***********************
 */

template< class TElastix >
void
DistancePreservingRigidityPenalty< TElastix >
::Initialize( void ) throw ( itk::ExceptionObject )
{
  /** Initialize this class with the Superclass initializer. */
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();

  /** Stop and print the timer. */
  timer.Stop();
  elxout << "Initialization of DistancePreservingRigidityPenalty term took: "
         << static_cast< long >( timer.GetMean() * 1000 ) << " ms." << std::endl;

} // end Initialize()


} // end namespace elastix

#endif // end #ifndef __elxDistancePreservingRigidityPenaltyTerm_HXX__
