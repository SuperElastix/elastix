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
#ifndef elxDistancePreservingRigidityPenaltyTerm_hxx
#define elxDistancePreservingRigidityPenaltyTerm_hxx

#include "elxDistancePreservingRigidityPenaltyTerm.h"

#include "itkChangeInformationImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkTimeProbe.h"

namespace elastix
{

/**
 * ******************* BeforeRegistration ***********************
 */

template <class TElastix>
void
DistancePreservingRigidityPenalty<TElastix>::BeforeRegistration()
{
  /** Read the fixed rigidity image. */
  std::string segmentedImageName = "";
  this->GetConfiguration()->ReadParameter(
    segmentedImageName, "SegmentedImageName", this->GetComponentLabel(), 0, -1, false);

  using SegmentedImageType = typename Superclass1::SegmentedImageType;
  using ChangeInfoFilterType = itk::ChangeInformationImageFilter<SegmentedImageType>;
  using ChangeInfoFilterPointer = typename ChangeInfoFilterType::Pointer;
  using DirectionType = typename SegmentedImageType::DirectionType;
  using SizeValueType = typename SegmentedImageType::SizeType::SizeValueType;

  /** Possibly overrule the direction cosines. */
  ChangeInfoFilterPointer infoChanger = ChangeInfoFilterType::New();
  infoChanger->SetOutputDirection(DirectionType::GetIdentity());
  infoChanger->SetChangeDirection(!this->GetElastix()->GetUseDirectionCosines());

  /** Do the reading. */
  try
  {
    const auto image = itk::ReadImage<SegmentedImageType>(segmentedImageName);
    infoChanger->SetInput(image);
    infoChanger->Update();
  }
  catch (itk::ExceptionObject & excp)
  {
    /** Add information to the exception. */
    excp.SetLocation("MattesMutualInformationWithRigidityPenalty - BeforeRegistration()");
    std::string err_str = excp.GetDescription();
    err_str += "\nError occurred while reading the segmented image.\n";
    excp.SetDescription(err_str);
    /** Pass the exception to an higher level. */
    throw;
  }

  this->SetSegmentedImage(infoChanger->GetOutput());

  /** Get information from the segmented image. */
  typename SegmentedImageType::SizeType  segmentedImageSize = this->GetSegmentedImage()->GetBufferedRegion().GetSize();
  typename SegmentedImageType::PointType segmentedImageOrigin = this->GetSegmentedImage()->GetOrigin();
  typename SegmentedImageType::SpacingType segmentedImageSpacing = this->GetSegmentedImage()->GetSpacing();

  /** Get the grid sampling spacing for calculation of the rigidity penalty term. */
  typename SegmentedImageType::SpacingType penaltyGridSpacingInVoxels;
  for (unsigned int dim = 0; dim < FixedImageDimension; ++dim)
  {
    this->m_Configuration->ReadParameter(
      penaltyGridSpacingInVoxels[dim], "PenaltyGridSpacingInVoxels", this->GetComponentLabel(), dim, 0);
  }

  /** Compute resampled spacing and size. */
  typename SegmentedImageType::SpacingType resampledImageSpacing;
  typename SegmentedImageType::SizeType    resampledImageSize;
  for (unsigned int dim = 0; dim < FixedImageDimension; ++dim)
  {
    resampledImageSpacing[dim] = segmentedImageSpacing[dim] * penaltyGridSpacingInVoxels[dim];
    resampledImageSize[dim] = static_cast<SizeValueType>(segmentedImageSize[dim] / penaltyGridSpacingInVoxels[dim]);
  }

  /** Create resampler, identity transform and linear interpolator. */
  using ResampleFilterType = itk::ResampleImageFilter<SegmentedImageType, SegmentedImageType>;
  auto resampler = ResampleFilterType::New();

  using IdentityTransformType = itk::IdentityTransform<double, Superclass1::ImageDimension>;
  auto identityTransform = IdentityTransformType::New();
  identityTransform->SetIdentity();

  using LinearInterpolatorType = itk::LinearInterpolateImageFunction<SegmentedImageType, double>;
  auto linearInterpolator = LinearInterpolatorType::New();

  /** Configure the resampler and run it. */
  resampler->SetInterpolator(linearInterpolator);
  resampler->SetTransform(identityTransform);
  resampler->SetOutputSpacing(resampledImageSpacing);
  resampler->SetOutputOrigin(segmentedImageOrigin);
  resampler->SetSize(resampledImageSize);
  resampler->SetInput(this->GetSegmentedImage());
  resampler->Update();

  this->SetSampledSegmentedImage(resampler->GetOutput());

} // end BeforeRegistration()


/**
 * ******************* Initialize ***********************
 */

template <class TElastix>
void
DistancePreservingRigidityPenalty<TElastix>::Initialize()
{
  /** Initialize this class with the Superclass initializer. */
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();

  /** Stop and print the timer. */
  timer.Stop();
  elxout << "Initialization of DistancePreservingRigidityPenalty term took: "
         << static_cast<long>(timer.GetMean() * 1000) << " ms." << std::endl;

} // end Initialize()


} // end namespace elastix

#endif // end #ifndef elxDistancePreservingRigidityPenaltyTerm_hxx
