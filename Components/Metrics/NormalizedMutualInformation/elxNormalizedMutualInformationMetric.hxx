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
#ifndef elxNormalizedMutualInformationMetric_hxx
#define elxNormalizedMutualInformationMetric_hxx

#include "elxNormalizedMutualInformationMetric.h"

#include "itkHardLimiterFunction.h"
#include "itkExponentialLimiterFunction.h"
#include "itkTimeProbe.h"
#include <string>

namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template <class TElastix>
void
NormalizedMutualInformationMetric<TElastix>::Initialize()
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  elxout << "Initialization of NormalizedMutualInformation metric took: " << static_cast<long>(timer.GetMean() * 1000)
         << " ms." << std::endl;

} // end Initialize()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void
NormalizedMutualInformationMetric<TElastix>::BeforeEachResolution()
{
  /** Get the current resolution level. */
  unsigned int level = (this->m_Registration->GetAsITKBaseType())->GetCurrentLevel();

  /** Get and set the number of histogram bins. */
  unsigned int numberOfHistogramBins = 32;
  this->GetConfiguration()->ReadParameter(
    numberOfHistogramBins, "NumberOfHistogramBins", this->GetComponentLabel(), level, 0);
  this->SetNumberOfFixedHistogramBins(numberOfHistogramBins);
  this->SetNumberOfMovingHistogramBins(numberOfHistogramBins);

  unsigned int numberOfFixedHistogramBins = numberOfHistogramBins;
  unsigned int numberOfMovingHistogramBins = numberOfHistogramBins;
  this->GetConfiguration()->ReadParameter(
    numberOfFixedHistogramBins, "NumberOfFixedHistogramBins", this->GetComponentLabel(), level, 0);
  this->GetConfiguration()->ReadParameter(
    numberOfMovingHistogramBins, "NumberOfMovingHistogramBins", this->GetComponentLabel(), level, 0);
  this->SetNumberOfFixedHistogramBins(numberOfFixedHistogramBins);
  this->SetNumberOfMovingHistogramBins(numberOfMovingHistogramBins);

  /** Set limiters */
  using FixedLimiterType = itk::HardLimiterFunction<RealType, FixedImageDimension>;
  using MovingLimiterType = itk::ExponentialLimiterFunction<RealType, MovingImageDimension>;
  this->SetFixedImageLimiter(FixedLimiterType::New());
  this->SetMovingImageLimiter(MovingLimiterType::New());

  /** Get and set the number of histogram bins. */
  double fixedLimitRangeRatio = 0.01;
  double movingLimitRangeRatio = 0.01;
  this->GetConfiguration()->ReadParameter(
    fixedLimitRangeRatio, "FixedLimitRangeRatio", this->GetComponentLabel(), level, 0);
  this->GetConfiguration()->ReadParameter(
    movingLimitRangeRatio, "MovingLimitRangeRatio", this->GetComponentLabel(), level, 0);
  this->SetFixedLimitRangeRatio(fixedLimitRangeRatio);
  this->SetMovingLimitRangeRatio(movingLimitRangeRatio);

  /** Set B-spline parzen kernel orders */
  unsigned int fixedKernelBSplineOrder = 0;
  unsigned int movingKernelBSplineOrder = 3;
  this->GetConfiguration()->ReadParameter(
    fixedKernelBSplineOrder, "FixedKernelBSplineOrder", this->GetComponentLabel(), level, 0);
  this->GetConfiguration()->ReadParameter(
    movingKernelBSplineOrder, "MovingKernelBSplineOrder", this->GetComponentLabel(), level, 0);
  this->SetFixedKernelBSplineOrder(fixedKernelBSplineOrder);
  this->SetMovingKernelBSplineOrder(movingKernelBSplineOrder);

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef elxNormalizedMutualInformationMetric_hxx
