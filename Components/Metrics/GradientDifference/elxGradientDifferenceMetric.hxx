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
#ifndef elxGradientDifferenceMetric_hxx
#define elxGradientDifferenceMetric_hxx

#include "elxGradientDifferenceMetric.h"
#include "itkTimeProbe.h"

namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template <typename TElastix>
void
GradientDifferenceMetric<TElastix>::Initialize()
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  log::info(std::ostringstream{} << "Initialization of GradientDifference metric took: "
                                 << static_cast<std::int64_t>(timer.GetMean() * 1000) << " ms.");

} // end Initialize()


/**
 * ***************** BeforeRegistration ***********************
 */

template <typename TElastix>
void
GradientDifferenceMetric<TElastix>::BeforeRegistration()
{

  if constexpr (TElastix::FixedDimension != 3)
  {
    itkExceptionMacro("FixedImage must be 3D");
  }
  if constexpr (TElastix::FixedDimension == 3)
  {
    if (this->m_Elastix->GetFixedImage()->GetLargestPossibleRegion().GetSize()[2] != 1)
    {
      itkExceptionMacro("Metric can only be used for 2D-3D registration. FixedImageSize[2] must be 1");
    }
  }

} // end BeforeRegistration()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <typename TElastix>
void
GradientDifferenceMetric<TElastix>::BeforeEachResolution()
{
  this->SetScales(this->m_Elastix->GetElxOptimizerBase()->GetAsITKBaseType()->GetScales());

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef elxGradientDifferenceMetric_hxx
