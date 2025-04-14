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
#ifndef elxPatternIntensityMetric_hxx
#define elxPatternIntensityMetric_hxx

#include "elxPatternIntensityMetric.h"
#include "itkTimeProbe.h"
#include <itkDeref.h>

namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template <typename TElastix>
void
PatternIntensityMetric<TElastix>::Initialize()
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  log::info(std::ostringstream{} << "Initialization of PatternIntensity metric took: "
                                 << static_cast<std::int64_t>(timer.GetMean() * 1000) << " ms.");

} // end Initialize()


/**
 * ***************** BeforeRegistration ***********************
 */

template <typename TElastix>
void
PatternIntensityMetric<TElastix>::BeforeRegistration()
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
PatternIntensityMetric<TElastix>::BeforeEachResolution()
{
  const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());
  const std::string     componentLabel = BaseComponent::GetComponentLabel();

  /** Get the current resolution level.*/
  unsigned int level = (this->m_Registration->GetAsITKBaseType())->GetCurrentLevel();

  /** Set noise constant, */
  double sigma = 100;
  configuration.ReadParameter(sigma, "Sigma", componentLabel, level, 0);
  this->SetNoiseConstant(sigma * sigma);

  /** Set optimization of normalization factor. */
  bool optimizenormalizationfactor = false;
  configuration.ReadParameter(optimizenormalizationfactor, "OptimizeNormalizationFactor", componentLabel, level, 0);
  this->SetOptimizeNormalizationFactor(optimizenormalizationfactor);

  this->SetScales(this->m_Elastix->GetElxOptimizerBase()->GetAsITKBaseType()->GetScales());

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef elxPatternIntensityMetric_hxx
