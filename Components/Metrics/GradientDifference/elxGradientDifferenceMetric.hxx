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

template <class TElastix>
void
GradientDifferenceMetric<TElastix>::Initialize()
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  elxout << "Initialization of GradientDifference metric took: " << static_cast<long>(timer.GetMean() * 1000) << " ms."
         << std::endl;

} // end Initialize()


/**
 * ***************** BeforeRegistration ***********************
 */

template <class TElastix>
void
GradientDifferenceMetric<TElastix>::BeforeRegistration()
{

  if (this->m_Elastix->GetFixedImage()->GetImageDimension() != 3)
  {
    itkExceptionMacro(<< "FixedImage must be 3D");
  }
  if (this->m_Elastix->GetFixedImage()->GetImageDimension() == 3)
  {
    if (this->m_Elastix->GetFixedImage()->GetLargestPossibleRegion().GetSize()[2] != 1)
    {
      itkExceptionMacro(<< "Metric can only be used for 2D-3D registration. FixedImageSize[2] must be 1");
    }
  }

} // end BeforeRegistration()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void
GradientDifferenceMetric<TElastix>::BeforeEachResolution()
{
  using ScalesType = typename elastix::OptimizerBase<TElastix>::ITKBaseType::ScalesType;
  ScalesType scales = this->m_Elastix->GetElxOptimizerBase()->GetAsITKBaseType()->GetScales();
  this->SetScales(scales);

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef elxGradientDifferenceMetric_hxx
