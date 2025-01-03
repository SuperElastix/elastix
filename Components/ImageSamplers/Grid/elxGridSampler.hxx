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

#ifndef elxGridSampler_hxx
#define elxGridSampler_hxx

#include "elxGridSampler.h"
#include <itkDeref.h>

namespace elastix
{

/**
 * ******************* BeforeEachResolution ******************
 */

template <typename TElastix>
void
GridSampler<TElastix>::BeforeEachResolution()
{
  const Configuration & configuration = itk::Deref(Superclass2::GetConfiguration());

  const unsigned int level = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  GridSpacingType gridSpacing;

  /** Read the desired grid spacing of the samples. */
  unsigned int spacing_dim;
  for (unsigned int dim = 0; dim < InputImageDimension; ++dim)
  {
    spacing_dim = 2;
    configuration.ReadParameter(
      spacing_dim, "SampleGridSpacing", this->GetComponentLabel(), level * InputImageDimension + dim, -1);
    gridSpacing[dim] = static_cast<SampleGridSpacingValueType>(spacing_dim);
  }
  this->SetSampleGridSpacing(gridSpacing);

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef elxGridSampler_hxx
