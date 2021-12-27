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
#ifndef itkGPUBSplineBaseTransform_hxx
#define itkGPUBSplineBaseTransform_hxx

#include "itkGPUBSplineBaseTransform.h"

#include <iomanip>

//------------------------------------------------------------------------------
namespace itk
{
template <typename TScalarType, unsigned int NDimensions>
GPUBSplineBaseTransform<TScalarType, NDimensions>::GPUBSplineBaseTransform()
{
  this->m_SplineOrder = 3;

  // Add GPUBSplineTransform source
  const std::string sourcePath(GPUBSplineTransformKernel::GetOpenCLSource());
  this->m_Sources.push_back(sourcePath);
}


//------------------------------------------------------------------------------
template <typename TScalarType, unsigned int NDimensions>
auto
GPUBSplineBaseTransform<TScalarType, NDimensions>::GetGPUCoefficientImages() const -> const GPUCoefficientImageArray
{
  return this->m_GPUBSplineTransformCoefficientImages;
}


//------------------------------------------------------------------------------
template <typename TScalarType, unsigned int NDimensions>
auto
GPUBSplineBaseTransform<TScalarType, NDimensions>::GetGPUCoefficientImagesBases() const
  -> const GPUCoefficientImageBaseArray
{
  return this->m_GPUBSplineTransformCoefficientImagesBase;
}


//------------------------------------------------------------------------------
template <typename TScalarType, unsigned int NDimensions>
void
GPUBSplineBaseTransform<TScalarType, NDimensions>::SetSplineOrder(const unsigned int splineOrder)
{
  if (this->m_SplineOrder != splineOrder)
  {
    this->m_SplineOrder = splineOrder;
  }
}


//------------------------------------------------------------------------------
template <typename TScalarType, unsigned int NDimensions>
bool
GPUBSplineBaseTransform<TScalarType, NDimensions>::GetSourceCode(std::string & source) const
{
  if (this->m_Sources.empty())
  {
    return false;
  }

  // Create the final source code
  std::ostringstream sources;

  // Add other sources
  for (std::size_t i = 0; i < this->m_Sources.size(); ++i)
  {
    sources << this->m_Sources[i] << std::endl;
  }

  source = sources.str();
  return true;
}


} // end namespace itk

#endif /* itkGPUBSplineBaseTransform_hxx */
