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
#ifndef itkGPUIdentityTransform_hxx
#define itkGPUIdentityTransform_hxx

#include "itkGPUIdentityTransform.h"
#include "itkGPUMatrixOffsetTransformBase.h"
#include <iomanip>

namespace itk
{
template <typename TScalarType, unsigned int NDimensions, typename TParentTransform>
GPUIdentityTransform<TScalarType, NDimensions, TParentTransform>::GPUIdentityTransform()
{
  // Add GPUIdentityTransform source
  const std::string sourcePath(GPUIdentityTransformKernel::GetOpenCLSource());
  this->m_Sources.push_back(sourcePath);
}


//------------------------------------------------------------------------------
template <typename TScalarType, unsigned int NDimensions, typename TParentTransform>
bool
GPUIdentityTransform<TScalarType, NDimensions, TParentTransform>::GetSourceCode(std::string & source) const
{
  if (this->m_Sources.empty())
  {
    return false;
  }

  // Create the final source code
  std::ostringstream sources;
  for (std::size_t i = 0; i < this->m_Sources.size(); ++i)
  {
    sources << this->m_Sources[i] << std::endl;
  }
  source = sources.str();
  return true;
}


//------------------------------------------------------------------------------
template <typename TScalarType, unsigned int NDimensions, typename TParentTransform>
void
GPUIdentityTransform<TScalarType, NDimensions, TParentTransform>::PrintSelf(std::ostream & os, Indent indent) const
{
  CPUSuperclass::PrintSelf(os, indent);
}


} // end namespace itk

#endif /* itkGPUIdentityTransform_hxx */
