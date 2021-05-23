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
#ifndef itkGPULinearInterpolateImageFunction_hxx
#define itkGPULinearInterpolateImageFunction_hxx

#include "itkGPULinearInterpolateImageFunction.h"
#include "itkGPUImageFunction.h"
#include <iomanip>

namespace itk
{
template <typename TInputImage, typename TCoordRep>
GPULinearInterpolateImageFunction<TInputImage, TCoordRep>::GPULinearInterpolateImageFunction()
{
  // Add GPUImageFunction implementation
  const std::string sourcePath0(GPUImageFunctionKernel::GetOpenCLSource());
  this->m_Sources.push_back(sourcePath0);

  // Add GPULinearInterpolateImageFunction implementation
  const std::string sourcePath1(GPULinearInterpolateImageFunctionKernel::GetOpenCLSource());
  this->m_Sources.push_back(sourcePath1);
}


//------------------------------------------------------------------------------
template <typename TInputImage, typename TCoordRep>
bool
GPULinearInterpolateImageFunction<TInputImage, TCoordRep>::GetSourceCode(std::string & source) const
{
  if (this->m_Sources.empty())
  {
    return false;
  }

  // Create the source code
  std::ostringstream sources;
  for (std::size_t i = 0; i < this->m_Sources.size(); ++i)
  {
    sources << this->m_Sources[i] << std::endl;
  }

  source = sources.str();
  return true;
}


//------------------------------------------------------------------------------
template <typename TInputImage, typename TCoordRep>
void
GPULinearInterpolateImageFunction<TInputImage, TCoordRep>::PrintSelf(std::ostream & os, Indent indent) const
{
  CPUSuperclass::PrintSelf(os, indent);
  GPUSuperclass::PrintSelf(os, indent);
}


} // end namespace itk

#endif /* itkGPULinearInterpolateImageFunction_hxx */
