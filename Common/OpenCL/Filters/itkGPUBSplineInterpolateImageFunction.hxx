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
#ifndef itkGPUBSplineInterpolateImageFunction_hxx
#define itkGPUBSplineInterpolateImageFunction_hxx

#include "itkGPUBSplineInterpolateImageFunction.h"
#include "itkGPUImageFunction.h"
#include <iomanip>

namespace itk
{
template <typename TInputImage, typename TCoordRep, typename TCoefficientType>
GPUBSplineInterpolateImageFunction<TInputImage, TCoordRep, TCoefficientType>::GPUBSplineInterpolateImageFunction()
{
  // Create GPU coefficients image
  this->m_GPUCoefficients = GPUCoefficientImageType::New();
  this->m_GPUCoefficientsImageBase = GPUDataManager::New();

  // Add GPUImageFunction implementation
  const std::string sourcePath0(GPUImageFunctionKernel::GetOpenCLSource());
  this->m_Sources.push_back(sourcePath0);

  // Add GPUBSplineInterpolateImageFunction implementation
  const std::string sourcePath1(GPUBSplineInterpolateImageFunctionKernel::GetOpenCLSource());
  this->m_Sources.push_back(sourcePath1);
}


//------------------------------------------------------------------------------
template <typename TInputImage, typename TCoordRep, typename TCoefficientType>
void
GPUBSplineInterpolateImageFunction<TInputImage, TCoordRep, TCoefficientType>::SetInputImage(
  const TInputImage * inputData)
{
  Superclass::SetInputImage(inputData);
  m_GPUCoefficients->Graft(this->m_Coefficients);
}


//------------------------------------------------------------------------------
template <typename TInputImage, typename TCoordRep, typename TCoefficientType>
auto
GPUBSplineInterpolateImageFunction<TInputImage, TCoordRep, TCoefficientType>::GetGPUCoefficients() const
  -> const GPUCoefficientImagePointer
{
  return this->m_GPUCoefficients;
}


//------------------------------------------------------------------------------
template <typename TInputImage, typename TCoordRep, typename TCoefficientType>
auto
GPUBSplineInterpolateImageFunction<TInputImage, TCoordRep, TCoefficientType>::GetGPUCoefficientsImageBase() const
  -> const GPUDataManagerPointer
{
  return this->m_GPUCoefficientsImageBase;
}


//------------------------------------------------------------------------------
template <typename TInputImage, typename TCoordRep, typename TCoefficientType>
bool
GPUBSplineInterpolateImageFunction<TInputImage, TCoordRep, TCoefficientType>::GetSourceCode(std::string & source) const
{
  if (this->m_Sources.empty())
  {
    return false;
  }

  // Create the source code
  std::ostringstream sources;

  // Add other sources
  for (std::size_t i = 0; i < this->m_Sources.size(); ++i)
  {
    sources << this->m_Sources[i] << std::endl;
  }

  source = sources.str();
  return true;
}


//------------------------------------------------------------------------------
template <typename TInputImage, typename TCoordRep, typename TCoefficientType>
void
GPUBSplineInterpolateImageFunction<TInputImage, TCoordRep, TCoefficientType>::PrintSelf(std::ostream & os,
                                                                                        Indent         indent) const
{
  GPUSuperclass::PrintSelf(os, indent);
}


} // end namespace itk

#endif /* itkGPUBSplineInterpolateImageFunction_hxx */
