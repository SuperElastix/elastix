/*=========================================================================
*
*  Copyright Insight Software Consortium
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*         http://www.apache.org/licenses/LICENSE-2.0.txt
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
*=========================================================================*/
#ifndef __itkGPUBSplineInterpolateImageFunction_hxx
#define __itkGPUBSplineInterpolateImageFunction_hxx

#include "itkGPUBSplineInterpolateImageFunction.h"
#include "itkGPUKernelManagerHelperFunctions.h"
#include <iomanip>

namespace itk
{
template< class TInputImage, class TCoordRep, class TCoefficientType >
GPUBSplineInterpolateImageFunction< TInputImage, TCoordRep, TCoefficientType >
::GPUBSplineInterpolateImageFunction()
{
  // Load GPUImageFunction implementation
  std::string sname = "GPUImageFunction implementation";
  const std::string sourcePath0(oclGPUImageFunction);
  m_SourcesLoaded = LoadProgramFromFile(sourcePath0, m_Sources, sname, true);
  if(!m_SourcesLoaded)
  {
    itkGenericExceptionMacro( << sname << " has not been loaded from: " << sourcePath0 );
  }

  // Load GPUBSplineInterpolateImageFunction implementation
  sname = "GPUBSplineInterpolateImageFunction implementation";
  const std::string sourcePath1(oclGPUBSplineInterpolateImageFunction);
  m_SourcesLoaded = m_SourcesLoaded && LoadProgramFromFile(sourcePath1, m_Sources, sname, true);
  if(!m_SourcesLoaded)
  {
    itkGenericExceptionMacro( << sname << " has not been loaded from: " << sourcePath1 );
  }
}

//------------------------------------------------------------------------------
template< class TInputImage, class TCoordRep, class TCoefficientType >
bool GPUBSplineInterpolateImageFunction< TInputImage, TCoordRep, TCoefficientType >
::GetSourceCode(std::string &_source) const
{
  if(!m_SourcesLoaded)
    return false;

  // Create the source code
  std::ostringstream source;

  // Variable length array declaration not allowed in OpenCL, therefore we are using #define
  source << "#define GPUBSplineOrder (" << this->m_SplineOrder << ")" << std::endl;

  // Calculate MaxNumberInterpolationPoints
  unsigned int maxNumberInterpolationPoints = 1;
  for(unsigned int n = 0; n < InputImageDimension; n++)
  {
    maxNumberInterpolationPoints *= (this->m_SplineOrder + 1);
  }
  // Variable length array declaration not allowed in OpenCL, therefore we are using #define
  source << "#define GPUMaxNumberInterpolationPoints (" << maxNumberInterpolationPoints << ")" << std::endl;

  // Add other sources
  for(std::size_t i=0; i<m_Sources.size(); i++)
  {
    source << m_Sources[i] << std::endl;
  }

  _source = source.str();
  return true;
}

//------------------------------------------------------------------------------
template< class TInputImage, class TCoordRep, class TCoefficientType >
void GPUBSplineInterpolateImageFunction< TInputImage, TCoordRep, TCoefficientType >
::PrintSelf(std::ostream & os, Indent indent) const
{
  //CPUSuperclass::PrintSelf(os, indent);
  GPUSuperclass::PrintSelf(os, indent);
}

} // namespace

#endif
