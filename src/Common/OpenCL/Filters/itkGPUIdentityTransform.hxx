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
#ifndef __itkGPUIdentityTransform_hxx
#define __itkGPUIdentityTransform_hxx

#include "itkGPUIdentityTransform.h"
#include "itkGPUKernelManagerHelperFunctions.h"
#include <iomanip>

namespace itk
{
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
GPUIdentityTransform< TScalarType, NDimensions, TParentImageFilter >::GPUIdentityTransform()
{
  // Load GPUMatrixOffsetTransformBase header
  std::string sname = "GPUMatrixOffsetTransformBase header";
  const std::string sourcePath0(oclhGPUMatrixOffsetTransformBase);
  m_SourcesLoaded = LoadProgramFromFile(sourcePath0, m_Sources, sname, true);
  if(!m_SourcesLoaded)
  {
    itkGenericExceptionMacro( << sname << " has not been loaded from: " << sourcePath0 );
  }

  // Load GPUIdentityTransform source
  sname = "GPUIdentityTransform source";
  const std::string sourcePath1(oclGPUIdentityTransform);
  m_SourcesLoaded = m_SourcesLoaded && LoadProgramFromFile(sourcePath1, m_Sources, sname, true);
  if(!m_SourcesLoaded)
  {
    itkGenericExceptionMacro( << sname << " has not been loaded from: " << sourcePath1 );
  }
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
bool GPUIdentityTransform< TScalarType, NDimensions, TParentImageFilter >
::GetSourceCode(std::string &_source) const
{
  if(!m_SourcesLoaded)
    return false;

  // Create the final source code
  std::ostringstream source;
  for(unsigned int i=0; i<m_Sources.size(); i++)
  {
    source << m_Sources[i] << std::endl;
  }
  _source = source.str();
  return true;
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
void GPUIdentityTransform< TScalarType, NDimensions, TParentImageFilter >
::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}

} // namespace

#endif
