/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPULinearInterpolateImageFunction_hxx
#define __itkGPULinearInterpolateImageFunction_hxx

#include "itkGPULinearInterpolateImageFunction.h"
#include "itkGPUImageFunction.h"
#include <iomanip>

namespace itk
{
template< class TInputImage, class TCoordRep >
GPULinearInterpolateImageFunction< TInputImage, TCoordRep >
::GPULinearInterpolateImageFunction()
{
  // Add GPUImageFunction implementation
  const std::string sourcePath0(GPUImageFunctionKernel::GetOpenCLSource());
  m_Sources.push_back(sourcePath0);

  // Add GPULinearInterpolateImageFunction implementation
  const std::string sourcePath1(GPULinearInterpolateImageFunctionKernel::GetOpenCLSource());
  m_Sources.push_back(sourcePath1);

  m_SourcesLoaded = true; // we set it to true, sources are loaded from strings
}

//------------------------------------------------------------------------------
template< class TInputImage, class TCoordRep >
bool GPULinearInterpolateImageFunction< TInputImage, TCoordRep >
::GetSourceCode(std::string &_source) const
{
  if(!m_SourcesLoaded)
    return false;

  // Create the source code
  std::ostringstream source;
  for(std::size_t i=0; i<m_Sources.size(); i++)
  {
    source << m_Sources[i] << std::endl;
  }

  _source = source.str();
  return true;
}

//------------------------------------------------------------------------------
template< class TInputImage, class TCoordRep >
void GPULinearInterpolateImageFunction< TInputImage, TCoordRep >
::PrintSelf(std::ostream & os, Indent indent) const
{
  CPUSuperclass::PrintSelf(os, indent);
  GPUSuperclass::PrintSelf(os, indent);
}

} // namespace

#endif
