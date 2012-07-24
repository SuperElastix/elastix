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
#include "itkGPUKernelManagerHelperFunctions.h"
#include <iomanip>

namespace itk
{
template< class TInputImage, class TCoordRep >
GPULinearInterpolateImageFunction< TInputImage, TCoordRep >
::GPULinearInterpolateImageFunction()
{
  // Load GPUImageFunction implementation
  std::string sname = "GPUImageFunction implementation";
  const std::string sourcePath0(oclGPUImageFunction);
  m_SourcesLoaded = LoadProgramFromFile(sourcePath0, m_Sources, sname, true);
  if(!m_SourcesLoaded)
  {
    itkGenericExceptionMacro( << sname << " has not been loaded from: " << sourcePath0 );
  }

  // Load GPULinearInterpolateImageFunction implementation
  sname = "GPULinearInterpolateImageFunction implementation";
  const std::string sourcePath1(oclGPULinearInterpolateImageFunction);
  m_SourcesLoaded = m_SourcesLoaded && LoadProgramFromFile(sourcePath1, m_Sources, sname, true);
  if(!m_SourcesLoaded)
  {
    itkGenericExceptionMacro( << sname << " has not been loaded from: " << sourcePath1 );
  }
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
