/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUIdentityTransform_hxx
#define __itkGPUIdentityTransform_hxx

#include "itkGPUIdentityTransform.h"
#include "itkGPUMatrixOffsetTransformBase.h"
#include <iomanip>

namespace itk
{
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
GPUIdentityTransform< TScalarType, NDimensions, TParentImageFilter >::GPUIdentityTransform()
{
  // Add GPUIdentityTransform source
  const std::string sourcePath(
    GPUIdentityTransformKernel::GetOpenCLSource() );

  m_Sources.push_back( sourcePath );

  m_SourcesLoaded = true; // we set it to true, sources are loaded from strings
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
bool GPUIdentityTransform< TScalarType, NDimensions, TParentImageFilter >
::GetSourceCode( std::string & _source ) const
{
  if ( !m_SourcesLoaded )
  {
    return false;
  }

  // Create the final source code
  std::ostringstream source;
  for ( std::size_t i = 0; i < m_Sources.size(); i++ )
  {
    source << m_Sources[i] << std::endl;
  }
  _source = source.str();
  return true;
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
void GPUIdentityTransform< TScalarType, NDimensions, TParentImageFilter >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  CPUSuperclass::PrintSelf( os, indent );
}
} // end namespace itk

#endif /* __itkGPUIdentityTransform_hxx */
