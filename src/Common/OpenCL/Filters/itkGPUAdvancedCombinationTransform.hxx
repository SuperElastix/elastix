/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUAdvancedCombinationTransform_hxx
#define __itkGPUAdvancedCombinationTransform_hxx

#include "itkGPUAdvancedCombinationTransform.h"
#include "itkGPUMatrixOffsetTransformBase.h"
#include <iomanip>

//------------------------------------------------------------------------------
namespace itk
{
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
GPUAdvancedCombinationTransform< TScalarType, NDimensions, TParentImageFilter >
::GPUAdvancedCombinationTransform()
//:Superclass( NDimensions )
{
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
bool GPUAdvancedCombinationTransform< TScalarType, NDimensions, TParentImageFilter >
::GetSourceCode( std::string & _source ) const
{
  if ( !m_SourcesLoaded )
  {
    return false;
  }

  // Create the final source code
  std::ostringstream source;
  // Add other sources
  for ( unsigned int i = 0; i < m_Sources.size(); i++ )
  {
    source << m_Sources[i] << std::endl;
  }
  _source = source.str();
  return true;
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
void GPUAdvancedCombinationTransform< TScalarType, NDimensions, TParentImageFilter >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
}

} // end namespace itk

#endif /* __itkGPUAdvancedCombinationTransform_hxx */
