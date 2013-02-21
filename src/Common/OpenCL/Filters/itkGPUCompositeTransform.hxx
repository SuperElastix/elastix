/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUCompositeTransform_hxx
#define __itkGPUCompositeTransform_hxx

#include "itkGPUCompositeTransform.h"

namespace itk
{
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
size_t GPUCompositeTransform< TScalarType, NDimensions, TParentImageFilter >
::GetNumberOfCPUTransforms() const
{
  return CPUSuperclass::GetNumberOfTransforms();
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
TransformTypePointer GPUCompositeTransform< TScalarType, NDimensions, TParentImageFilter >
::GetNthTransform( SizeValueType n )
{
  return CPUSuperclass::GetNthTransform( n );
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
TransformTypeConstPointer GPUCompositeTransform< TScalarType, NDimensions, TParentImageFilter >
::GetNthTransform( SizeValueType n ) const
{
  return CPUSuperclass::GetNthTransform( n );
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
void GPUCompositeTransform< TScalarType, NDimensions, TParentImageFilter >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  CPUSuperclass::PrintSelf( os, indent );
}
} // end namespace itk

#endif /* __itkGPUCompositeTransform_hxx */
