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

namespace itk
{
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
size_t GPUAdvancedCombinationTransform< TScalarType, NDimensions, TParentImageFilter >
::GetNumberOfTransforms() const
{
  size_t                       num = 0;
  CurrentTransformConstPointer currentTransform = CPUSuperclass::GetCurrentTransform();

  while ( currentTransform.IsNotNull() )
  {
    const CPUSuperclass *currentTransformCasted =
      dynamic_cast< const CPUSuperclass * >( currentTransform.GetPointer() );

    if ( currentTransformCasted )
    {
      InitialTransformConstPointer initialTransform = currentTransformCasted->GetInitialTransform();
      const CPUSuperclass *        initialTransformCasted =
        dynamic_cast< const CPUSuperclass * >( initialTransform.GetPointer() );

      if ( initialTransformCasted )
      {
        currentTransform = initialTransformCasted->GetCurrentTransform();
        if ( currentTransform.IsNotNull() )
        {
          num++;
        }
      }
    }
  }

  return num;
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
typename GPUAdvancedCombinationTransform< TScalarType, NDimensions, TParentImageFilter >::TransformTypePointer
GPUAdvancedCombinationTransform< TScalarType, NDimensions, TParentImageFilter >
::GetNthTransform( SizeValueType n )
{
  const size_t numTransforms = GetNumberOfTransforms();

  if ( n > numTransforms - 1 )
  {
    itkExceptionMacro( << " The AdvancedCombinationTransform contains "
                       << numTransforms << " transforms. Unable to retrieve Nth transform with index " << n );
  }

  TransformTypePointer    nthTransform;
  size_t                  currentItemID = 0;
  CurrentTransformPointer currentTransform = CPUSuperclass::GetCurrentTransform();

  while ( currentTransform.IsNotNull() )
  {
    if ( currentItemID == n )
    {
      nthTransform = currentTransform;
      return nthTransform;
    }

    CPUSuperclass *currentTransformCasted =
      dynamic_cast< CPUSuperclass * >( currentTransform.GetPointer() );

    if ( currentTransformCasted )
    {
      InitialTransformPointer initialTransform = currentTransformCasted->GetInitialTransform();
      CPUSuperclass *         initialTransformCasted =
        dynamic_cast< CPUSuperclass * >( initialTransform.GetPointer() );

      if ( initialTransformCasted )
      {
        currentTransform = initialTransformCasted->GetCurrentTransform();
        if ( currentTransform.IsNotNull() )
        {
          currentItemID++;
        }
      }
    }
  }

  return nthTransform;
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
typename GPUAdvancedCombinationTransform< TScalarType, NDimensions, TParentImageFilter >::TransformTypeConstPointer
GPUAdvancedCombinationTransform< TScalarType, NDimensions, TParentImageFilter >
::GetNthTransform( SizeValueType n ) const
{
  const size_t numTransforms = GetNumberOfTransforms();

  if ( n > numTransforms - 1 )
  {
    itkExceptionMacro( << " The AdvancedCombinationTransform contains "
                       << numTransforms << " transforms. Unable to retrieve Nth transform with index " << n );
  }

  TransformTypeConstPointer    nthTransform;
  size_t                       currentItemID = 0;
  CurrentTransformConstPointer currentTransform = CPUSuperclass::GetCurrentTransform();

  while ( currentTransform.IsNotNull() )
  {
    if ( currentItemID == n )
    {
      nthTransform = currentTransform;
      return nthTransform;
    }

    const CPUSuperclass *currentTransformCasted =
      dynamic_cast< const CPUSuperclass * >( currentTransform.GetPointer() );

    if ( currentTransformCasted )
    {
      InitialTransformConstPointer initialTransform = currentTransformCasted->GetInitialTransform();
      const CPUSuperclass *        initialTransformCasted =
        dynamic_cast< const CPUSuperclass * >( initialTransform.GetPointer() );

      if ( initialTransformCasted )
      {
        currentTransform = initialTransformCasted->GetCurrentTransform();
        if ( currentTransform.IsNotNull() )
        {
          currentItemID++;
        }
      }
    }
  }

  return nthTransform;
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
void GPUAdvancedCombinationTransform< TScalarType, NDimensions, TParentImageFilter >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  CPUSuperclass::PrintSelf( os, indent );
}
} // end namespace itk

#endif /* __itkGPUAdvancedCombinationTransform_hxx */
