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

// ITK GPU supported transforms
#include "itkGPUIdentityTransform.h"
#include "itkGPUMatrixOffsetTransformBase.h"
#include "itkGPUTranslationTransformBase.h"
#include "itkGPUBSplineTransform.h"

#include <iomanip>

//------------------------------------------------------------------------------
namespace itk
{
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
GPUCompositeTransform< TScalarType, NDimensions, TParentImageFilter >
::GPUCompositeTransform()
{
  m_SourcesLoaded = false;
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
GPUDataManager::Pointer GPUCompositeTransform< TScalarType, NDimensions, TParentImageFilter >
::GetParametersDataManager( const std::size_t index ) const
{
  GPUDataManager::Pointer parameters;

  if ( this->GetNumberOfTransforms() == 0 )
  {
    return parameters;
  }
  else
  {
    const GPUTransformBase *transformBase =
      dynamic_cast< const GPUTransformBase * >( this->GetNthTransform( index ).GetPointer() );

    if ( transformBase )
    {
      return transformBase->GetParametersDataManager();
    }
    else
    {
      itkExceptionMacro( << "Could not get GPU transform base." );
    }
  }
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
bool GPUCompositeTransform< TScalarType, NDimensions, TParentImageFilter >
::GetSourceCode( std::string & _source ) const
{
  // Create the final source code
  std::ostringstream sources;
  std::string        source;

  bool identityLoaded    = false;
  bool affineLoaded      = false;
  bool translationLoaded = false;
  bool bsplineLoaded     = false;

  // Add sources based on Transform type
  for ( std::size_t i = 0; i < this->GetNumberOfTransforms(); i++ )
  {
    if ( IsIdentityTransform( i, true, source ) && !identityLoaded )
    {
      sources << source << std::endl;
      identityLoaded = true;
    }

    if ( IsMatrixOffsetTransform( i, true, source ) && !affineLoaded )
    {
      sources << source << std::endl;
      affineLoaded = true;
    }

    if ( IsTranslationTransform( i, true, source ) && !translationLoaded )
    {
      sources << source << std::endl;
      translationLoaded = true;
    }

    if ( IsBSplineTransform( i, true, source ) && !bsplineLoaded )
    {
      sources << source << std::endl;
      bsplineLoaded = true;
    }
  }

  // Get final sources
  _source = sources.str();
  return true;
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
bool GPUCompositeTransform< TScalarType, NDimensions, TParentImageFilter >
::HasIdentityTransform() const
{
  for ( std::size_t i = 0; i < this->GetNumberOfTransforms(); i++ )
  {
    if ( IsIdentityTransform( i ) )
    {
      return true;
    }
  }

  return false;
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
bool GPUCompositeTransform< TScalarType, NDimensions, TParentImageFilter >
::HasMatrixOffsetTransform() const
{
  for ( std::size_t i = 0; i < this->GetNumberOfTransforms(); i++ )
  {
    if ( IsMatrixOffsetTransform( i ) )
    {
      return true;
    }
  }

  return false;
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
bool GPUCompositeTransform< TScalarType, NDimensions, TParentImageFilter >
::HasTranslationTransform() const
{
  for ( std::size_t i = 0; i < this->GetNumberOfTransforms(); i++ )
  {
    if ( IsTranslationTransform( i ) )
    {
      return true;
    }
  }

  return false;
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
bool GPUCompositeTransform< TScalarType, NDimensions, TParentImageFilter >
::HasBSplineTransform() const
{
  for ( std::size_t i = 0; i < this->GetNumberOfTransforms(); i++ )
  {
    if ( IsBSplineTransform( i ) )
    {
      return true;
    }
  }

  return false;
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
bool GPUCompositeTransform< TScalarType, NDimensions, TParentImageFilter >
::IsIdentityTransform( const std::size_t _index ) const
{
  typedef GPUIdentityTransform< ScalarType, NDimensions > IdentityTransformType;
  const IdentityTransformType *identity =
    dynamic_cast< IdentityTransformType * >( this->GetNthTransform( _index ).GetPointer() );
  if ( identity )
  {
    return true;
  }
  else
  {
    return false;
  }
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
bool GPUCompositeTransform< TScalarType, NDimensions, TParentImageFilter >
::IsMatrixOffsetTransform( const std::size_t _index ) const
{
  typedef GPUMatrixOffsetTransformBase<
      ScalarType, InputSpaceDimension, OutputSpaceDimension > MatrixOffsetTransformBaseType;
  const MatrixOffsetTransformBaseType *matrixoffset =
    dynamic_cast< MatrixOffsetTransformBaseType * >( this->GetNthTransform( _index ).GetPointer() );
  if ( matrixoffset )
  {
    return true;
  }
  else
  {
    return false;
  }
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
bool GPUCompositeTransform< TScalarType, NDimensions, TParentImageFilter >
::IsTranslationTransform( const std::size_t _index ) const
{
  typedef GPUTranslationTransformBase<
    ScalarType, InputSpaceDimension > TranslationTransformBaseType;
  const TranslationTransformBaseType *translation =
    dynamic_cast< TranslationTransformBaseType * >( this->GetNthTransform( _index ).GetPointer() );
  if ( translation )
  {
    return true;
  }
  else
  {
    return false;
  }
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
bool GPUCompositeTransform< TScalarType, NDimensions, TParentImageFilter >
::IsBSplineTransform( const std::size_t _index ) const
{
  typedef GPUBSplineTransform< ScalarType, NDimensions, 1 > BSplineTransformType1;
  typedef GPUBSplineTransform< ScalarType, NDimensions, 2 > BSplineTransformType2;
  typedef GPUBSplineTransform< ScalarType, NDimensions, 3 > BSplineTransformType3;

  const BSplineTransformType1 *bspline1 =
    dynamic_cast< BSplineTransformType1 * >( this->GetNthTransform( _index ).GetPointer() );
  const BSplineTransformType2 *bspline2 =
    dynamic_cast< BSplineTransformType2 * >( this->GetNthTransform( _index ).GetPointer() );
  const BSplineTransformType3 *bspline3 =
    dynamic_cast< BSplineTransformType3 * >( this->GetNthTransform( _index ).GetPointer() );

  if ( bspline1 || bspline2 || bspline3 )
  {
    return true;
  }
  else
  {
    return false;
  }
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
bool GPUCompositeTransform< TScalarType, NDimensions, TParentImageFilter >
::IsIdentityTransform( const std::size_t _index,
                       const bool _loadSource, std::string & _source ) const
{
  typedef GPUIdentityTransform< ScalarType, NDimensions > IdentityTransformType;
  const IdentityTransformType *identity =
    dynamic_cast< IdentityTransformType * >( this->GetNthTransform( _index ).GetPointer() );
  if ( identity )
  {
    if ( _loadSource )
    {
      const GPUTransformBase *transformBase =
        dynamic_cast< const GPUTransformBase * >( this->GetNthTransform( _index ).GetPointer() );
      transformBase->GetSourceCode( _source );
    }
    return true;
  }
  else
  {
    return false;
  }
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
bool GPUCompositeTransform< TScalarType, NDimensions, TParentImageFilter >
::IsMatrixOffsetTransform( const std::size_t _index,
                           const bool _loadSource, std::string & _source ) const
{
  typedef GPUMatrixOffsetTransformBase<
      ScalarType, InputSpaceDimension, OutputSpaceDimension > MatrixOffsetTransformBaseType;
  const MatrixOffsetTransformBaseType *matrixoffset =
    dynamic_cast< MatrixOffsetTransformBaseType * >( this->GetNthTransform( _index ).GetPointer() );
  if ( matrixoffset )
  {
    if ( _loadSource )
    {
      const GPUTransformBase *transformBase =
        dynamic_cast< const GPUTransformBase * >( this->GetNthTransform( _index ).GetPointer() );
      transformBase->GetSourceCode( _source );
    }
    return true;
  }
  else
  {
    return false;
  }
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
bool GPUCompositeTransform< TScalarType, NDimensions, TParentImageFilter >
::IsTranslationTransform( const std::size_t _index,
                          const bool _loadSource, std::string & _source ) const
{
  typedef GPUTranslationTransformBase<
    ScalarType, InputSpaceDimension > TranslationTransformBaseType;
  const TranslationTransformBaseType *translation =
    dynamic_cast< TranslationTransformBaseType * >( this->GetNthTransform( _index ).GetPointer() );
  if ( translation )
  {
    if ( _loadSource )
    {
      const GPUTransformBase *transformBase =
        dynamic_cast< const GPUTransformBase * >( this->GetNthTransform( _index ).GetPointer() );
      transformBase->GetSourceCode( _source );
    }
    return true;
  }
  else
  {
    return false;
  }
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
bool GPUCompositeTransform< TScalarType, NDimensions, TParentImageFilter >
::IsBSplineTransform( const std::size_t _index,
                      const bool _loadSource, std::string & _source ) const
{
  typedef GPUBSplineTransform< ScalarType, NDimensions, 1 > BSplineTransformType1;
  typedef GPUBSplineTransform< ScalarType, NDimensions, 2 > BSplineTransformType2;
  typedef GPUBSplineTransform< ScalarType, NDimensions, 3 > BSplineTransformType3;

  const BSplineTransformType1 *bspline1 =
    dynamic_cast< BSplineTransformType1 * >( this->GetNthTransform( _index ).GetPointer() );
  const BSplineTransformType2 *bspline2 =
    dynamic_cast< BSplineTransformType2 * >( this->GetNthTransform( _index ).GetPointer() );
  const BSplineTransformType3 *bspline3 =
    dynamic_cast< BSplineTransformType3 * >( this->GetNthTransform( _index ).GetPointer() );

  if ( bspline1 || bspline2 || bspline3 )
  {
    if ( _loadSource )
    {
      const GPUTransformBase *transformBase =
        dynamic_cast< const GPUTransformBase * >( this->GetNthTransform( _index ).GetPointer() );
      transformBase->GetSourceCode( _source );
    }
    return true;
  }
  else
  {
    return false;
  }
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
void GPUCompositeTransform< TScalarType, NDimensions, TParentImageFilter >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
}
} // end namespace itk

#endif /* __itkGPUCompositeTransform_hxx */
