/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUCompositeTransformBase_hxx
#define __itkGPUCompositeTransformBase_hxx

#include "itkGPUCompositeTransformBase.h"

// ITK GPU supported transforms
#include "itkGPUIdentityTransform.h"
#include "itkGPUMatrixOffsetTransformBase.h"
#include "itkGPUTranslationTransformBase.h"

#include <iomanip>

//------------------------------------------------------------------------------
namespace itk
{
template< class TScalarType, unsigned int NDimensions >
GPUCompositeTransformBase< TScalarType, NDimensions >
::GPUCompositeTransformBase()
{
  m_SourcesLoaded = false;
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions >
GPUDataManager::Pointer GPUCompositeTransformBase< TScalarType, NDimensions >
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
template< class TScalarType, unsigned int NDimensions >
bool GPUCompositeTransformBase< TScalarType, NDimensions >
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
template< class TScalarType, unsigned int NDimensions >
bool GPUCompositeTransformBase< TScalarType, NDimensions >
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
template< class TScalarType, unsigned int NDimensions >
bool GPUCompositeTransformBase< TScalarType, NDimensions >
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
template< class TScalarType, unsigned int NDimensions >
bool GPUCompositeTransformBase< TScalarType, NDimensions >
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
template< class TScalarType, unsigned int NDimensions >
bool GPUCompositeTransformBase< TScalarType, NDimensions >
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
template< class TScalarType, unsigned int NDimensions >
bool GPUCompositeTransformBase< TScalarType, NDimensions >
::IsIdentityTransform( const std::size_t _index ) const
{
  // First quick check if Linear
  if ( this->GetNthTransform( _index )->GetTransformCategory() != TransformBase::Linear )
  {
    return false;
  }

  // Perform specific check
  typedef GPUIdentityTransform< ScalarType, NDimensions > IdentityTransformType;
  const IdentityTransformType *identity =
    dynamic_cast< const IdentityTransformType * >( this->GetNthTransform( _index ).GetPointer() );
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
template< class TScalarType, unsigned int NDimensions >
bool GPUCompositeTransformBase< TScalarType, NDimensions >
::IsMatrixOffsetTransform( const std::size_t _index ) const
{
  // First quick check if Linear
  if ( this->GetNthTransform( _index )->GetTransformCategory() != TransformBase::Linear )
  {
    return false;
  }

  // Perform specific check
  typedef GPUMatrixOffsetTransformBase<
      ScalarType, InputSpaceDimension, OutputSpaceDimension > MatrixOffsetTransformBaseType;
  const MatrixOffsetTransformBaseType *matrixoffset =
    dynamic_cast< const MatrixOffsetTransformBaseType * >( this->GetNthTransform( _index ).GetPointer() );
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
template< class TScalarType, unsigned int NDimensions >
bool GPUCompositeTransformBase< TScalarType, NDimensions >
::IsTranslationTransform( const std::size_t _index ) const
{
  // First quick check if Linear
  if ( this->GetNthTransform( _index )->GetTransformCategory() != TransformBase::Linear )
  {
    return false;
  }

  // Perform specific check
  typedef GPUTranslationTransformBase<
      ScalarType, InputSpaceDimension > TranslationTransformBaseType;
  const TranslationTransformBaseType *translation =
    dynamic_cast< const TranslationTransformBaseType * >( this->GetNthTransform( _index ).GetPointer() );
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
template< class TScalarType, unsigned int NDimensions >
bool GPUCompositeTransformBase< TScalarType, NDimensions >
::IsBSplineTransform( const std::size_t _index ) const
{
  if ( this->GetNthTransform( _index )->GetTransformCategory() == TransformBase::BSpline )
  {
    return true;
  }
  else
  {
    return false;
  }
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions >
bool GPUCompositeTransformBase< TScalarType, NDimensions >
::IsIdentityTransform( const std::size_t _index,
                       const bool _loadSource, std::string & _source ) const
{
  typedef GPUIdentityTransform< ScalarType, NDimensions > IdentityTransformType;
  const IdentityTransformType *identity =
    dynamic_cast< const IdentityTransformType * >( this->GetNthTransform( _index ).GetPointer() );
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
template< class TScalarType, unsigned int NDimensions >
bool GPUCompositeTransformBase< TScalarType, NDimensions >
::IsMatrixOffsetTransform( const std::size_t _index,
                           const bool _loadSource, std::string & _source ) const
{
  typedef GPUMatrixOffsetTransformBase<
      ScalarType, InputSpaceDimension, OutputSpaceDimension > MatrixOffsetTransformBaseType;
  const MatrixOffsetTransformBaseType *matrixoffset =
    dynamic_cast< const MatrixOffsetTransformBaseType * >( this->GetNthTransform( _index ).GetPointer() );
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
template< class TScalarType, unsigned int NDimensions >
bool GPUCompositeTransformBase< TScalarType, NDimensions >
::IsTranslationTransform( const std::size_t _index,
                          const bool _loadSource, std::string & _source ) const
{
  typedef GPUTranslationTransformBase<
      ScalarType, InputSpaceDimension > TranslationTransformBaseType;
  const TranslationTransformBaseType *translation =
    dynamic_cast< const TranslationTransformBaseType * >( this->GetNthTransform( _index ).GetPointer() );
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
template< class TScalarType, unsigned int NDimensions >
bool GPUCompositeTransformBase< TScalarType, NDimensions >
::IsBSplineTransform( const std::size_t _index,
                      const bool _loadSource, std::string & _source ) const
{
  if ( this->GetNthTransform( _index )->GetTransformCategory() == TransformBase::BSpline )
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
template< class TScalarType, unsigned int NDimensions >
void GPUCompositeTransformBase< TScalarType, NDimensions >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  CPUSuperclass::PrintSelf( os, indent );
}
} // end namespace itk

#endif /* __itkGPUCompositeTransformBase_hxx */
