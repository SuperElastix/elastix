/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
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
//------------------------------------------------------------------------------
template< typename TScalarType, unsigned int NDimensions >
GPUDataManager::Pointer
GPUCompositeTransformBase< TScalarType, NDimensions >
::GetParametersDataManager( const std::size_t index ) const
{
  GPUDataManager::Pointer parameters;

  if( this->GetNumberOfTransforms() == 0 )
  {
    return parameters;
  }
  else
  {
    const GPUTransformBase * transformBase
      = dynamic_cast< const GPUTransformBase * >( this->GetNthTransform( index ).GetPointer() );

    if( transformBase )
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
template< typename TScalarType, unsigned int NDimensions >
bool
GPUCompositeTransformBase< TScalarType, NDimensions >
::GetSourceCode( std::string & source ) const
{
  // Create the final source code
  std::ostringstream sources;
  std::string        source_i;

  bool identityLoaded    = false;
  bool affineLoaded      = false;
  bool translationLoaded = false;
  bool bsplineLoaded     = false;

  // Add sources based on Transform type
  for( std::size_t i = 0; i < this->GetNumberOfTransforms(); i++ )
  {
    if( this->IsIdentityTransform( i, true, source_i ) && !identityLoaded )
    {
      sources << source_i << std::endl;
      identityLoaded = true;
    }

    if( this->IsMatrixOffsetTransform( i, true, source_i ) && !affineLoaded )
    {
      sources << source_i << std::endl;
      affineLoaded = true;
    }

    if( this->IsTranslationTransform( i, true, source_i ) && !translationLoaded )
    {
      sources << source_i << std::endl;
      translationLoaded = true;
    }

    if( this->IsBSplineTransform( i, true, source_i ) && !bsplineLoaded )
    {
      sources << source_i << std::endl;
      bsplineLoaded = true;
    }
  }

  // Get final sources
  source = sources.str();
  return true;
}


//------------------------------------------------------------------------------
template< typename TScalarType, unsigned int NDimensions >
bool
GPUCompositeTransformBase< TScalarType, NDimensions >
::HasIdentityTransform( void ) const
{
  for( std::size_t i = 0; i < this->GetNumberOfTransforms(); i++ )
  {
    if( this->IsIdentityTransform( i ) )
    {
      return true;
    }
  }

  return false;
}


//------------------------------------------------------------------------------
template< typename TScalarType, unsigned int NDimensions >
bool
GPUCompositeTransformBase< TScalarType, NDimensions >
::HasMatrixOffsetTransform( void ) const
{
  for( std::size_t i = 0; i < this->GetNumberOfTransforms(); i++ )
  {
    if( this->IsMatrixOffsetTransform( i ) )
    {
      return true;
    }
  }

  return false;
}


//------------------------------------------------------------------------------
template< typename TScalarType, unsigned int NDimensions >
bool
GPUCompositeTransformBase< TScalarType, NDimensions >
::HasTranslationTransform( void ) const
{
  for( std::size_t i = 0; i < this->GetNumberOfTransforms(); i++ )
  {
    if( this->IsTranslationTransform( i ) )
    {
      return true;
    }
  }

  return false;
}


//------------------------------------------------------------------------------
template< typename TScalarType, unsigned int NDimensions >
bool
GPUCompositeTransformBase< TScalarType, NDimensions >
::HasBSplineTransform( void ) const
{
  for( std::size_t i = 0; i < this->GetNumberOfTransforms(); i++ )
  {
    if( this->IsBSplineTransform( i ) )
    {
      return true;
    }
  }

  return false;
}


//------------------------------------------------------------------------------
template< typename TScalarType, unsigned int NDimensions >
bool
GPUCompositeTransformBase< TScalarType, NDimensions >
::IsIdentityTransform( const std::size_t index ) const
{
  // First quick check if Linear
  if( this->GetNthTransform( index )->GetTransformCategory()
    != TransformBaseTemplate< TScalarType >::Linear )
  {
    return false;
  }

  // Perform specific check
  typedef GPUIdentityTransform< ScalarType, NDimensions > IdentityTransformType;
  const IdentityTransformType * identity
    = dynamic_cast< const IdentityTransformType * >( this->GetNthTransform( index ).GetPointer() );
  if( !identity ) { return false; }

  return true;
}


//------------------------------------------------------------------------------
template< typename TScalarType, unsigned int NDimensions >
bool
GPUCompositeTransformBase< TScalarType, NDimensions >
::IsMatrixOffsetTransform( const std::size_t index ) const
{
  // First quick check if Linear
  if( this->GetNthTransform( index )->GetTransformCategory()
    != TransformBaseTemplate< TScalarType >::Linear )
  {
    return false;
  }

  // Perform specific check
  typedef GPUMatrixOffsetTransformBase<
    ScalarType, InputSpaceDimension, OutputSpaceDimension > MatrixOffsetTransformBaseType;
  const MatrixOffsetTransformBaseType * matrixoffset
    = dynamic_cast< const MatrixOffsetTransformBaseType * >( this->GetNthTransform( index ).GetPointer() );
  if( !matrixoffset ) { return false; }

  return true;
}


//------------------------------------------------------------------------------
template< typename TScalarType, unsigned int NDimensions >
bool
GPUCompositeTransformBase< TScalarType, NDimensions >
::IsTranslationTransform( const std::size_t index ) const
{
  // First quick check if Linear
  if( this->GetNthTransform( index )->GetTransformCategory()
    != TransformBaseTemplate< TScalarType >::Linear )
  {
    return false;
  }

  // Perform specific check
  typedef GPUTranslationTransformBase<
    ScalarType, InputSpaceDimension > TranslationTransformBaseType;
  const TranslationTransformBaseType * translation
    = dynamic_cast< const TranslationTransformBaseType * >( this->GetNthTransform( index ).GetPointer() );
  if( !translation ) { return false; }

  return true;
}


//------------------------------------------------------------------------------
template< typename TScalarType, unsigned int NDimensions >
bool
GPUCompositeTransformBase< TScalarType, NDimensions >
::IsBSplineTransform( const std::size_t index ) const
{
  if( this->GetNthTransform( index )->GetTransformCategory()
    == TransformBaseTemplate< TScalarType >::BSpline )
  {
    return true;
  }

  return false;
}


//------------------------------------------------------------------------------
template< typename TScalarType, unsigned int NDimensions >
bool
GPUCompositeTransformBase< TScalarType, NDimensions >
::IsIdentityTransform( const std::size_t index,
  const bool loadSource, std::string & source ) const
{
  typedef GPUIdentityTransform< ScalarType, NDimensions > IdentityTransformType;
  const IdentityTransformType * identity
    = dynamic_cast< const IdentityTransformType * >( this->GetNthTransform( index ).GetPointer() );

  if( !identity ) { return false; }

  if( loadSource )
  {
    const GPUTransformBase * transformBase
      = dynamic_cast< const GPUTransformBase * >( this->GetNthTransform( index ).GetPointer() );
    transformBase->GetSourceCode( source );
  }

  return true;
}


//------------------------------------------------------------------------------
template< typename TScalarType, unsigned int NDimensions >
bool
GPUCompositeTransformBase< TScalarType, NDimensions >
::IsMatrixOffsetTransform( const std::size_t index,
  const bool loadSource, std::string & source ) const
{
  typedef GPUMatrixOffsetTransformBase<
    ScalarType, InputSpaceDimension, OutputSpaceDimension > MatrixOffsetTransformBaseType;
  const MatrixOffsetTransformBaseType * matrixoffset
    = dynamic_cast< const MatrixOffsetTransformBaseType * >( this->GetNthTransform( index ).GetPointer() );

  if( !matrixoffset ) { return false; }

  if( loadSource )
  {
    const GPUTransformBase * transformBase
      = dynamic_cast< const GPUTransformBase * >( this->GetNthTransform( index ).GetPointer() );
    transformBase->GetSourceCode( source );
  }

  return true;
}


//------------------------------------------------------------------------------
template< typename TScalarType, unsigned int NDimensions >
bool
GPUCompositeTransformBase< TScalarType, NDimensions >
::IsTranslationTransform( const std::size_t index,
  const bool loadSource, std::string & source ) const
{
  typedef GPUTranslationTransformBase<
    ScalarType, InputSpaceDimension > TranslationTransformBaseType;
  const TranslationTransformBaseType * translation
    = dynamic_cast< const TranslationTransformBaseType * >(
    this->GetNthTransform( index ).GetPointer() );

  if( !translation ) { return false; }

  if( loadSource )
  {
    const GPUTransformBase * transformBase
      = dynamic_cast< const GPUTransformBase * >( this->GetNthTransform( index ).GetPointer() );
    transformBase->GetSourceCode( source );
  }

  return true;
}


//------------------------------------------------------------------------------
template< typename TScalarType, unsigned int NDimensions >
bool
GPUCompositeTransformBase< TScalarType, NDimensions >
::IsBSplineTransform(
  const std::size_t _index,
  const bool loadSource, std::string & source ) const
{
  if( this->GetNthTransform( _index )->GetTransformCategory()
    == TransformBaseTemplate< TScalarType >::BSpline )
  {
    if( loadSource )
    {
      const GPUTransformBase * transformBase
        = dynamic_cast< const GPUTransformBase * >( this->GetNthTransform( _index ).GetPointer() );
      transformBase->GetSourceCode( source );
    }
    return true;
  }

  return false;
}


} // end namespace itk

#endif /* __itkGPUCompositeTransformBase_hxx */
