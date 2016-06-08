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
#ifndef __itkGPUTransformCopier_hxx
#define __itkGPUTransformCopier_hxx

#include "itkGPUTransformCopier.h"

// ITK CPU transforms
#include "itkAffineTransform.h"
#include "itkTranslationTransform.h"
#include "itkBSplineTransform.h"
#include "itkEuler2DTransform.h"
#include "itkEuler3DTransform.h"
#include "itkSimilarity2DTransform.h"
#include "itkSimilarity3DTransform.h"

// ITK GPU transforms
#include "itkGPUAffineTransform.h"
#include "itkGPUTranslationTransform.h"
#include "itkGPUBSplineTransform.h"
#include "itkGPUEuler2DTransform.h"
#include "itkGPUEuler3DTransform.h"
#include "itkGPUSimilarity2DTransform.h"
#include "itkGPUSimilarity3DTransform.h"

// GPU factory include
#include "itkGPUImageFactory.h"

namespace itk
{
//------------------------------------------------------------------------------
template< typename TTypeList, typename NDimensions, typename TTransform, typename TOutputTransformPrecisionType >
GPUTransformCopier< TTypeList, NDimensions, TTransform, TOutputTransformPrecisionType >
::GPUTransformCopier()
{
  this->m_InputTransform        = NULL;
  this->m_Output                = NULL;
  this->m_InternalTransformTime = 0;
  this->m_ExplicitMode          = true;
}


//------------------------------------------------------------------------------
template< typename TTypeList, typename NDimensions, typename TTransform, typename TOutputTransformPrecisionType >
void
GPUTransformCopier< TTypeList, NDimensions, TTransform, TOutputTransformPrecisionType >
::Update( void )
{
  if( !this->m_InputTransform )
  {
    itkExceptionMacro( << "Input Transform has not been connected" );
    return;
  }

  // Update only if the input AdvancedCombinationTransform has been modified
  const ModifiedTimeType t = this->m_InputTransform->GetMTime();

  if( t == this->m_InternalTransformTime )
  {
    return; // No need to update
  }
  else if( t > this->m_InternalTransformTime )
  {
    // Cache the timestamp
    this->m_InternalTransformTime = t;

    // Copy transform
    const bool copyResult = this->CopyTransform( this->m_InputTransform, this->m_Output );
    if( !copyResult || this->m_Output.IsNull() )
    {
      itkExceptionMacro( << "GPUTransformCopier was unable to copy transform from: " << this->m_InputTransform );
    }
  }
}


//------------------------------------------------------------------------------
template< typename TTypeList, typename NDimensions, typename TTransform, typename TOutputTransformPrecisionType >
bool
GPUTransformCopier< TTypeList, NDimensions, TTransform, TOutputTransformPrecisionType >
::CopyTransform(
  const CPUTransformConstPointer & fromTransform,
  GPUTransformPointer & toTransform )
{
  // Try Affine
  typedef AffineTransform< CPUScalarType, InputSpaceDimension >
    AffineTransformType;
  const typename AffineTransformType::ConstPointer affine
    = dynamic_cast< const AffineTransformType * >( fromTransform.GetPointer() );

  if( affine )
  {
    GPUTransformPointer affineTransform;
    if( this->m_ExplicitMode )
    {
      // Create GPU Affine transform in explicit mode
      typedef GPUAffineTransform< GPUScalarType, InputSpaceDimension >
        GPUAffineTransformType;
      affineTransform = GPUAffineTransformType::New();
    }
    else
    {
      // Create GPU Affine transform in implicit mode
      typedef AffineTransform< GPUScalarType, InputSpaceDimension >
        GPUAffineTransformType;
      affineTransform = GPUAffineTransformType::New();
    }
    this->CastCopyTransformParameters( fromTransform, affineTransform );
    toTransform = affineTransform;
    return true;
  }

  // Try Translation
  typedef TranslationTransform< CPUScalarType, InputSpaceDimension >
    TranslationTransformType;
  const typename TranslationTransformType::ConstPointer translation
    = dynamic_cast< const TranslationTransformType * >( fromTransform.GetPointer() );

  if( translation )
  {
    GPUTransformPointer translationTransform;
    if( this->m_ExplicitMode )
    {
      // Create GPU Translation transform in explicit mode
      typedef GPUTranslationTransform< GPUScalarType, InputSpaceDimension >
        GPUTranslationTransformType;
      translationTransform = GPUTranslationTransformType::New();
    }
    else
    {
      // Create GPU Translation transform in implicit mode
      typedef TranslationTransform< GPUScalarType, InputSpaceDimension >
        GPUTranslationTransformType;
      translationTransform = GPUTranslationTransformType::New();
    }
    this->CastCopyTransformParameters( fromTransform, translationTransform );
    toTransform = translationTransform;
    return true;
  }

  // For BSpline we have to check all possible spline orders
  const bool bsplineCopyResult = this->CopyBSplineTransform( fromTransform, toTransform );
  if( bsplineCopyResult )
  {
    return bsplineCopyResult;
  }

  // For Euler and Similarity transforms we have to use partial
  // template specialization logic.
  const unsigned int                                         InputDimension = InputSpaceDimension;
  const TransformSpaceDimensionToType< InputSpaceDimension > idim           = {};

  // Try Euler
  bool eulerCopyResult = false;
  switch( InputDimension )
  {
    case 2:
      eulerCopyResult = this->CopyEuler2DTransform( fromTransform, toTransform, idim );
      break;
    case 3:
      eulerCopyResult = this->CopyEuler3DTransform( fromTransform, toTransform, idim );
      break;
    default:
      break;
  }

  if( eulerCopyResult )
  {
    return eulerCopyResult;
  }

  // Try Similarity
  bool similarityCopyResult = false;
  switch( InputDimension )
  {
    case 2:
      similarityCopyResult = this->CopySimilarity2DTransform( fromTransform, toTransform, idim );
      break;
    case 3:
      similarityCopyResult = this->CopySimilarity3DTransform( fromTransform, toTransform, idim );
      break;
    default:
      break;
  }

  if( similarityCopyResult )
  {
    return similarityCopyResult;
  }

  return false;
}


//------------------------------------------------------------------------------
template< typename TTypeList, typename NDimensions, typename TTransform, typename TOutputTransformPrecisionType >
void
GPUTransformCopier< TTypeList, NDimensions, TTransform, TOutputTransformPrecisionType >
::CastCopyTransformParameters(
  const CPUTransformConstPointer & fromTransform,
  GPUTransformPointer & toTransform )
{
  const CPUFixedParametersType & fixedParametersFrom
    = fromTransform->GetFixedParameters();
  const CPUParametersType & parametersFrom
    = fromTransform->GetParameters();

  GPUFixedParametersType fixedParametersTo;
  GPUParametersType      parametersTo;

  this->CastCopyFixedParameters( fixedParametersFrom, fixedParametersTo );
  this->CastCopyParameters( parametersFrom, parametersTo );

  toTransform->SetFixedParameters( fixedParametersTo );
  toTransform->SetParameters( parametersTo );
}


//------------------------------------------------------------------------------
template< typename TTypeList, typename NDimensions, typename TTransform, typename TOutputTransformPrecisionType >
void
GPUTransformCopier< TTypeList, NDimensions, TTransform, TOutputTransformPrecisionType >
::CastCopyParameters( const CPUParametersType & from, GPUParametersType & to )
{
  if( from.GetSize() == 0 ) { return; }

  to.SetSize( from.GetSize() );
  for( SizeValueType i = 0; i < from.GetSize(); ++i )
  {
    to[ i ] = static_cast< GPUScalarType >( from[ i ] );
  }
}


//------------------------------------------------------------------------------
template< typename TTypeList, typename NDimensions, typename TTransform, typename TOutputTransformPrecisionType >
void
GPUTransformCopier< TTypeList, NDimensions, TTransform, TOutputTransformPrecisionType >
::CastCopyFixedParameters( const CPUFixedParametersType & from, GPUFixedParametersType & to )
{
  if( from.GetSize() == 0 ) { return; }
  to.SetSize( from.GetSize() );
  for( SizeValueType i = 0; i < from.GetSize(); ++i )
  {
    to[ i ] = static_cast< GPUScalarType >( from[ i ] );
  }
}


//------------------------------------------------------------------------------
template< typename TTypeList, typename NDimensions, typename TTransform, typename TOutputTransformPrecisionType >
bool
GPUTransformCopier< TTypeList, NDimensions, TTransform, TOutputTransformPrecisionType >
::CopyBSplineTransform(
  const CPUTransformConstPointer & fromTransform,
  GPUTransformPointer & toTransform )
{
  typedef BSplineTransform< CPUScalarType, InputSpaceDimension, 0 >
    BSplineOrder0TransformType;
  typedef BSplineTransform< CPUScalarType, InputSpaceDimension, 1 >
    BSplineOrder1TransformType;
  typedef BSplineTransform< CPUScalarType, InputSpaceDimension, 2 >
    BSplineOrder2TransformType;
  typedef BSplineTransform< CPUScalarType, InputSpaceDimension, 3 >
    BSplineOrder3TransformType;

  GPUTransformPointer bsplineTransform;

  // When creating the GPUBSplineTransform in explicit mode
  // We also have to register GPUImageFactory because
  // GPUBSplineTransform using m_Coefficients as ITK images
  // inside the implementation, therefore we define GPUImageFactory pointer
  typedef itk::GPUImageFactory2< TTypeList, NDimensions > GPUImageFactoryType;
  typedef typename GPUImageFactoryType::Pointer           GPUImageFactoryPointer;

  // Try BSpline Order 3 first
  const typename BSplineOrder3TransformType::ConstPointer bsplineOrder3
    = dynamic_cast< const BSplineOrder3TransformType * >( fromTransform.GetPointer() );

  if( bsplineOrder3 )
  {
    if( this->m_ExplicitMode )
    {
      // Register image factory
      GPUImageFactoryPointer imageFactory = GPUImageFactoryType::New();
      itk::ObjectFactoryBase::RegisterFactory( imageFactory );

      // Create GPU BSpline transform in explicit mode
      typedef GPUBSplineTransform< GPUScalarType, InputSpaceDimension, 3 >
        GPUBSplineTransformType;
      bsplineTransform = GPUBSplineTransformType::New();

      // UnRegister image factory
      itk::ObjectFactoryBase::UnRegisterFactory( imageFactory );
    }
    else
    {
      // Create GPU BSpline transform in implicit mode
      typedef BSplineTransform< GPUScalarType, InputSpaceDimension, 3 >
        GPUBSplineTransformType;
      bsplineTransform = GPUBSplineTransformType::New();
    }
    this->CastCopyTransformParameters( fromTransform, bsplineTransform );
    toTransform = bsplineTransform;
    return true;
  }
  else
  {
    // Try BSpline Order 0
    const typename BSplineOrder0TransformType::ConstPointer bsplineOrder0
      = dynamic_cast< const BSplineOrder0TransformType * >( fromTransform.GetPointer() );

    if( bsplineOrder0 )
    {
      if( this->m_ExplicitMode )
      {
        // Register image factory
        GPUImageFactoryPointer imageFactory = GPUImageFactoryType::New();
        itk::ObjectFactoryBase::RegisterFactory( imageFactory );

        // Create GPU BSpline transform in explicit mode
        typedef GPUBSplineTransform< GPUScalarType, InputSpaceDimension, 0 >
          GPUBSplineTransformType;
        bsplineTransform = GPUBSplineTransformType::New();

        // UnRegister image factory
        itk::ObjectFactoryBase::UnRegisterFactory( imageFactory );
      }
      else
      {
        // Create GPU BSpline transform in implicit mode
        typedef BSplineTransform< GPUScalarType, InputSpaceDimension, 0 >
          GPUBSplineTransformType;
        bsplineTransform = GPUBSplineTransformType::New();
      }
      this->CastCopyTransformParameters( fromTransform, bsplineTransform );
      toTransform = bsplineTransform;
      return true;
    }
    else
    {
      // Try BSpline Order 1
      const typename BSplineOrder1TransformType::ConstPointer bsplineOrder1
        = dynamic_cast< const BSplineOrder1TransformType * >( fromTransform.GetPointer() );

      if( bsplineOrder1 )
      {
        if( this->m_ExplicitMode )
        {
          // Register image factory
          GPUImageFactoryPointer imageFactory = GPUImageFactoryType::New();
          itk::ObjectFactoryBase::RegisterFactory( imageFactory );

          // Create GPU BSpline transform in explicit mode
          typedef GPUBSplineTransform< GPUScalarType, InputSpaceDimension, 1 >
            GPUBSplineTransformType;
          bsplineTransform = GPUBSplineTransformType::New();

          // UnRegister image factory
          itk::ObjectFactoryBase::UnRegisterFactory( imageFactory );
        }
        else
        {
          // Create GPU BSpline transform in implicit mode
          typedef BSplineTransform< GPUScalarType, InputSpaceDimension, 1 >
            GPUBSplineTransformType;
          bsplineTransform = GPUBSplineTransformType::New();
        }
        this->CastCopyTransformParameters( fromTransform, bsplineTransform );
        toTransform = bsplineTransform;
        return true;
      }
      else
      {
        // Try BSpline Order 2
        const typename BSplineOrder2TransformType::ConstPointer bsplineOrder2
          = dynamic_cast< const BSplineOrder2TransformType * >( fromTransform.GetPointer() );

        if( bsplineOrder2 )
        {
          if( this->m_ExplicitMode )
          {
            // Register image factory
            GPUImageFactoryPointer imageFactory = GPUImageFactoryType::New();
            itk::ObjectFactoryBase::RegisterFactory( imageFactory );

            // Create GPU BSpline transform in explicit mode
            typedef GPUBSplineTransform< GPUScalarType, InputSpaceDimension, 2 >
              GPUBSplineTransformType;
            bsplineTransform = GPUBSplineTransformType::New();

            // UnRegister image factory
            itk::ObjectFactoryBase::UnRegisterFactory( imageFactory );
          }
          else
          {
            // Create GPU BSpline transform in implicit mode
            typedef BSplineTransform< GPUScalarType, InputSpaceDimension, 2 >
              GPUBSplineTransformType;
            bsplineTransform = GPUBSplineTransformType::New();
          }
          this->CastCopyTransformParameters( fromTransform, bsplineTransform );
          toTransform = bsplineTransform;
          return true;
        }
      }
    }
  }

  return false;
}


//------------------------------------------------------------------------------
template< typename TTypeList, typename NDimensions, typename TTransform, typename TOutputTransformPrecisionType >
bool
GPUTransformCopier< TTypeList, NDimensions, TTransform, TOutputTransformPrecisionType >
::CopyEuler2DTransform(
  const CPUTransformConstPointer & fromTransform,
  GPUTransformPointer & toTransform,
  TransformSpaceDimensionToType< 2 > )
{
  typedef Euler2DTransform< CPUScalarType > CPUEulerTransformType;
  const typename CPUEulerTransformType::ConstPointer euler
    = dynamic_cast< const CPUEulerTransformType * >( fromTransform.GetPointer() );

  if( euler )
  {
    GPUTransformPointer eulerTransform;
    if( this->m_ExplicitMode )
    {
      // Create GPU Euler transform in explicit mode
      typedef GPUEuler2DTransform< GPUScalarType >
        GPUEulerTransformType;
      eulerTransform = GPUEulerTransformType::New();
    }
    else
    {
      // Create GPU Euler transform in implicit mode
      typedef Euler2DTransform< GPUScalarType >
        GPUEulerTransformType;
      eulerTransform = GPUEulerTransformType::New();
    }
    this->CastCopyTransformParameters( fromTransform, eulerTransform );
    toTransform = eulerTransform;
    return true;
  }

  return false;
}


//------------------------------------------------------------------------------
template< typename TTypeList, typename NDimensions, typename TTransform, typename TOutputTransformPrecisionType >
bool
GPUTransformCopier< TTypeList, NDimensions, TTransform, TOutputTransformPrecisionType >
::CopyEuler3DTransform(
  const CPUTransformConstPointer & fromTransform,
  GPUTransformPointer & toTransform,
  TransformSpaceDimensionToType< 3 > )
{
  typedef Euler3DTransform< CPUScalarType > CPUEulerTransformType;
  const typename CPUEulerTransformType::ConstPointer euler
    = dynamic_cast< const CPUEulerTransformType * >( fromTransform.GetPointer() );

  if( euler )
  {
    GPUTransformPointer eulerTransform;
    if( this->m_ExplicitMode )
    {
      // Create GPU Euler transform in explicit mode
      typedef GPUEuler3DTransform< GPUScalarType >
        GPUEulerTransformType;
      eulerTransform = GPUEulerTransformType::New();
    }
    else
    {
      // Create GPU Euler transform in implicit mode
      typedef Euler3DTransform< GPUScalarType >
        GPUEulerTransformType;
      eulerTransform = GPUEulerTransformType::New();
    }
    this->CastCopyTransformParameters( fromTransform, eulerTransform );
    toTransform = eulerTransform;
    return true;
  }

  return false;
}


//------------------------------------------------------------------------------
template< typename TTypeList, typename NDimensions, typename TTransform, typename TOutputTransformPrecisionType >
bool
GPUTransformCopier< TTypeList, NDimensions, TTransform, TOutputTransformPrecisionType >
::CopySimilarity2DTransform(
  const CPUTransformConstPointer & fromTransform,
  GPUTransformPointer & toTransform,
  TransformSpaceDimensionToType< 2 > )
{
  typedef Similarity2DTransform< CPUScalarType > CPUSimilarityTransformType;
  const typename CPUSimilarityTransformType::ConstPointer similarity
    = dynamic_cast< const CPUSimilarityTransformType * >( fromTransform.GetPointer() );

  if( similarity )
  {
    GPUTransformPointer similarityTransform;
    if( this->m_ExplicitMode )
    {
      // Create GPU Similarity transform in explicit mode
      typedef GPUSimilarity2DTransform< GPUScalarType >
        GPUSimilarityTransformType;
      similarityTransform = GPUSimilarityTransformType::New();
    }
    else
    {
      // Create GPU Similarity transform in implicit mode
      typedef Similarity2DTransform< GPUScalarType >
        GPUSimilarityTransformType;
      similarityTransform = GPUSimilarityTransformType::New();
    }
    this->CastCopyTransformParameters( fromTransform, similarityTransform );
    toTransform = similarityTransform;
    return true;
  }

  return false;
}


//------------------------------------------------------------------------------
template< typename TTypeList, typename NDimensions, typename TTransform, typename TOutputTransformPrecisionType >
bool
GPUTransformCopier< TTypeList, NDimensions, TTransform, TOutputTransformPrecisionType >
::CopySimilarity3DTransform(
  const CPUTransformConstPointer & fromTransform,
  GPUTransformPointer & toTransform,
  TransformSpaceDimensionToType< 3 > )
{
  typedef Similarity3DTransform< CPUScalarType > CPUSimilarityTransformType;
  const typename CPUSimilarityTransformType::ConstPointer similarity
    = dynamic_cast< const CPUSimilarityTransformType * >( fromTransform.GetPointer() );

  if( similarity )
  {
    GPUTransformPointer similarityTransform;
    if( this->m_ExplicitMode )
    {
      // Create GPU Similarity transform in explicit mode
      typedef GPUSimilarity3DTransform< GPUScalarType >
        GPUSimilarityTransformType;
      similarityTransform = GPUSimilarityTransformType::New();
    }
    else
    {
      // Create GPU Similarity transform in implicit mode
      typedef Similarity3DTransform< GPUScalarType >
        GPUSimilarityTransformType;
      similarityTransform = GPUSimilarityTransformType::New();
    }
    this->CastCopyTransformParameters( fromTransform, similarityTransform );
    toTransform = similarityTransform;
    return true;
  }

  return false;
}


//------------------------------------------------------------------------------
template< typename TTypeList, typename NDimensions, typename TTransform, typename TOutputTransformPrecisionType >
void
GPUTransformCopier< TTypeList, NDimensions, TTransform, TOutputTransformPrecisionType >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "Input Transform: " << this->m_InputTransform << std::endl;
  os << indent << "Output Transform: " << this->m_Output << std::endl;
  os << indent << "Internal Transform Time: " << this->m_InternalTransformTime << std::endl;
  os << indent << "Explicit Mode: " << this->m_ExplicitMode << std::endl;
}


} // end namespace itk

#endif
