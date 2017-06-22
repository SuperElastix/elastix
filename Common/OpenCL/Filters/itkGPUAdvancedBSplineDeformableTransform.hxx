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
#ifndef __itkGPUAdvancedBSplineDeformableTransform_hxx
#define __itkGPUAdvancedBSplineDeformableTransform_hxx

#include "itkGPUAdvancedBSplineDeformableTransform.h"

#include "itkGPUMatrixOffsetTransformBase.h"
#include "itkGPUImage.h"

namespace itk
{
/**
 * ***************** Constructor ***********************
 */

template< typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder, typename TParentTransform >
GPUAdvancedBSplineDeformableTransform< TScalarType, NDimensions, VSplineOrder, TParentTransform >
::GPUAdvancedBSplineDeformableTransform()
{
  GPUSuperclass::SetSplineOrder( CPUSuperclass::SplineOrder );

  typedef typename CPUSuperclass::ImageType       CPUCoefficientImage;
  typedef typename CPUCoefficientImage::PixelType CPUCoefficientsImagePixelType;

  typedef GPUImage< CPUCoefficientsImagePixelType,
    CPUCoefficientImage::ImageDimension > GPUCoefficientsImageType;
  typedef typename GPUCoefficientsImageType::Pointer GPUCoefficientsImagePointer;

  for( unsigned int i = 0; i < SpaceDimension; ++i )
  {
    GPUCoefficientsImagePointer gpuCoefficientImage
      = dynamic_cast< GPUCoefficientsImageType * >( this->m_CoefficientImages[ i ].GetPointer() );

    if( gpuCoefficientImage )
    {
      gpuCoefficientImage->GetGPUDataManager()->SetGPUBufferLock( true );
      gpuCoefficientImage->GetGPUDataManager()->SetCPUBufferLock( true );
    }
  }
} // end constructor


/**
 * ***************** SetCoefficientImages ***********************
 */

template< typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder, typename TParentTransform >
void
GPUAdvancedBSplineDeformableTransform< TScalarType, NDimensions, VSplineOrder, TParentTransform >
::SetCoefficientImages( ImagePointer images[] )
{
  CPUSuperclass::SetCoefficientImages( images );
  this->CopyCoefficientImagesToGPU();
} // end SetCoefficientImages()


/**
 * ***************** SetParameters ***********************
 */

template< typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder, typename TParentTransform >
void
GPUAdvancedBSplineDeformableTransform< TScalarType, NDimensions, VSplineOrder, TParentTransform >
::SetParameters( const ParametersType & parameters )
{
  CPUSuperclass::SetParameters( parameters );
  this->CopyCoefficientImagesToGPU();
} // end SetParameters()


/**
 * ***************** CopyCoefficientImagesToGPU ***********************
 */

template< typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder, typename TParentTransform >
void
GPUAdvancedBSplineDeformableTransform< TScalarType, NDimensions, VSplineOrder, TParentTransform >
::CopyCoefficientImagesToGPU( void )
{
  typedef typename CPUSuperclass::ImageType             CPUCoefficientImage;
  typedef typename CPUCoefficientImage::PixelType       CPUCoefficientsImagePixelType;
  typedef typename GPUSuperclass::GPUDataManagerPointer GPUDataManagerPointer;

  typedef GPUImage< CPUCoefficientsImagePixelType,
    CPUCoefficientImage::ImageDimension > GPUCoefficientsImageType;
  typedef typename GPUCoefficientsImageType::Pointer GPUCoefficientsImagePointer;

  for( unsigned int i = 0; i < SpaceDimension; ++i )
  {
    GPUCoefficientsImagePointer gpuCoefficientImage
      = dynamic_cast< GPUCoefficientsImageType * >( this->m_CoefficientImages[ i ].GetPointer() );

    if( gpuCoefficientImage )
    {
      gpuCoefficientImage->GetGPUDataManager()->SetGPUBufferLock( false );
      gpuCoefficientImage->AllocateGPU();
      gpuCoefficientImage->GetGPUDataManager()->SetGPUDirtyFlag( true );
      gpuCoefficientImage->GetGPUDataManager()->UpdateGPUBuffer();
      gpuCoefficientImage->GetGPUDataManager()->SetGPUBufferLock( true );
    }

    this->m_GPUBSplineTransformCoefficientImages[ i ] = gpuCoefficientImage;

    GPUDataManagerPointer gpuCoefficientsBase = GPUDataManager::New();
    this->m_GPUBSplineTransformCoefficientImagesBase[ i ] = gpuCoefficientsBase;
  }
} // end CopyCoefficientImagesToGPU()


/**
 * ***************** PrintSelf ***********************
 */

template< typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder, typename TParentTransform >
void
GPUAdvancedBSplineDeformableTransform< TScalarType, NDimensions, VSplineOrder, TParentTransform >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  CPUSuperclass::PrintSelf( os, indent );
} // end PrintSelf()


} // end namespace itk

#endif /* __itkGPUAdvancedBSplineDeformableTransform_hxx */
