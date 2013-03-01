/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUBSplineTransform_hxx
#define __itkGPUBSplineTransform_hxx

#include "itkGPUBSplineTransform.h"

#include "itkGPUMatrixOffsetTransformBase.h"
#include "itkGPUImage.h"
#include "itkGPUExplicitSynchronization.h"

#include "itkCastImageFilter.h"

//------------------------------------------------------------------------------
namespace itk
{
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder, class TParentImageFilter >
void GPUBSplineTransform< TScalarType, NDimensions, VSplineOrder, TParentImageFilter >
::SetCoefficientImages( const CoefficientImageArray & images )
{
  CPUSuperclass::SetCoefficientImages( images );
  CopyCoefficientImagesToGPU();
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder, class TParentImageFilter >
void GPUBSplineTransform< TScalarType, NDimensions, VSplineOrder, TParentImageFilter >
::SetParameters( const ParametersType & parameters )
{
  CPUSuperclass::SetParameters( parameters );
  CopyCoefficientImagesToGPU();
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder, class TParentImageFilter >
void GPUBSplineTransform< TScalarType, NDimensions, VSplineOrder, TParentImageFilter >
::CopyCoefficientImagesToGPU()
{
  typedef typename CPUSuperclass::ImageType                                              CPUCoefficientImage;
  typedef typename CPUCoefficientImage::PixelType                                        CPUCoefficientsImagePixelType;
  typedef GPUImage< CPUCoefficientsImagePixelType, CPUCoefficientImage::ImageDimension > GPUCoefficientsImageType;

  for ( unsigned int j = 0; j < SpaceDimension; j++ )
  {
    GPUCoefficientsImageType *GPUCoefficientImage =
      dynamic_cast< GPUCoefficientsImageType * >( this->m_CoefficientImages[j].GetPointer() );

    if ( GPUCoefficientImage )
    {
      GPUCoefficientImage->GetGPUDataManager()->SetGPUBufferLock( false );

      GPUCoefficientImage->GetGPUDataManager()->SetCPUBufferPointer( GPUCoefficientImage->GetBufferPointer() );
      GPUCoefficientImage->GetGPUDataManager()->SetGPUDirtyFlag( true );
      GPUCoefficientImage->GetGPUDataManager()->UpdateGPUBuffer();

      GPUCoefficientImage->GetGPUDataManager()->SetGPUBufferLock( true );
      GPUCoefficientImage->GetGPUDataManager()->SetCPUBufferLock( true );
    }
  }

  // CPU Typedefs
  typedef BSplineTransform< TScalarType, NDimensions, VSplineOrder > BSplineTransformType;
  typedef typename BSplineTransformType::ImageType                   TransformCoefficientImageType;
  typedef typename BSplineTransformType::ImagePointer                TransformCoefficientImagePointer;
  typedef typename BSplineTransformType::CoefficientImageArray       CoefficientImageArray;

  // GPU Typedefs
  typedef typename GPUSuperclass::GPUCoefficientImageType    GPUTransformCoefficientImageType;
  typedef typename GPUSuperclass::GPUCoefficientImagePointer GPUTransformCoefficientImagePointer;
  typedef typename GPUSuperclass::GPUDataManagerPointer      GPUDataManagerPointer;

  const CoefficientImageArray coefficientImageArray = this->GetCoefficientImages();

  // Typedef for caster
  typedef CastImageFilter< TransformCoefficientImageType, GPUTransformCoefficientImageType > CasterType;

  for ( unsigned int i = 0; i < coefficientImageArray.Size(); i++ )
  {
    TransformCoefficientImagePointer coefficients = coefficientImageArray[i];

    GPUTransformCoefficientImagePointer GPUCoefficients = GPUTransformCoefficientImageType::New();
    GPUCoefficients->CopyInformation( coefficients );
    GPUCoefficients->SetRegions( coefficients->GetBufferedRegion() );
    GPUCoefficients->Allocate();

    // Create caster
    typename CasterType::Pointer caster = CasterType::New();
    caster->SetInput( coefficients );
    caster->GraftOutput( GPUCoefficients );
    caster->Update();

    GPUExplicitSync< CasterType, GPUTransformCoefficientImageType >( caster, false );

    this->m_GPUBSplineTransformCoefficientImages[i] = GPUCoefficients;

    GPUDataManagerPointer GPUCoefficientsBase = GPUDataManager::New();
    this->m_GPUBSplineTransformCoefficientImagesBase[i] = GPUCoefficientsBase;
  }
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder, class TParentImageFilter >
void GPUBSplineTransform< TScalarType, NDimensions, VSplineOrder, TParentImageFilter >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  CPUSuperclass::PrintSelf( os, indent );
}
} // namespace

#endif /* __itkGPUBSplineTransform_hxx */
