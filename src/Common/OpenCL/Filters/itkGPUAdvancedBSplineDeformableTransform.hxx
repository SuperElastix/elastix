/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUAdvancedBSplineDeformableTransform_hxx
#define __itkGPUAdvancedBSplineDeformableTransform_hxx

#include "itkGPUAdvancedBSplineDeformableTransform.h"

#include "itkGPUMatrixOffsetTransformBase.h"
#include "itkGPUImage.h"
#include "itkGPUExplicitSynchronization.h"

#include "itkCastImageFilter.h"

#include <iomanip>

//------------------------------------------------------------------------------
namespace itk
{
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder, class TParentImageFilter >
GPUAdvancedBSplineDeformableTransform< TScalarType, NDimensions, VSplineOrder, TParentImageFilter >
::GPUAdvancedBSplineDeformableTransform()
{
  // Add GPUBSplineTransform source
  const std::string sourcePath(
    GPUBSplineTransformKernel::GetOpenCLSource() );

  m_Sources.push_back( sourcePath );

  m_SourcesLoaded = true; // we set it to true, sources are loaded from strings
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder, class TParentImageFilter >
bool GPUAdvancedBSplineDeformableTransform< TScalarType, NDimensions, VSplineOrder, TParentImageFilter >
::GetSourceCode( std::string & _source ) const
{
  if ( !m_SourcesLoaded )
  {
    return false;
  }

  // Create the final source code
  std::ostringstream source;

  source << "//------------------------------------------------------------------------------\n";
  // Variable length array declaration not allowed in OpenCL, therefore we are
  // using #define
  source << "#define GPUBSplineTransformOrder (" << this->SplineOrder << ")" << std::endl;

  // Calculate number of weights;
  const unsigned long numberOfWeights =
    static_cast< unsigned long >( vcl_pow( static_cast< double >( this->SplineOrder + 1 ),
                                           static_cast< double >( this->SpaceDimension ) ) );

  // Variable length array declaration not allowed in OpenCL, therefore we are
  // using #define
  source << "#define GPUBSplineTransformNumberOfWeights (" << numberOfWeights << ")" << std::endl;

  // Add other sources
  for ( std::size_t i = 0; i < m_Sources.size(); i++ )
  {
    source << m_Sources[i] << std::endl;
  }

  _source = source.str();
  return true;
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder, class TParentImageFilter >
void GPUAdvancedBSplineDeformableTransform< TScalarType, NDimensions, VSplineOrder, TParentImageFilter >
::SetCoefficientImages( ImagePointer images[] )
{
  CPUSuperclass::SetCoefficientImages( images );
  CopyCoefficientImagesToGPU();
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder, class TParentImageFilter >
void GPUAdvancedBSplineDeformableTransform< TScalarType, NDimensions, VSplineOrder, TParentImageFilter >
::SetParameters( const ParametersType & parameters )
{
  CPUSuperclass::SetParameters( parameters );
  CopyCoefficientImagesToGPU();
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder, class TParentImageFilter >
void GPUAdvancedBSplineDeformableTransform< TScalarType, NDimensions, VSplineOrder, TParentImageFilter >
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
  typedef AdvancedBSplineDeformableTransform< TScalarType, NDimensions,
                                              VSplineOrder >            AdvancedBSplineDeformableTransformType;
  typedef typename AdvancedBSplineDeformableTransformType::ImageType    TransformCoefficientImageType;
  typedef typename AdvancedBSplineDeformableTransformType::ImagePointer TransformCoefficientImagePointer;

  // GPU Typedefs
  typedef typename GPUSuperclass::GPUCoefficientImageType    GPUTransformCoefficientImageType;
  typedef typename GPUSuperclass::GPUCoefficientImagePointer GPUTransformCoefficientImagePointer;
  typedef typename GPUSuperclass::GPUDataManagerPointer      GPUDataManagerPointer;

  const ImagePointer *coefficientImageArray = this->GetCoefficientImages();

  // Typedef for caster
  typedef CastImageFilter< TransformCoefficientImageType, GPUTransformCoefficientImageType > CasterType;

  for ( unsigned int i = 0; i < SpaceDimension; i++ )
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
void GPUAdvancedBSplineDeformableTransform< TScalarType, NDimensions, VSplineOrder, TParentImageFilter >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  CPUSuperclass::PrintSelf( os, indent );
}
} // namespace

#endif /* __itkGPUAdvancedBSplineDeformableTransform_hxx */
