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
#include <iomanip>

namespace itk
{
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder, class TParentImageFilter >
GPUBSplineTransform< TScalarType, NDimensions, VSplineOrder, TParentImageFilter >::GPUBSplineTransform()
{
  // Add GPUMatrixOffsetTransformBase header
  const std::string sourcePath0(GPUMatrixOffsetTransformBaseHeaderKernel::GetOpenCLSource());
  m_Sources.push_back(sourcePath0);

  // Add GPUBSplineTransform source
  const std::string sourcePath1(GPUBSplineTransformKernel::GetOpenCLSource());
  m_Sources.push_back(sourcePath1);

  m_SourcesLoaded = true; // we set it to true, sources are loaded from strings
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder, class TParentImageFilter >
bool GPUBSplineTransform< TScalarType, NDimensions, VSplineOrder, TParentImageFilter >
::GetSourceCode(std::string &_source) const
{
  if(!m_SourcesLoaded)
    return false;

  // Create the final source code
  std::ostringstream source;

  source << "//------------------------------------------------------------------------------\n";
  // Variable length array declaration not allowed in OpenCL, therefore we are using #define
  source << "#define GPUBSplineTransformOrder (" << this->SplineOrder << ")" << std::endl;

  // Calculate number of weights;
  const unsigned long numberOfWeights =
    static_cast< unsigned long >( vcl_pow( static_cast< double >( this->SplineOrder + 1 ),
    static_cast< double >( this->SpaceDimension ) ) );

  // Variable length array declaration not allowed in OpenCL, therefore we are using #define
  source << "#define GPUBSplineTransformNumberOfWeights (" << numberOfWeights << ")" << std::endl;

  // Add other sources
  for(unsigned int i=0; i<m_Sources.size(); i++)
  {
    source << m_Sources[i] << std::endl;
  }

  _source = source.str();
  return true;
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder, class TParentImageFilter >
void GPUBSplineTransform< TScalarType, NDimensions, VSplineOrder, TParentImageFilter >
::SetParameters(const ParametersType & parameters)
{
  Superclass::SetParameters( parameters );

  typedef typename Superclass::ImageType CPUCoefficientImage;
  typedef typename CPUCoefficientImage::PixelType CPUCoefficientsImagePixelType;
  typedef GPUImage<CPUCoefficientsImagePixelType, CPUCoefficientImage::ImageDimension> GPUCoefficientsImageType;

  for(unsigned int j = 0; j < SpaceDimension; j++)
  {
    GPUCoefficientsImageType *GPUCoefficientImage =
      dynamic_cast<GPUCoefficientsImageType *>(this->m_CoefficientImages[j].GetPointer());

    if(GPUCoefficientImage)
    {
      GPUCoefficientImage->GetGPUDataManager()->SetGPUBufferLock(false);

      GPUCoefficientImage->GetGPUDataManager()->SetCPUBufferPointer( GPUCoefficientImage->GetBufferPointer() );
      GPUCoefficientImage->GetGPUDataManager()->SetGPUDirtyFlag(true);
      GPUCoefficientImage->GetGPUDataManager()->UpdateGPUBuffer();

      GPUCoefficientImage->GetGPUDataManager()->SetGPUBufferLock(true);
      GPUCoefficientImage->GetGPUDataManager()->SetCPUBufferLock(true);
    }
  }
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder, class TParentImageFilter >
void GPUBSplineTransform< TScalarType, NDimensions, VSplineOrder, TParentImageFilter >
::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
}

} // namespace

#endif
