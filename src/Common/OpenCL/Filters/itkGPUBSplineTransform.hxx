/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef __itkGPUBSplineTransform_hxx
#define __itkGPUBSplineTransform_hxx

#include "itkGPUBSplineTransform.h"
#include "itkGPUKernelManagerHelperFunctions.h"
#include <iomanip>

namespace itk
{
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder, class TParentImageFilter >
GPUBSplineTransform< TScalarType, NDimensions, VSplineOrder, TParentImageFilter >::GPUBSplineTransform()
{
  // Load GPUMatrixOffsetTransformBase header
  std::string sname = "GPUMatrixOffsetTransformBase header";
  const std::string sourcePath0(oclhGPUMatrixOffsetTransformBase);
  m_SourcesLoaded = LoadProgramFromFile(sourcePath0, m_Sources, sname, true);
  if(!m_SourcesLoaded)
  {
    itkGenericExceptionMacro( << sname << " has not been loaded from: " << sourcePath0 );
  }

  // Load GPUBSplineTransform source
  sname = "GPUBSplineTransform source";
  const std::string sourcePath1(oclGPUBSplineTransform);
  m_SourcesLoaded = m_SourcesLoaded && LoadProgramFromFile(sourcePath1, m_Sources, sname, true);
  if(!m_SourcesLoaded)
  {
    itkGenericExceptionMacro( << sname << " has not been loaded from: " << sourcePath1 );
  }
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
  typedef itk::GPUImage<CPUCoefficientsImagePixelType, CPUCoefficientImage::ImageDimension> GPUCoefficientsImageType;

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
