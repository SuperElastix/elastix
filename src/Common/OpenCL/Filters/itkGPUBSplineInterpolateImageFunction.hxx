/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUBSplineInterpolateImageFunction_hxx
#define __itkGPUBSplineInterpolateImageFunction_hxx

#include "itkGPUBSplineInterpolateImageFunction.h"
#include "itkGPUImageFunction.h"
#include <iomanip>

namespace itk
{
template< class TInputImage, class TCoordRep, class TCoefficientType >
GPUBSplineInterpolateImageFunction< TInputImage, TCoordRep, TCoefficientType >
::GPUBSplineInterpolateImageFunction()
{
  // Create GPU coefficients image
  this->m_GPUCoefficients = GPUCoefficientImageType::New();
  this->m_GPUCoefficientsImageBase = GPUDataManager::New();

  // Add GPUImageFunction implementation
  const std::string sourcePath0(GPUImageFunctionKernel::GetOpenCLSource());
  m_Sources.push_back(sourcePath0);

  // Add GPUBSplineInterpolateImageFunction implementation
  const std::string sourcePath1(GPUBSplineInterpolateImageFunctionKernel::GetOpenCLSource());
  m_Sources.push_back(sourcePath1);

  m_SourcesLoaded = true; // we set it to true, sources are loaded from strings
}

//------------------------------------------------------------------------------
template< class TInputImage, class TCoordRep , class TCoefficientType >
void GPUBSplineInterpolateImageFunction<TInputImage, TCoordRep, TCoefficientType>
::SetInputImage( const TInputImage *inputData )
{
  Superclass::SetInputImage( inputData );
  m_GPUCoefficients->Graft( this->m_Coefficients );
}

//------------------------------------------------------------------------------
template< class TInputImage, class TCoordRep, class TCoefficientType >
bool GPUBSplineInterpolateImageFunction< TInputImage, TCoordRep, TCoefficientType >
::GetSourceCode(std::string &_source) const
{
  if(!m_SourcesLoaded)
    return false;

  // Create the source code
  std::ostringstream source;

  // Variable length array declaration not allowed in OpenCL, therefore we are using #define
  source << "#define GPUBSplineOrder (" << this->m_SplineOrder << ")" << std::endl;

  // Calculate MaxNumberInterpolationPoints
  unsigned int maxNumberInterpolationPoints = 1;
  for(unsigned int n = 0; n < InputImageDimension; n++)
  {
    maxNumberInterpolationPoints *= (this->m_SplineOrder + 1);
  }
  // Variable length array declaration not allowed in OpenCL, therefore we are using #define
  source << "#define GPUMaxNumberInterpolationPoints (" << maxNumberInterpolationPoints << ")" << std::endl;

  // Add other sources
  for(std::size_t i=0; i<m_Sources.size(); i++)
  {
    source << m_Sources[i] << std::endl;
  }

  _source = source.str();
  return true;
}

//------------------------------------------------------------------------------
template< class TInputImage, class TCoordRep, class TCoefficientType >
void GPUBSplineInterpolateImageFunction< TInputImage, TCoordRep, TCoefficientType >
::PrintSelf(std::ostream & os, Indent indent) const
{
  //CPUSuperclass::PrintSelf(os, indent);
  GPUSuperclass::PrintSelf(os, indent);
}

} // namespace

#endif
