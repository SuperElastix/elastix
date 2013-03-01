/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUBSplineBaseTransform_hxx
#define __itkGPUBSplineBaseTransform_hxx

#include "itkGPUBSplineBaseTransform.h"

#include <iomanip>

//------------------------------------------------------------------------------
namespace itk
{
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
GPUBSplineBaseTransform<TScalarType, NDimensions, VSplineOrder>::GPUBSplineBaseTransform()
{
  // Add GPUBSplineTransform source
  const std::string sourcePath(
    GPUBSplineTransformKernel::GetOpenCLSource() );

  m_Sources.push_back( sourcePath );

  m_SourcesLoaded = true; // we set it to true, sources are loaded from strings
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder >
bool GPUBSplineBaseTransform<TScalarType, NDimensions, VSplineOrder>
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
} // end namespace itk

#endif /* __itkGPUBSplineBaseTransform_hxx */
