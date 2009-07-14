/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkElasticBodySplineKernelTransform2.txx,v $
  Language:  C++
  Date:      $Date: 2004/12/12 22:05:02 $
  Version:   $Revision: 1.20 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _itkElasticBodySplineKernelTransform2_txx
#define _itkElasticBodySplineKernelTransform2_txx
#include "itkElasticBodySplineKernelTransform2.h"

namespace itk
{

template <class TScalarType, unsigned int NDimensions>
ElasticBodySplineKernelTransform2<TScalarType, NDimensions>::
ElasticBodySplineKernelTransform2() 
{
  // Alpha = 12 ( 1 - \nu ) - 1
  m_Alpha = 12.0 * ( 1.0 - .25 ) - 1;
}

template <class TScalarType, unsigned int NDimensions>
ElasticBodySplineKernelTransform2<TScalarType, NDimensions>::
~ElasticBodySplineKernelTransform2()
{
}

template <class TScalarType, unsigned int NDimensions>
void
ElasticBodySplineKernelTransform2<TScalarType, NDimensions>
::ComputeG(const InputVectorType & x, GMatrixType & GMatrix) const
{
  const TScalarType r       = x.GetNorm();
  const TScalarType factor  = -3.0 * r;
  const TScalarType radial  = m_Alpha * ( r * r ) * r;
  for(unsigned int i=0; i<NDimensions; i++)
    {
    const typename InputVectorType::ValueType xi = x[i] * factor;
    // G is symmetric
    for(unsigned int j=0; j<i; j++)
      {
      const TScalarType value = xi * x[j]; 
      GMatrix[i][j] = value;
      GMatrix[j][i] = value;
      }
    GMatrix[i][i] =  radial + xi * x[i];
    }
 
}

template <class TScalarType, unsigned int NDimensions>
void
ElasticBodySplineKernelTransform2<TScalarType, NDimensions>
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);
  os << indent << "m_Alpha: " << m_Alpha << std::endl;
}

} // namespace itk
#endif
