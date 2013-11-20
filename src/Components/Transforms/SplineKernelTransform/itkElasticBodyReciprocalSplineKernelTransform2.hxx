/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkElasticBodyReciprocalSplineKernelTransform2.txx,v $
  Language:  C++
  Date:      $Date: 2004/12/12 22:05:02 $
  Version:   $Revision: 1.8 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _itkElasticBodyReciprocalSplineKernelTransform2_hxx
#define _itkElasticBodyReciprocalSplineKernelTransform2_hxx

#include "itkElasticBodyReciprocalSplineKernelTransform2.h"

namespace itk
{

template <class TScalarType, unsigned int NDimensions>
ElasticBodyReciprocalSplineKernelTransform2<TScalarType, NDimensions>::
ElasticBodyReciprocalSplineKernelTransform2()
{
  this->m_Alpha = 8.0 * ( 1.0 - .25 ) - 1.0;
}


template <class TScalarType, unsigned int NDimensions>
void
ElasticBodyReciprocalSplineKernelTransform2<TScalarType, NDimensions>
::ComputeG( const InputVectorType & x, GMatrixType & GMatrix) const
{
  const TScalarType r       = x.GetNorm();
  const TScalarType factor  =
    ( r > 1e-8 ) ? ( -1.0 / r ): NumericTraits<TScalarType>::Zero;
  const TScalarType radial  = this->m_Alpha * r;
  for ( unsigned int i = 0; i < NDimensions; i++ )
  {
    const typename InputVectorType::ValueType xi = x[ i ] * factor;
    // G is symmetric
    for ( unsigned int j = 0; j < i; j++ )
    {
      const TScalarType value = xi * x[ j ];
      GMatrix[ i ][ j ] = value;
      GMatrix[ j ][ i ] = value;
    }
    GMatrix[ i ][ i ] =  radial + xi * x[ i ];
  }

} // end ComputeG()


template <class TScalarType, unsigned int NDimensions>
void
ElasticBodyReciprocalSplineKernelTransform2<TScalarType, NDimensions>
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "m_Alpha: " << this->m_Alpha << std::endl;

} // end PrintSelf()


} // namespace itk

#endif
