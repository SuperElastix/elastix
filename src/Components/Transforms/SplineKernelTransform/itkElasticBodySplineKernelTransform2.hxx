/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
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
#ifndef _itkElasticBodySplineKernelTransform2_hxx
#define _itkElasticBodySplineKernelTransform2_hxx

#include "itkElasticBodySplineKernelTransform2.h"

namespace itk
{

template< class TScalarType, unsigned int NDimensions >
ElasticBodySplineKernelTransform2< TScalarType, NDimensions >::ElasticBodySplineKernelTransform2()
{
  this->m_Alpha = 12.0 * ( 1.0 - .25 ) - 1.0;
}


template< class TScalarType, unsigned int NDimensions >
void
ElasticBodySplineKernelTransform2< TScalarType, NDimensions >
::ComputeG( const InputVectorType & x, GMatrixType & GMatrix ) const
{
  const TScalarType r      = x.GetNorm();
  const TScalarType factor = -3.0 * r;
  const TScalarType radial = this->m_Alpha * r * r * r;
  for( unsigned int i = 0; i < NDimensions; i++ )
  {
    const typename InputVectorType::ValueType xi = x[ i ] * factor;
    // G is symmetric
    for( unsigned int j = 0; j < i; j++ )
    {
      const TScalarType value = xi * x[ j ];
      GMatrix[ i ][ j ] = value;
      GMatrix[ j ][ i ] = value;
    }
    GMatrix[ i ][ i ] =  radial + xi * x[ i ];
  }

} // end ComputeG()


template< class TScalarType, unsigned int NDimensions >
void
ElasticBodySplineKernelTransform2< TScalarType, NDimensions >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "m_Alpha: " << this->m_Alpha << std::endl;

} // end PrintSelf()


} // namespace itk

#endif
