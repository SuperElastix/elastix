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
  Module:    $RCSfile: itkThinPlateR2LogRSplineKernelTransform2.txx,v $
  Language:  C++
  Date:      $Date: 2006/03/19 04:36:59 $
  Version:   $Revision: 1.8 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _itkThinPlateR2LogRSplineKernelTransform2_hxx
#define _itkThinPlateR2LogRSplineKernelTransform2_hxx

#include "itkThinPlateR2LogRSplineKernelTransform2.h"

namespace itk
{

template< class TScalarType, unsigned int NDimensions >
void
ThinPlateR2LogRSplineKernelTransform2< TScalarType, NDimensions >::ComputeG( const InputVectorType & x, GMatrixType & GMatrix ) const
{
  const TScalarType r = x.GetNorm();
  GMatrix.fill( NumericTraits< TScalarType >::Zero );
  const TScalarType          R2logR
    = ( r > 1e-8 ) ? r * r * vcl_log( r ) : NumericTraits< TScalarType >::Zero;

  GMatrix.fill_diagonal( R2logR );
}


template< class TScalarType, unsigned int NDimensions >
void
ThinPlateR2LogRSplineKernelTransform2< TScalarType, NDimensions >::ComputeDeformationContribution( const InputPointType  & thisPoint,
  OutputPointType & result     ) const
{
  const unsigned long numberOfLandmarks = this->m_SourceLandmarks->GetNumberOfPoints();

  PointsIterator sp = this->m_SourceLandmarks->GetPoints()->Begin();

  for( unsigned int lnd = 0; lnd < numberOfLandmarks; lnd++ )
  {
    InputVectorType            position = thisPoint - sp->Value();
    const TScalarType          r        = position.GetNorm();
    const TScalarType          R2logR
      = ( r > 1e-8 ) ? r * r * vcl_log( r ) : NumericTraits< TScalarType >::Zero;
    for( unsigned int odim = 0; odim < NDimensions; odim++ )
    {
      result[ odim ] += R2logR * this->m_DMatrix( odim, lnd );
    }
    ++sp;
  }

}


} // namespace itk

#endif
