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
  Module:    $RCSfile: itkThinPlateSplineKernelTransform2.txx,v $
  Language:  C++
  Date:      $Date: 2006-11-28 14:22:18 $
  Version:   $Revision: 1.1 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _itkThinPlateSplineKernelTransform2_hxx
#define _itkThinPlateSplineKernelTransform2_hxx

#include "itkThinPlateSplineKernelTransform2.h"

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template< class TScalarType, unsigned int NDimensions >
void
ThinPlateSplineKernelTransform2< TScalarType, NDimensions >
::ComputeG( const InputVectorType & x, GMatrixType & GMatrix ) const
{
  const TScalarType r = x.GetNorm();
  GMatrix.fill( NumericTraits< TScalarType >::Zero );
  GMatrix.fill_diagonal( r );

} // end ComputeG()


/**
 * ******************* ComputeDeformationContribution *******************
 */

template< class TScalarType, unsigned int NDimensions >
void
ThinPlateSplineKernelTransform2< TScalarType, NDimensions >
::ComputeDeformationContribution(
  const InputPointType & thisPoint, OutputPointType & opp ) const
{
  const unsigned long numberOfLandmarks = this->m_SourceLandmarks->GetNumberOfPoints();
  PointsIterator      sp                = this->m_SourceLandmarks->GetPoints()->Begin();

  for( unsigned long lnd = 0; lnd < numberOfLandmarks; lnd++ )
  {
    InputVectorType   position = thisPoint - sp->Value();
    const TScalarType r        = position.GetNorm();

    for( unsigned int odim = 0; odim < NDimensions; odim++ )
    {
      opp[ odim ] += r * this->m_DMatrix( odim, lnd );
    }
    ++sp;
  }

} // end ComputeDeformationContribution()


} // namespace itk

#endif
