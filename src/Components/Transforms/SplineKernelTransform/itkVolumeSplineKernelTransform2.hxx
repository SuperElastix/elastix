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
  Module:    $RCSfile: itkVolumeSplineKernelTransform2.txx,v $
  Language:  C++
  Date:      $Date: 2006/03/18 18:06:38 $
  Version:   $Revision: 1.8 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkVolumeSplineKernelTransform2_hxx
#define __itkVolumeSplineKernelTransform2_hxx

#include "itkVolumeSplineKernelTransform2.h"

namespace itk
{

template< class TScalarType, unsigned int NDimensions >
void
VolumeSplineKernelTransform2< TScalarType, NDimensions >
::ComputeG( const InputVectorType & x, GMatrixType & GMatrix ) const
{
  const TScalarType r = x.GetNorm();
  GMatrix.fill( NumericTraits< TScalarType >::Zero );
  const TScalarType r3 = r * r * r;
  GMatrix.fill_diagonal( r3 );
//   for ( unsigned int i = 0; i < NDimensions; i++ )
//   {
//     GMatrix[ i ][ i ] = r3;
//   }

} // end ComputeG()


template< class TScalarType, unsigned int NDimensions >
void
VolumeSplineKernelTransform2< TScalarType, NDimensions >
::ComputeDeformationContribution(
  const InputPointType  & thisPoint, OutputPointType & opp ) const
{
  const unsigned long numberOfLandmarks = this->m_SourceLandmarks->GetNumberOfPoints();
  PointsIterator      sp                = this->m_SourceLandmarks->GetPoints()->Begin();

  for( unsigned int lnd = 0; lnd < numberOfLandmarks; lnd++ )
  {
    InputVectorType   position = thisPoint - sp->Value();
    const TScalarType r        = position.GetNorm();
    const TScalarType r3       = r * r * r;

    for( unsigned int odim = 0; odim < NDimensions; odim++ )
    {
      opp[ odim ] += r3 * this->m_DMatrix( odim, lnd );
    }
    ++sp;
  }

} // end ComputeDeformationContribution()


} // namespace itk

#endif
