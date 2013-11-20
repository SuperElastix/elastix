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
  Module:    $RCSfile: itkAdvancedTransform.txx,v $
  Language:  C++
  Date:      $Date: 2007-11-20 20:08:16 $
  Version:   $Revision: 1.27 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _itkAdvancedTransform_hxx
#define _itkAdvancedTransform_hxx

#include "itkAdvancedTransform.h"

namespace itk
{


/**
 * ********************* Constructor ****************************
 */

template < class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
AdvancedTransform<TScalarType,NInputDimensions,NOutputDimensions>
::AdvancedTransform() : Superclass()
{
  this->m_HasNonZeroSpatialHessian = true;
  this->m_HasNonZeroJacobianOfSpatialHessian = true;

} // end Constructor


/**
 * ********************* Constructor ****************************
 */

template < class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
AdvancedTransform<TScalarType,NInputDimensions,NOutputDimensions>
::AdvancedTransform( NumberOfParametersType numberOfParameters ) :
  Superclass( numberOfParameters )
{
  this->m_HasNonZeroSpatialHessian = true;
  this->m_HasNonZeroJacobianOfSpatialHessian = true;
} // end Constructor


/**
 * ********************* GetNumberOfNonZeroJacobianIndices ****************************
 */

template < class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
typename AdvancedTransform<TScalarType,NInputDimensions,NOutputDimensions>::NumberOfParametersType
AdvancedTransform<TScalarType,NInputDimensions,NOutputDimensions>
::GetNumberOfNonZeroJacobianIndices( void ) const
{
  return this->GetNumberOfParameters();

} // end GetNumberOfNonZeroJacobianIndices()


} // end namespace itk


#endif
