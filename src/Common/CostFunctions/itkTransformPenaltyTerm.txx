/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkTransformPenaltyTerm_txx
#define __itkTransformPenaltyTerm_txx

#include "itkTransformPenaltyTerm.h"


namespace itk
{

/**
 * ****************** Constructor *******************************
 */

template< class TFixedImage, class TScalarType >
TransformPenaltyTerm< TFixedImage, TScalarType >
::TransformPenaltyTerm()
{
  /** Initialize member variables. */
  this->m_AdvancedTransform  = 0;

} // end constructor


/**
 * ****************** PrintSelf *******************************
 */

template< class TFixedImage, class TScalarType >
void
TransformPenaltyTerm< TFixedImage, TScalarType >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  this->Superclass::PrintSelf( os, indent );

  os << indent << "AdvancedTransform: "
    << this->m_AdvancedTransform.GetPointer() << std::endl;

} // end PrintSelf()


/**
 * ****************** Initialize *******************************
 */

template< class TFixedImage, class TScalarType >
void
TransformPenaltyTerm< TFixedImage, TScalarType >
::Initialize( void ) throw ( ExceptionObject )
{
  /** Call the superclass to check that standard components are available. */
  this->Superclass::Initialize();

  /** Check for the transform. */
  if ( !this->GetTransform() )
  {
    itkExceptionMacro( << "Transform is not present" );
  }
  else
  {
    /** Try to cast to an AdvancedTransform. */
    this->m_AdvancedTransform
      = dynamic_cast< TransformType * >( this->m_Transform.GetPointer() );
    if ( !this->m_AdvancedTransform )
    {
      itkExceptionMacro( << "ERROR: The transform is not an AdvancedTransform,"
        << " which is needed for this penalty term." );
    }
  }

} // end Initialize()


/**
 * ****************** CheckForBSplineTransform *******************************
 */

template< class TFixedImage, class TScalarType >
bool
TransformPenaltyTerm< TFixedImage, TScalarType >
::CheckForBSplineTransform( BSplineTransformPointer & bspline ) const
{
  /** Check if this transform is a B-spline transform. */
  BSplineTransformType * testPtr1
    = dynamic_cast<BSplineTransformType *>( this->m_AdvancedTransform.GetPointer() );
  CombinationTransformType * testPtr2a
    = dynamic_cast<CombinationTransformType *>( this->m_AdvancedTransform.GetPointer() );
  bool transformIsBSpline = false;
  if ( testPtr1 )
  {
    /** The transform is of type AdvancedBSplineDeformableTransform. */
    transformIsBSpline = true;
    bspline = testPtr1;
  }
  else if ( testPtr2a )
  {
    /** The transform is of type AdvancedCombinationTransform. */
    BSplineTransformType * testPtr2b = dynamic_cast<BSplineTransformType *>(
      (testPtr2a->GetCurrentTransform()) );
    if ( testPtr2b )
    {
      /** The current transform is of type AdvancedBSplineDeformableTransform. */
      transformIsBSpline = true;
      bspline = testPtr2b;
    }
  }

  return transformIsBSpline;

} // end CheckForBSplineTransform()


} // end namespace itk

#endif // #ifndef __itkTransformPenaltyTerm_txx

