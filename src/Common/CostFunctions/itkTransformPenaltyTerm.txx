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
      itkExceptionMacro( << "ERROR: The transform is not an AdvancedTransform, which is needed for this penalty term." );
    }
  }

} // end Initialize()


} // end namespace itk

#endif // #ifndef __itkTransformPenaltyTerm_txx

