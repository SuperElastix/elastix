/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkSingleValuedPointSetToPointSetMetric.txx,v $
  Language:  C++
  Date:      $Date: 2009-01-26 21:45:56 $
  Version:   $Revision: 1.2 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkSingleValuedPointSetToPointSetMetric_txx
#define __itkSingleValuedPointSetToPointSetMetric_txx

#include "itkSingleValuedPointSetToPointSetMetric.h"

namespace itk
{

/**
 * ******************* Constructor ***********************
 */

template <class TFixedPointSet, class TMovingPointSet>
SingleValuedPointSetToPointSetMetric<TFixedPointSet,TMovingPointSet>
::SingleValuedPointSetToPointSetMetric()
{
  this->m_FixedPointSet   = 0; // has to be provided by the user.
  this->m_MovingPointSet  = 0; // has to be provided by the user.
  this->m_Transform       = 0; // has to be provided by the user.
  this->m_FixedImageMask  = 0;
  this->m_MovingImageMask = 0;

  this->m_NumberOfPointsCounted = 0;

} // end Constructor


/**
 * ******************* SetTransformParameters ***********************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
SingleValuedPointSetToPointSetMetric<TFixedPointSet,TMovingPointSet>
::SetTransformParameters( const ParametersType & parameters ) const
{
  if( !this->m_Transform )
  {
    itkExceptionMacro( << "Transform has not been assigned" );
  }
  this->m_Transform->SetParameters( parameters );

} // end SetTransformParameters()


/**
 * ******************* Initialize ***********************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
SingleValuedPointSetToPointSetMetric<TFixedPointSet,TMovingPointSet>
::Initialize( void) throw ( ExceptionObject )
{
  if ( !this->m_Transform )
  {
    itkExceptionMacro( << "Transform is not present" );
  }

  if ( !this->m_MovingPointSet )
  {
    itkExceptionMacro( << "MovingPointSet is not present" );
  }

  if ( !this->m_FixedPointSet )
  {
    itkExceptionMacro( << "FixedPointSet is not present" );
  }

  // If the PointSet is provided by a source, update the source.
  if ( this->m_MovingPointSet->GetSource() )
  {
    this->m_MovingPointSet->GetSource()->Update();
  }

  // If the point set is provided by a source, update the source.
  if ( this->m_FixedPointSet->GetSource() )
  {
    this->m_FixedPointSet->GetSource()->Update();
  }

} // end Initialize()


/**
 * ******************* PrintSelf ***********************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
SingleValuedPointSetToPointSetMetric<TFixedPointSet,TMovingPointSet>
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << "Fixed  PointSet: " << this->m_FixedPointSet.GetPointer() << std::endl;
  os << "Moving PointSet: " << this->m_MovingPointSet.GetPointer() << std::endl;
  os << "Fixed mask: " << this->m_FixedImageMask.GetPointer() << std::endl;
  os << "Moving mask: " << this->m_MovingImageMask.GetPointer() << std::endl;
  os << "Transform: " << this->m_Transform.GetPointer() << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef __itkSingleValuedPointSetToPointSetMetric_txx
