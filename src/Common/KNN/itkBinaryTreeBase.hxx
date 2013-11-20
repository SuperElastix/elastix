/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkBinaryTreeBase_hxx
#define __itkBinaryTreeBase_hxx

#include "itkBinaryTreeBase.h"

namespace itk
{

/**
 * ************************ Constructor *************************
 */

template < class TListSample >
BinaryTreeBase<TListSample>
::BinaryTreeBase()
{
  this->m_Sample = 0;
} // end Constructor


/**
 * ************************ GetNumberOfDataPoints *************************
 */

template < class TListSample >
typename BinaryTreeBase<TListSample>::TotalAbsoluteFrequencyType
BinaryTreeBase<TListSample>
::GetNumberOfDataPoints( void ) const
{
  if ( this->m_Sample )
  {
    return this->m_Sample->GetTotalFrequency();
  }
  return NumericTraits< TotalAbsoluteFrequencyType >::Zero;

} // end GetNumberOfDataPoints()


/**
 * ************************ GetActualNumberOfDataPoints *************************
 */

template < class TListSample >
typename BinaryTreeBase<TListSample>::TotalAbsoluteFrequencyType
BinaryTreeBase<TListSample>
::GetActualNumberOfDataPoints( void ) const
{
  if ( this->m_Sample )
  {
    return this->m_Sample->GetActualSize();
  }
  return NumericTraits< TotalAbsoluteFrequencyType >::Zero;

} // end GetActualNumberOfDataPoints()


/**
 * ************************ GetDataDimension *************************
 */

template < class TListSample >
typename BinaryTreeBase<TListSample>::MeasurementVectorSizeType
BinaryTreeBase<TListSample>
::GetDataDimension( void ) const
{
  if ( this->m_Sample )
  {
    return this->m_Sample->GetMeasurementVectorSize();
  }
  return NumericTraits< MeasurementVectorSizeType >::Zero;

} // end GetDataDimension()


/*
 * ****************** PrintSelf ******************
 */

template < class TListSample >
void
BinaryTreeBase<TListSample>
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "Sample: " << this->m_Sample.GetPointer() << std::endl;

} // end PrintSelf()


} // end namespace itk


#endif // end #ifndef __itkBinaryTreeBase_hxx
