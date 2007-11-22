#ifndef __itkBinaryTreeBase_txx
#define __itkBinaryTreeBase_txx

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
    typename BinaryTreeBase<TListSample>::TotalFrequencyType
    BinaryTreeBase<TListSample>
    ::GetNumberOfDataPoints( void ) const
  {
    if ( this->m_Sample )
    {
      return this->m_Sample->GetTotalFrequency();
    }
    return NumericTraits< TotalFrequencyType >::Zero;

  } // end GetNumberOfDataPoints()


  /**
	 * ************************ GetActualNumberOfDataPoints *************************
	 */

  template < class TListSample >
    typename BinaryTreeBase<TListSample>::TotalFrequencyType
    BinaryTreeBase<TListSample>
    ::GetActualNumberOfDataPoints( void ) const
  {
    if ( this->m_Sample )
    {
      return this->m_Sample->GetActualSize();
    }
    return NumericTraits< TotalFrequencyType >::Zero;

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


#endif // end #ifndef __itkBinaryTreeBase_txx

