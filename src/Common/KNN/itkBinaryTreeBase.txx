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
	 * ************************ Destructor *************************
	 */

	template < class TListSample >
		BinaryTreeBase<TListSample>
		::~BinaryTreeBase()
	{
  } // end Destructor

} // end namespace itk


#endif // end #ifndef __itkBinaryTreeBase_txx

