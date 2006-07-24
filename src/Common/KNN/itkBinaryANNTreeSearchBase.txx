#ifndef __itkBinaryANNTreeSearchBase_txx
#define __itkBinaryANNTreeSearchBase_txx

#include "itkBinaryANNTreeSearchBase.h"

namespace itk
{
	
	/**
	 * ************************ Constructor *************************
	 */

	template < class TBinaryTree >
		BinaryANNTreeSearchBase<TBinaryTree>
		::BinaryANNTreeSearchBase()
	{
    this->m_BinaryTreeAsITKANNType = 0;
  } // end Constructor


  /**
	 * ************************ Destructor *************************
	 */

	template < class TBinaryTree >
		BinaryANNTreeSearchBase<TBinaryTree>
		::~BinaryANNTreeSearchBase()
	{
  } // end Destructor


  /**
	 * ************************ SetBinaryTree *************************
	 */

	template < class TBinaryTree >
		void BinaryANNTreeSearchBase<TBinaryTree>
		::SetBinaryTree( BinaryTreeType * tree )
	{
    this->Superclass::SetBinaryTree( tree );
    if ( tree )
    {
      BinaryANNTreeType * testPtr = dynamic_cast<BinaryANNTreeType *>( tree );
      if ( testPtr )
      {
        if ( testPtr != this->m_BinaryTreeAsITKANNType )
        {
          this->m_BinaryTreeAsITKANNType = testPtr;
          this->Modified();
        }
      }
      else
      {
        itkExceptionMacro( << "ERROR: The tree is not of type BinaryANNTreeBase." );
      }
    }
    else
    {
      if ( this->m_BinaryTreeAsITKANNType.IsNotNull() )
      {
        this->m_BinaryTreeAsITKANNType = 0;
        this->Modified();
      }
    }

  } // end SetBinaryTree


  /**
	 * ************************ GetBinaryTree *************************
	 *

	template < class TBinaryTree >
    const typename BinaryANNTreeSearchBase<TBinaryTree>::BinaryTreeType *
    BinaryANNTreeSearchBase<TBinaryTree>
		::GetBinaryTree( void ) const
	{
    return this->m_BinaryTree.GetPointer();
  } // end GetBinaryTree
*/

} // end namespace itk


#endif // end #ifndef __itkBinaryANNTreeSearchBase_txx

