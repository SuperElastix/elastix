#ifndef __itkANNBruteForceTree_txx
#define __itkANNBruteForceTree_txx

#include "itkANNBruteForceTree.h"
#include "itkANNBinaryTreeCreator.h"

namespace itk
{
	
	/**
	 * ************************ Constructor	*************************
	 */

	template < class TListSample >
		ANNBruteForceTree<TListSample>
		::ANNBruteForceTree()
	{
    this->m_ANNTree = 0;    
  } // end Constructor


  /**
	 * ************************ Destructor *************************
	 */

	template < class TListSample >
		ANNBruteForceTree<TListSample>
		::~ANNBruteForceTree()
	{
    ANNBinaryTreeCreator::DeleteANNBruteForceTree( this->m_ANNTree );
  } // end Destructor


  /**
	 * ************************ GenerateTree *************************
	 */

	template < class TListSample >
    void ANNBruteForceTree<TListSample>
		::GenerateTree( void )
	{
    int dim = static_cast< int >( this->GetDataDimension() );
    int nop = static_cast< int >( this->GetActualNumberOfDataPoints() );
    
    ANNBinaryTreeCreator::DeleteANNBruteForceTree( this->m_ANNTree );

    this->m_ANNTree = ANNBinaryTreeCreator::CreateANNBruteForceTree(
      this->GetSample()->GetInternalContainer(), nop, dim );

  } // end GenerateTree
  

} // end namespace itk


#endif // end #ifndef __itkANNBruteForceTree_txx

