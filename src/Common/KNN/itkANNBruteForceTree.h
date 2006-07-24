#ifndef __itkANNBruteForceTree_h
#define __itkANNBruteForceTree_h

#include "itkBinaryANNTreeBase.h"

namespace itk
{
	
	/**
	 * \class ANNBruteForceTree
	 *
	 * \brief 
	 *
	 * 
	 * \ingroup Metrics?
	 */
	
	template < class TListSample >
  class ANNBruteForceTree : public BinaryANNTreeBase< TListSample >
	{
	public:
		
		/** Standard itk. */
		typedef ANNBruteForceTree                 Self;
		typedef BinaryANNTreeBase< TListSample >	Superclass;
		typedef SmartPointer< Self >				      Pointer;
		typedef SmartPointer< const Self >	      ConstPointer;
		
		/** New method for creating an object using a factory. */
		itkNewMacro( Self );
		
		/** ITK type info. */
		itkTypeMacro( ANNBruteForceTree, BinaryANNTreeBase );
		
    /** Typedef's from Superclass. */
    typedef typename Superclass::SampleType                 SampleType;
    typedef typename Superclass::MeasurementVectorType      MeasurementVectorType;
    typedef typename Superclass::MeasurementVectorSizeType  MeasurementVectorSizeType;
    typedef typename Superclass::TotalFrequencyType         TotalFrequencyType;

    /** Typedef's. */
    typedef ANNpointSet       ANNPointSetType;
    typedef ANNbruteForce     ANNBruteForceTreeType;
    
    /** Set the maximum number of points that are to be visited. *
    void SetMaximumNumberOfPointsToVisit( unsigned int num )
    {
      annMaxPtsVisit(	static_cast<int>( num ) );
    }

    /** Generate the tree. */
    virtual void GenerateTree( void );

    /** Get the ANN tree. */
    virtual ANNPointSetType * GetANNTree( void ) const
    {
      return this->m_ANNTree;
    }
   
	protected:
		
		ANNBruteForceTree();
		virtual ~ANNBruteForceTree();

    /** Member variables. */
    ANNBruteForceTreeType *       m_ANNTree;
    
	private:
		
		ANNBruteForceTree( const Self& ); // purposely not implemented
		void operator=( const Self& );    // purposely not implemented
		
	}; // end class ANNBruteForceTree
		
		
} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkANNBruteForceTree.txx"
#endif


#endif // end #ifndef __itkANNBruteForceTree_h

