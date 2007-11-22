#ifndef __itkBinaryTreeBase_h
#define __itkBinaryTreeBase_h

#include "itkDataObject.h"

namespace itk
{
	
	/**
	 * \class BinaryTreeBase
	 *
	 * \brief 
	 *
	 * 
	 * \ingroup Metrics?
	 */
	
	template < class TListSample >
	class BinaryTreeBase : public DataObject
	{
	public:
		
		/** Standard itk. */
		typedef BinaryTreeBase              Self;
		typedef DataObject									Superclass;
		typedef SmartPointer< Self >				Pointer;
		typedef SmartPointer< const Self >	ConstPointer;
		
		/** ITK type info. */
		itkTypeMacro( BinaryTreeBase, DataObject );
		
    /** Typedef's. */
    typedef TListSample       SampleType;

    /** Typedef's. */
    typedef typename SampleType::MeasurementVectorType      MeasurementVectorType;
    typedef typename SampleType::MeasurementVectorSizeType  MeasurementVectorSizeType;
    typedef typename SampleType::TotalFrequencyType         TotalFrequencyType;

    /** Set and get the samples: the array of points. */
    itkSetObjectMacro( Sample, SampleType );
    itkGetConstObjectMacro( Sample, SampleType );

    /** Get the number of data points. */
    TotalFrequencyType GetNumberOfDataPoints( void ) const;

    /** Get the actual number of data points. */
    TotalFrequencyType GetActualNumberOfDataPoints( void ) const;

    /** Get the dimension of the input data. */
    MeasurementVectorSizeType GetDataDimension( void ) const;
    
    /** Generate the tree. */
    virtual void GenerateTree( void ) = 0;
   
	protected:
		
    /** Constructor. */
		BinaryTreeBase();

    /** Destructor. */
    virtual ~BinaryTreeBase() {};

    /** PrintSelf. */
    virtual void PrintSelf( std::ostream& os, Indent indent ) const;
    
	private:
		
		BinaryTreeBase( const Self& );  // purposely not implemented
		void operator=( const Self& );  // purposely not implemented

    /** Store the samples. */
    typename SampleType::Pointer        m_Sample;

	}; // end class BinaryTreeBase
		
		
} // end namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBinaryTreeBase.txx"
#endif


#endif // end #ifndef __itkBinaryTreeBase_h

