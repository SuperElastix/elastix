#ifndef __itkDeformationFieldRegulizer_HXX__
#define __itkDeformationFieldRegulizer_HXX__

#include "itkDeformationFieldRegulizer.h"


namespace itk
{
	
	/**
	 * ************************ Constructor	*************************
	 */

	template <class TAnyITKTransform>
		DeformationFieldRegulizer<TAnyITKTransform>
		::DeformationFieldRegulizer()
	{
		/** Initialize. */
		m_IntermediaryDeformationFieldTransform = 0;
		m_IntermediaryDeformationField = 0;
		m_Initialized = false;
				
	} // end Constructor
	

	/**
	 * *********************** Destructor ***************************
	 */

	template <class TAnyITKTransform>
		DeformationFieldRegulizer<TAnyITKTransform>
		::~DeformationFieldRegulizer()
	{
		//nothing
	} // end Destructor


	/**
	 * ********* InitializeIntermediaryDeformationField **************
	 */

	template <class TAnyITKTransform>
		void DeformationFieldRegulizer<TAnyITKTransform>
		::InitializeIntermediaryDeformationField( void )
	{
		/** Initialize m_IntermediaryDeformationFieldTransform. */
		m_IntermediaryDeformationFieldTransform = IntermediaryDFTransformType::New();

		/** Initialize m_IntermediaryDeformationField. */
		m_IntermediaryDeformationField = VectorImageType::New();
		m_IntermediaryDeformationField->SetRegions( m_DeformationFieldRegion );
		m_IntermediaryDeformationField->SetSpacing( m_DeformationFieldSpacing );
		m_IntermediaryDeformationField->SetOrigin( m_DeformationFieldOrigin );
		m_IntermediaryDeformationField->Allocate();

		/** Set everything to zero. */
		IteratorType it( m_IntermediaryDeformationField,
			m_IntermediaryDeformationField->GetLargestPossibleRegion() );
		VectorPixelType vec;
		vec.Fill( NumericTraits<ScalarType>::Zero );
		while ( !it.IsAtEnd() )
		{
			it.Set( vec );
			++it;
		}

		/** Set the deformation field in the transform. */
		m_IntermediaryDeformationFieldTransform->SetCoefficientImage( m_IntermediaryDeformationField );

		/** Set to initialized. */
		m_Initialized = true;

	} // end InitializeIntermediaryDeformationField


	/**
	 * *********************** TransformPoint ***********************
	 */

	template <class TAnyITKTransform>
		typename DeformationFieldRegulizer<TAnyITKTransform>::OutputPointType
		DeformationFieldRegulizer<TAnyITKTransform>
		::TransformPoint( const InputPointType & point ) const
	{
		/** Get the outputpoint of the BSpline and the deformationfield. */
		OutputPointType oppBS, oppDF, oppSum;
		oppBS = this->Superclass::TransformPoint( point );
		oppDF = m_IntermediaryDeformationFieldTransform->TransformPoint( point );

		/** Add them. */
		for ( unsigned int i = 0; i < OutputSpaceDimension; i++ )
		{
			oppSum[ i ] = oppBS[ i ] + oppDF[ i ];
		}

		/** Return a value. */
		return oppSum;

	} // end TransformPoint


	/**
	 * ******** UpdateIntermediaryDeformationFieldTransform *********
	 */

	template <class TAnyITKTransform>
		void DeformationFieldRegulizer<TAnyITKTransform>
		::UpdateIntermediaryDeformationFieldTransform( VectorImagePointer vecImage )
	{
		/** Initialize m_IntermediaryDeformationFieldTransform. */
		if ( !m_Initialized )
		{
			this->InitializeIntermediaryDeformationField();
		}

		/** Create iterators. */
		IteratorType it1( m_IntermediaryDeformationField,
			m_IntermediaryDeformationField->GetLargestPossibleRegion() );
		IteratorType it2( vecImage,
			vecImage->GetLargestPossibleRegion() );

		/** Update m_IntermediaryDeformationFieldTransform. */
		it1.GoToBegin();
		it2.GoToBegin();
		while ( !it1.IsAtEnd() )
		{
			it1.Set( it1.Get() + it2.Get() );

			/** Increase iterators. */
			++it1;
			++it2;
		}
		
		/** Set the updated deformation field in the transform. */
		m_IntermediaryDeformationFieldTransform->SetCoefficientImage( m_IntermediaryDeformationField );

	} // end SetIntermediaryDeformationFieldTransform


} // end namespace itk


#endif // end #ifndef __itkDeformationFieldRegulizer_HXX__

