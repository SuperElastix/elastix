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
		m_TempField = 0;
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
		::InitializeDeformationFields( void )
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

		/** Initialize m_TempField. */
		m_TempField = VectorImageType::New();
		m_TempField->SetRegions( m_DeformationFieldRegion );
		m_TempField->SetSpacing( m_DeformationFieldSpacing );
		m_TempField->SetOrigin( m_DeformationFieldOrigin );
		m_TempField->Allocate();

		/** Set the deformation field in the transform. */
		m_IntermediaryDeformationFieldTransform->SetCoefficientImage( m_IntermediaryDeformationField );

		/** Set to initialized. */
		m_Initialized = true;

	} // end InitializeDeformationFields


	/**
	 * *********************** TransformPoint ***********************
	 */

	template <class TAnyITKTransform>
		typename DeformationFieldRegulizer<TAnyITKTransform>::OutputPointType
		DeformationFieldRegulizer<TAnyITKTransform>
		::TransformPoint( const InputPointType & inputPoint ) const
	{
		/** Get the outputpoint of the BSpline and the deformationfield. */
		OutputPointType oppBS, oppDF, opp;
		oppBS = this->Superclass::TransformPoint( inputPoint );
		oppDF = m_IntermediaryDeformationFieldTransform->TransformPoint( inputPoint );

		/** Add them. */
		for ( unsigned int i = 0; i < OutputSpaceDimension; i++ )
		{
			opp[ i ] = oppBS[ i ] + oppDF[ i ] - inputPoint[ i ];
		}

		/** Return a value. */
		return opp;

	} // end TransformPoint

	/**
	 * ******** UpdateIntermediaryDeformationFieldTransformTemp *********
	 */

	template <class TAnyITKTransform>
		void DeformationFieldRegulizer<TAnyITKTransform>
		::UpdateIntermediaryDeformationFieldTransformTemp( typename VectorImageType::Pointer vecImage )
	{
		/** Initialize the deformation fields. */
		if ( !m_Initialized )
		{
			this->InitializeDeformationFields();
		}

		/** Create iterators. */
		IteratorType it0( m_TempField,
			m_TempField->GetLargestPossibleRegion() );
		IteratorType it2( vecImage,
			vecImage->GetLargestPossibleRegion() );

		/** Update IntermediaryDeformationFieldTransform. */
		it0.GoToBegin();
		it2.GoToBegin();
		while ( !it0.IsAtEnd() )
		{
			/** Replace the old deformation field with the new one. */
			it0.Set( it2.Get() );

			/** Increase iterators. */
			++it0;
			++it2;
		}
		
		/** Set the updated deformation field in the transform. */
		m_IntermediaryDeformationFieldTransform->SetCoefficientImage( m_TempField );

	} // end UpdateIntermediaryDeformationFieldTransformTemp


	/**
	 * ******** UpdateIntermediaryDeformationFieldTransform *********
	 */

	template <class TAnyITKTransform>
		void DeformationFieldRegulizer<TAnyITKTransform>
		::UpdateIntermediaryDeformationFieldTransform( typename VectorImageType::Pointer  vecImage )
	{
		/** Initialize the deformation fields. */
		if ( !m_Initialized )
		{
			this->InitializeDeformationFields();
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
			/** Replace the old deformation field with the new one. */
			it1.Set( it2.Get() );

			/** Increase iterators. */
			++it1;
			++it2;
		}
		
		/** Set the updated deformation field in the transform. */
		m_IntermediaryDeformationFieldTransform->SetCoefficientImage( m_IntermediaryDeformationField );

	} // end UpdateIntermediaryDeformationFieldTransform


} // end namespace itk


#endif // end #ifndef __itkDeformationFieldRegulizer_HXX__

