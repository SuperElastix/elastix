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
		typename VectorImageType::Pointer intermediaryDeformationField = VectorImageType::New();
		intermediaryDeformationField->SetRegions( m_DeformationFieldRegion );
		intermediaryDeformationField->SetSpacing( m_DeformationFieldSpacing );
		intermediaryDeformationField->SetOrigin( m_DeformationFieldOrigin );
		intermediaryDeformationField->Allocate();

		/** Set everything to zero. */
		IteratorType it( intermediaryDeformationField,
			intermediaryDeformationField->GetLargestPossibleRegion() );
		VectorPixelType vec;
		vec.Fill( NumericTraits<ScalarType>::Zero );
		while ( !it.IsAtEnd() )
		{
			it.Set( vec );
			++it;
		}

		/** Set the deformation field in the transform. */
		m_IntermediaryDeformationFieldTransform->SetCoefficientImage( intermediaryDeformationField );

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
		/** Get the outputpoint of any ITK Transform and the deformationfield. */
		OutputPointType oppAnyT, oppDF, opp;
		oppAnyT = this->Superclass::TransformPoint( inputPoint );
		oppDF = m_IntermediaryDeformationFieldTransform->TransformPoint( inputPoint );

		/** Add them: don't forget to subtract ipp. */
		for ( unsigned int i = 0; i < OutputSpaceDimension; i++ )
		{
			opp[ i ] = oppAnyT[ i ] + oppDF[ i ] - inputPoint[ i ];
		}

		/** Return a value. */
		return opp;

	} // end TransformPoint


	/**
	 * ******** UpdateIntermediaryDeformationFieldTransform *********
	 */

	template <class TAnyITKTransform>
		void DeformationFieldRegulizer<TAnyITKTransform>
		::UpdateIntermediaryDeformationFieldTransform( typename VectorImageType::Pointer  vecImage )
	{
		/** Set the vecImage (which is allocated elsewhere) and put it in
		 * Intermediary deformationFieldtransform (where it is copied and split up).
		 */
		m_IntermediaryDeformationFieldTransform->SetCoefficientImage( vecImage );

	} // end UpdateIntermediaryDeformationFieldTransform


} // end namespace itk


#endif // end #ifndef __itkDeformationFieldRegulizer_HXX__

