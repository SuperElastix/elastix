#ifndef __itkBSplineCombinationTransform_hxx
#define __itkBSplineCombinationTransform_hxx

#include "itkBSplineCombinationTransform.h"


namespace itk
{
	
	/**
	 * ************************ Constructor	*************************
	 */

	template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
		BSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>
		::BSplineCombinationTransform() : Superclass()
	{
		/** Initialize.*/
		this->m_CurrentTransformAsBSplineTransform = 0;
				
		this->m_SelectedTransformPointBSplineFunction = 
			&Self::TransformPointBSplineNoCurrentTransform;
		
	} // end Constructor
	
	
	/**
   * ****************** TransformPoint ****************************
	 */

	template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
		typename BSplineCombinationTransform<
			TScalarType, NDimensions, VSplineOrder>::OutputPointType
		BSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>::	
		TransformPoint(const InputPointType  & point ) const
	{ 
		return this->Superclass::TransformPoint( point );
	} // end TransformPoint


	/**
   * ****************** TransformPoint ****************************
	 */

	template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
		void BSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>::	
		TransformPoint(
			const InputPointType &inputPoint,
			OutputPointType &outputPoint,
			WeightsType &weights,
			ParameterIndexArrayType &indices, 
			bool &inside ) const
	{ 
		/** Call the selected TransformPointBSplineFunction */
		((*this).*m_SelectedTransformPointBSplineFunction)(
			inputPoint, outputPoint, weights,	indices, inside);

	} // end TransformPoint with extra arguments

	
	/**
	 * ******************* SetCurrentTransform **********************
	 */

	template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
		void BSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>
		::SetCurrentTransform( CurrentTransformType * _arg )
	{
		/** Set the the current transform and call the 
		 * the UpdateCombinationMethod */
		if ( this->m_CurrentTransform != _arg )
		{
			/** if a zero pointer is given: */
			if ( _arg == 0 )
			{
				this->m_CurrentTransform = 0;
				this->m_CurrentTransformAsBSplineTransform = 0;
				this->Modified();
				this->UpdateCombinationMethod();
				return;
			}

			/** if the pointer is nonzero, try to cast it to a BSpline
			 * transform */
			BSplineTransformType * testPointer = 
				dynamic_cast<BSplineTransformType *>( _arg );
			if ( testPointer )
			{
        this->m_CurrentTransform = _arg;
				this->m_CurrentTransformAsBSplineTransform = testPointer;
				this->Modified();
				this->UpdateCombinationMethod();
			}
			else
			{
				itkExceptionMacro(<< "The entered CurrentTransform is not a BSplineTransform.");
			}
		}

	} // end SetCurrentTransform


	/**
	 * ****************** UpdateCombinationMethod ********************
	 */

	template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
		void BSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>
		::UpdateCombinationMethod( void )
	{
		this->Superclass::UpdateCombinationMethod();

    /** Update the m_SelectedTransformPointBSplineFunction */
		if ( this->m_CurrentTransform.IsNull() )
		{
			this->m_SelectedTransformPointBSplineFunction = 
				&Self::TransformPointBSplineNoCurrentTransform;
		}
		else if ( this->m_InitialTransform.IsNull() )
		{
			this->m_SelectedTransformPointBSplineFunction = 
				&Self::TransformPointBSplineNoInitialTransform;
		}
		else if ( this->m_UseAddition )
		{
			this->m_SelectedTransformPointBSplineFunction = 
				&Self::TransformPointBSplineUseAddition;
		}
		else
		{
			this->m_SelectedTransformPointBSplineFunction = 
				&Self::TransformPointBSplineUseComposition;
		}

	} // end UpdateCombinationMethod


	/**
	 * ************* ADDITION: T(x) = T0(x) + T1(x) - x **********************
	 */

	template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
		void BSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>::
	  TransformPointBSplineUseAddition(
			const InputPointType &inputPoint,
			OutputPointType &outputPoint,
			WeightsType &weights,
			ParameterIndexArrayType &indices, 
			bool &inside ) const
	{				
		/** The Initial transform */		 
		OutputPointType out0 = 
			this->m_InitialTransform->TransformPoint( inputPoint );
		
		/** The Current transform */
		this->m_CurrentTransformAsBSplineTransform->TransformPoint( 
			inputPoint, outputPoint, weights,	indices, inside);
		
		/** Both added together */
		for ( unsigned int i=0; i < SpaceDimension; i++ )
		{
			outputPoint[ i ] += ( out0[ i ] - inputPoint[ i ] );
		}
	
	} // end TransformPointBSplineUseAddition


	/**
	 * **************** COMPOSITION: T(x) = T1( T0(x) ) *************
	 */

	template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
		void BSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>::
		TransformPointBSplineUseComposition( 
			const InputPointType &inputPoint,
			OutputPointType &outputPoint,
			WeightsType &weights,
			ParameterIndexArrayType &indices, 
			bool &inside ) const
		{
			this->m_CurrentTransformAsBSplineTransform->TransformPoint( 
				this->m_InitialTransform->TransformPoint( inputPoint ),
				outputPoint, weights,	indices, inside);
		} // end TransformPointBSplineUseComposition
			
			
	/**
	 * **************** CURRENT ONLY: T(x) = T1(x) ******************
	 */

	template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
	void BSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>::	
		TransformPointBSplineNoInitialTransform( 
			const InputPointType &inputPoint,
			OutputPointType &outputPoint,
			WeightsType &weights,
			ParameterIndexArrayType &indices, 
			bool &inside ) const
	{
		this->m_CurrentTransformAsBSplineTransform->TransformPoint(
			inputPoint, outputPoint, weights,	indices, inside);
	} // end TransformPointBSplineNoInitialTransform


	/**
	 * ******** NO CURRENT TRANSFORM SET: throw an exception ******************
	 */

	template <typename TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
	void BSplineCombinationTransform<TScalarType, NDimensions, VSplineOrder>::	
		TransformPointBSplineNoCurrentTransform(
			const InputPointType &inputPoint,
			OutputPointType &outputPoint,
			WeightsType &weights,
			ParameterIndexArrayType &indices, 
			bool &inside ) const
	{
		/** throw an exception */
		this->NoCurrentTransformSet(); 
	} // end TransformPointBSplineNoCurrentTransform



} // end namespace itk


#endif // end #ifndef __itkBSplineCombinationTransform_hxx

