#ifndef __itkDeformationFieldRegulizerForBSpline_HXX__
#define __itkDeformationFieldRegulizerForBSpline_HXX__

#include "itkDeformationFieldRegulizerForBSpline.h"


namespace itk
{
	
	/**
	 * ************************ Constructor	*************************
	 */

	template < class TBSplineTransform >
		DeformationFieldRegulizerForBSpline< TBSplineTransform >
		::DeformationFieldRegulizerForBSpline()
	{

	} // end Constructor
	
	/**
	 * *********************** Destructor ***************************
	 */

	template < class TBSplineTransform >
		DeformationFieldRegulizerForBSpline< TBSplineTransform >
		::~DeformationFieldRegulizerForBSpline()
	{
		//nothing
	} // end Destructor


	/**
   * ****************** TransformPoint ****************************
	 */

	template < class TBSplineTransform >
		typename DeformationFieldRegulizerForBSpline< TBSplineTransform >::OutputPointType
		DeformationFieldRegulizerForBSpline< TBSplineTransform >
		::TransformPoint( const InputPointType &inputPoint ) const
	{
		/** Call the TransformPoint of the BSpline.
		 * DO NOT add
		 * oppDF = this->GetIntermediaryDeformationFieldTransform()->TransformPoint( inputPoint );
		 * return oppBS + oppDF
		 * because in the BSplineDeformableTransform::TransformPoint(1), the
		 * TransformPoint(5) of this class will be called, which also adds the
		 * oppDF. This way it will be added twice!
		 * DO NOT use Superclass::TransformPoint(), for the same reason.
		 */

		/** Return a value. */
		return this->BSplineTransformType::TransformPoint( inputPoint );

	} // end TransformPoint


	/**
   * ****************** TransformPoint ****************************
	 */

	template < class TBSplineTransform >
		void
		DeformationFieldRegulizerForBSpline< TBSplineTransform >
		::TransformPoint( const InputPointType &inputPoint,
			OutputPointType &outputPoint, WeightsType &weights,
			ParameterIndexArrayType &indices, bool &inside ) const
	{ 
		/** Get the outputpoints (opp's):  TP1(ipp) + TP2(ipp) - ipp(!). */
		OutputPointType oppBS, oppDF;
		this->BSplineTransformType::TransformPoint( inputPoint, oppBS, weights, indices, inside );
		oppDF = this->GetIntermediaryDeformationFieldTransform()->TransformPoint( inputPoint );

		for ( unsigned int i = 0; i < OutputSpaceDimension; i++ )
		{
			outputPoint[ i ] = oppBS[ i ] + oppDF[ i ] - inputPoint[ i ];
		}

	} // end TransformPoint


} // end namespace itk


#endif // end #ifndef __itkDeformationFieldRegulizerForBSpline_HXX__

