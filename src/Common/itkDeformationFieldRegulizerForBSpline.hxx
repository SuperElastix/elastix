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
		void
		DeformationFieldRegulizerForBSpline< TBSplineTransform >
		::TransformPoint( const InputPointType &inputPoint,
			OutputPointType &outputPoint, WeightsType &weights,
			ParameterIndexArrayType &indices, bool &inside ) const
	{ 
		//elxout << "testin, empty function yet ... " << std::endl;
		/** Get the opp's ?????? TP1(ipp) + TP2(ipp)??
		 * of TP2(TP1(ipp)) ?????
		 */
		this->BSplineTransformType::TransformPoint( inputPoint, outputPoint, weights, indices, inside );
		OutputPointType oppDF = this->GetIntermediaryDeformationFieldTransform()->TransformPoint( inputPoint );

		for ( unsigned int i = 0; i < OutputSpaceDimension; i++ )
		{
			outputPoint[ i ] = outputPoint[ i ] + oppDF[ i ] - inputPoint[ i ];
		}

	} // end TransformPoint


} // end namespace itk


#endif // end #ifndef __itkDeformationFieldRegulizerForBSpline_HXX__

