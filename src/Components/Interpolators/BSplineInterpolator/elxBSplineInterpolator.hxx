#ifndef __elxBSplineInterpolator_hxx
#define __elxBSplineInterpolator_hxx

#include "elxBSplineInterpolator.h"

namespace elastix
{
using namespace itk;


	/**
	 * ******************* BeforeRegistration ***********************
	 */
		
	template <class TElastix>
		void BSplineInterpolator<TElastix>::
		BeforeRegistration(void)
	{
		/** Set the SplineOrder, default value = 1.*/
		unsigned int splineOrder = 1;
		
		/** Read the desired splineOrder from the parameterFile.*/
		( this->GetConfiguration() )->
			ReadParameter( splineOrder, "BSplineInterpolationOrder", 0 );

		/** Set the splineOrder.*/
		this->SetSplineOrder( splineOrder );
		
	} // end BeforeRegistration


	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void BSplineInterpolator<TElastix>::
		BeforeEachResolution(void)
	{
		/** \todo Make it possible to set the spline order here
		 * May be hard, because it's not possible after setting the 
		 * input image, according to the help.
		 */
	}


} // end namespace elastix

#endif // end #ifndef __elxBSplineInterpolator_hxx

