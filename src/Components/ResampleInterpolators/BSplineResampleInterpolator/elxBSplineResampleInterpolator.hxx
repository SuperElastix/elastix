#ifndef __elxBSplineResampleInterpolator_hxx
#define __elxBSplineResampleInterpolator_hxx

#include "elxBSplineResampleInterpolator.h"

namespace elastix
{
	using namespace itk;


	/*
	 * ******************* BeforeRegistration ***********************
	 */

	template <class TElastix>
	void BSplineResampleInterpolator<TElastix>
	::BeforeRegistration(void)
	{
		/** BSplineResampleInterpolator specific.*/

		/** Set the SplineOrder, default = 3.*/
		unsigned int splineOrder = 3;  

		/** Read the desired splineOrder from the parameterFile.*/
		m_Configuration->ReadParameter( splineOrder, "FinalBSplineInterpolationOrder", 0 );

		/** Set the splineOrder in the superclass.*/
		this->SetSplineOrder( splineOrder );

	} // end BeforeRegistration


	/*
	 * ******************* ReadFromFile  ****************************
	 */

	template <class TElastix>
	void BSplineResampleInterpolator<TElastix>
	::ReadFromFile(void)
	{
		/** Call ReadFromFile of the ResamplerBase.*/
		this->Superclass2::ReadFromFile();

		/** BSplineResampleInterpolator specific.*/

		/** Set the SplineOrder, default = 3.*/
		unsigned int splineOrder = 3;  

		/** Read the desired splineOrder from the parameterFile.*/
		m_Configuration->ReadParameter( splineOrder, "FinalBSplineInterpolationOrder", 0 );

		/** Set the splineOrder in the superclass.*/
		this->SetSplineOrder( splineOrder );

	} // end ReadFromFile
	

	/**
	 * ******************* WriteToFile ******************************
	 */

	template <class TElastix>
		void BSplineResampleInterpolator<TElastix>
		::WriteToFile(void)
	{
		/** Call WriteTiFile of the ResamplerBase.*/
		this->Superclass2::WriteToFile();

		/** The BSplineResampleInterpolator adds:*/

		/** Write the FinalBSplineInterpolationOrder.*/
		xout["transpar"] << "(FinalBSplineInterpolationOrder " <<
			this->GetSplineOrder() << ")" << std::endl;

	} // end WriteToFile

} // end namespace elastix

#endif // end #ifndef __elxBSplineResampleInterpolator_hxx

