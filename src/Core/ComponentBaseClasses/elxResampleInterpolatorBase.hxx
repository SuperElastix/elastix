#ifndef __elxResampleInterpolatorBase_hxx
#define __elxResampleInterpolatorBase_hxx

#include "elxResampleInterpolatorBase.h"

namespace elastix
{
	using namespace itk;


	/**
	 * ******************* ReadFromFile *****************************
	 */

	template <class TElastix>
		void ResampleInterpolatorBase<TElastix>
		::ReadFromFile(void)
	{
		// nothing, but must be here

	} // end ReadFromFile


	/**
	 * ******************* WriteToFile ******************************
	 */

	template <class TElastix>
		void ResampleInterpolatorBase<TElastix>
		::WriteToFile(void)
	{
		/** Write ResampleInterpolator specific things. */
		xl::xout["transpar"] << std::endl << "// ResampleInterpolator specific" << std::endl;

		/** Write the name of the resample-interpolator. */
		xl::xout["transpar"] << "(ResampleInterpolator \""
			<< this->elxGetClassName() << "\")" << std::endl;		

	} // end WriteToFile


} // end namespace elastix

#endif // end #ifndef __elxResampleInterpolatorBase_hxx

