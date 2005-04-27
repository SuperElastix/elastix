#ifndef __elxMetricBase_hxx
#define __elxMetricBase_hxx

#include "elxMetricBase.h"


namespace elastix
{
	using namespace itk;

	template <class TElastix>
  void MetricBase<TElastix>::SelectNewSamples(void)
	{
		/**
		 * Force the metric to base its computation on a new subset of image samples.
		 * Not every metric may have implemented this, so invoke an exception if this
		 * method is called, without being overrided by a subclass.
		 */

		xl::xout["error"] << "ERROR: The SelectNewSamples function should be overridden or just not used." << std::endl;
		itkExceptionMacro(<< "ERROR: The SelectNewSamples method is not implemented in your metric.");

	} // end SelectNewSamples

} // end namespace elastix


#endif // end #ifndef __elxMetricBase_hxx

