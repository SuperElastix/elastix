#ifndef __elxFixedRecursivePyramid_hxx
#define __elxFixedRecursivePyramid_hxx

#include "elxFixedRecursivePyramid.h"

namespace elastix
{
	using namespace itk;

	/**
	 * ***************** BeforeRegistration ***********************
	 *

	template <class TElastix>
		void FixedRecursivePyramid<TElastix>
		::BeforeRegistration(void)
	{
		ScheduleType schedule = this->GetSchedule();

		std::cout << "Fixed Schedule:" << std::endl;
		for( unsigned int level = 0; level < m_NumberOfLevels; level++ )
    {
			for( unsigned int dim = 0; dim < ImageDimension; dim++ )
			{
				std::cout << schedule[level][dim] << " ";
			}
			std::cout << std::endl;
		}
	}

	/**
	 * ***************** BeforeEachResolution ***********************
	 *

	template <class TElastix>
		void FixedRecursivePyramid<TElastix>
		::BeforeEachResolution(void)
	{
		ScheduleType schedule = this->GetSchedule();

		std::cout << "Fixed Schedule:" << std::endl;
		for( unsigned int level = 0; level < m_NumberOfLevels; level++ )
    {
			for( unsigned int dim = 0; dim < ImageDimension; dim++ )
			{
				std::cout << schedule[level][dim] << " ";
			}
			std::cout << std::endl;
		}
	}*/

} // end namespace elastix

#endif //#ifndef __elxFixedRecursivePyramid_hxx

