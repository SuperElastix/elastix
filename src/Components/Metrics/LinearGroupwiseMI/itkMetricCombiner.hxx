#ifndef _itkMetricCombiner_HXX__
#define _itkMetricCombiner_HXX__

#include "itkMetricCombiner.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk
{
    /**
     * ********************* Constructor ******************************
     */
    
    MetricCombiner
    ::MetricCombiner()
    {

    }
    
    /**
     * ******************* Destructor *******************
     */
    
    MetricCombiner
    ::~MetricCombiner()
    {
    }
    
    /**
     * ******************* Initialize *******************
     */
    
    void
    MetricCombiner
    ::Initialize( void ) throw ( ExceptionObject )
    {
    }
    
    
}

#endif // end #ifndef _itkMetricCombiner_HXX__
