#ifndef _itkMetricSum_HXX__
#define _itkMetricSum_HXX__

#include "itkMetricSum.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk
{
    /**
     * ********************* Constructor ******************************
     */
    
    MetricSum
    ::MetricSum()
    {

    }
    
    /**
     * ******************* Destructor *******************
     */
    
    MetricSum
    ::~MetricSum()
    {
    }
    
    /**
     * ******************* Initialize *******************
     */
    
    void
    MetricSum
    ::Initialize( void ) throw ( ExceptionObject )
    {
        this->Superclass::Initialize();
    }
    
    /**
     * ******************* Combine *******************
     */
    
    void
    MetricSum
    ::Combine(double & value, std::vector<double> & values, unsigned int n, const bool minimize) const
    {
        if (minimize)
        {
            value = values[0];
            for(unsigned int i = 1; i < n; i++)
            {
                value += values[i];
            }
        }
        else
        {
            value = -values[0];
            for(unsigned int i = 1; i < n; i++)
            {
                value -= values[i];
            }

        }
        
    }

    
    /**
     * ******************* Combine *******************
     */
    
    void
    MetricSum
    ::Combine(double & value, Array<double> & derivative, std::vector<double> & values, std::vector< Array<double> > & derivatives, unsigned int n, const bool minimize) const
    {
        if (minimize)
        {
            value = values[0];
            derivative = derivatives[0];
            for(unsigned int d = 1; d < n; d++)
            {
                value += values[d];
                derivative += derivatives[d];

            }
        }
        else
        {
            value = -values[0];
            derivative = -derivatives[0];
            for(unsigned int d = 1; d < n; d++)
            {
                value -= values[d];
                derivative -= derivatives[d];

            }

        }
                
    }
    
}

#endif // end #ifndef _itkMetricSum_HXX__
