#ifndef _itkMetricAverage_HXX__
#define _itkMetricAverage_HXX__

#include "itkMetricAverage.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk
{
    /**
     * ********************* Constructor ******************************
     */
    
    MetricAverage
    ::MetricAverage()
    {

    }
    
    /**
     * ******************* Destructor *******************
     */
    
    MetricAverage
    ::~MetricAverage()
    {
    }
    
    /**
     * ******************* Initialize *******************
     */
    
    void
    MetricAverage
    ::Initialize( void ) throw ( ExceptionObject )
    {
        this->Superclass::Initialize();
    }
    
    /**
     * ******************* Combine *******************
     */
    
    void
    MetricAverage
    ::Combine(double & value, std::vector<double> & values, unsigned int n, const bool minimize) const
    {
        if (minimize)
        {
            value = 0;
            for(unsigned int i = 0; i < n; i++)
            {
                value += values[i]/static_cast<double>(n);
            }
        }
        else
        {
            value = 0;
            for(unsigned int i = 0; i < n; i++)
            {
                value -= values[i]/static_cast<double>(n);
            }

        }
        
    }

    
    /**
     * ******************* Combine *******************
     */
    
    void
    MetricAverage
    ::Combine(double & value, Array<double> & derivative, std::vector<double> & values, std::vector< Array<double> > & derivatives, unsigned int n, const bool minimize) const
    {
        if (minimize)
        {
            value = 0;
            derivative.Fill(0.0);
            for(unsigned int d = 0; d < n; d++)
            {
                value += values[d]/static_cast<double>(n);
                for(unsigned int i = 0; i < derivative.GetSize(); i++)
                {
                    derivative[i] += derivatives[d][i]/static_cast<double>(n);
                }
            }
        }
        else
        {
            value = 0;
            derivative.Fill(0.0);
            for(unsigned int d = 0; d < n; d++)
            {
                value -= values[d]/static_cast<double>(n);
                for(unsigned int i = 0; i < derivative.GetSize(); i++)
                {
                    derivative[i] -= derivatives[d][i]/static_cast<double>(n);
                }
            }

        }
        
    }
    
}

#endif // end #ifndef _itkMetricAverage_HXX__
