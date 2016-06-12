#ifndef _itkMetricSquaredSum_HXX__
#define _itkMetricSquaredSum_HXX__

#include "itkMetricSquaredSum.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk
{
    /**
     * ********************* Constructor ******************************
     */
    
    MetricSquaredSum
    ::MetricSquaredSum()
    {

    }
    
    /**
     * ******************* Destructor *******************
     */
    
    MetricSquaredSum
    ::~MetricSquaredSum()
    {
    }
    
    /**
     * ******************* Initialize *******************
     */
    
    void
    MetricSquaredSum
    ::Initialize( void ) throw ( ExceptionObject )
    {
        this->Superclass::Initialize();
    }
    
    /**
     * ******************* Combine *******************
     */
    
    void
    MetricSquaredSum
    ::Combine(double & value, std::vector<double> & values, unsigned int n, const bool minimize) const
    {
        if (minimize)
        {
            value = 0;
            for(unsigned int i = 0; i < n; i++)
            {
                value += values[i]*values[i];
            }
        }
        else
        {
            value = 0;
            for(unsigned int i = 0; i < n; i++)
            {
                value -= values[i]*values[i];
            }

        }

    }

    
    /**
     * ******************* Combine *******************
     */
    
    void
    MetricSquaredSum
    ::Combine(double & value, Array<double> & derivative, std::vector<double> & values, std::vector< Array<double> > & derivatives, unsigned int n, const bool minimize) const
    {
        if (minimize)
        {
            value = 0;
            derivative.Fill(0.0);
            for(unsigned int d = 0; d < n; d++)
            {
                value += values[d]*values[d];
            }
            for(unsigned int i = 0; i < derivative.GetSize(); i++)
            {
                for(unsigned int d = 0; d < n; d++)
                {
                    derivative[i] += 2*values[d]*derivatives[d][i];
                }
            }
        }
        else
        {
            value = 0;
            derivative.Fill(0.0);
            for(unsigned int d = 0; d < n; d++)
            {
                value -= values[d]*values[d];
            }
            for(unsigned int i = 0; i < derivative.GetSize(); i++)
            {
                for(unsigned int d = 0; d < n; d++)
                {
                    derivative[i] -= 2*values[d]*derivatives[d][i];
                }
            }

        }
        

    }

}

#endif // end #ifndef _itkMetricSquaredSum_HXX__
