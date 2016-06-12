#ifndef _itkMetricSquareRootSum_HXX__
#define _itkMetricSquareRootSum_HXX__

#include "itkMetricSquareRootSum.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk
{
    /**
     * ********************* Constructor ******************************
     */
    
    MetricSquareRootSum
    ::MetricSquareRootSum()
    {

    }
    
    /**
     * ******************* Destructor *******************
     */
    
    MetricSquareRootSum
    ::~MetricSquareRootSum()
    {
    }
    
    /**
     * ******************* Initialize *******************
     */
    
    void
    MetricSquareRootSum
    ::Initialize( void ) throw ( ExceptionObject )
    {
        this->Superclass::Initialize();
    }
    
    /**
     * ******************* Combine *******************
     */
    
    void
    MetricSquareRootSum
    ::Combine(double & value, std::vector<double> & values, unsigned int n, const bool minimize) const
    {
        if (minimize)
        {
            value = 0;
            for(unsigned int i = 0; i < n; i++)
            {
                value += std::pow(values[i],0.5);
            }
        }
        else
        {
            value = 0;
            for(unsigned int i = 0; i < n; i++)
            {
                value -= std::pow(values[i],0.5);
            }

        }

    }

    
    /**
     * ******************* Combine *******************
     */
    
    void
    MetricSquareRootSum
    ::Combine(double & value, Array<double> & derivative, std::vector<double> & values, std::vector< Array<double> > & derivatives, unsigned int n, const bool minimize) const
    {
        if (minimize)
        {
            value = 0;
            derivative.Fill(0.0);
            for(unsigned int d = 0; d < n; d++)
            {
                value += std::pow(values[d],0.5);
            }
            for(unsigned int i = 0; i < derivative.GetSize(); i++)
            {
                for(unsigned int d = 0; d < n; d++)
                {
                    derivative[i] += 0.5 / std::pow(values[d],0.5) * derivatives[d][i];
                }
            }
        }
        else
        {
            value = 0;
            derivative.Fill(0.0);
            for(unsigned int d = 0; d < n; d++)
            {
                value -= std::pow(values[d],0.5);
            }
            for(unsigned int i = 0; i < derivative.GetSize(); i++)
            {
                for(unsigned int d = 0; d < n; d++)
                {
                    derivative[i] -= 0.5 / std::pow(values[d],0.5)*derivatives[d][i];
                }
            }

        }
        

    }

}

#endif // end #ifndef _itkMetricSquareRootSum_HXX__
