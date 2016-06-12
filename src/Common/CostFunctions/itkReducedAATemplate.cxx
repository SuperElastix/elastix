#ifndef _itkReducedAATemplate_HXX__
#define _itkReducedAATemplate_HXX__

#include "itkReducedAATemplate.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk
{
    /**
     * ********************* Constructor ******************************
     */
    
    ReducedAATemplate
    ::ReducedAATemplate()
    {
    }
    
    /**
     * ******************* Destructor *******************
     */
    
    ReducedAATemplate
    ::~ReducedAATemplate()
    {
    }
    
    /**
     * ******************* Initialize *******************
     */
    
    void
    ReducedAATemplate
    ::Initialize( void ) throw ( ExceptionObject )
    {
        this->Superclass::Initialize();
    }
    
    /**
     * ******************* DetermineHistogramMin *******************
     */
    
    double
    ReducedAATemplate
    ::DetermineHistogramMin(const std::vector<double> & imageValues, const unsigned int n) const
    {
        double min = 9999999999;
        for(unsigned int i=0; i < n; i++)
        {
            if (min > imageValues[i])
            {
                min = imageValues[i];
            }
        }
        
        return min;
        
    }
    
    /**
     * ******************* DetermineHistogramMax *******************
     */
    
    double
    ReducedAATemplate
    ::DetermineHistogramMax(const std::vector<double> & imageValues, const unsigned int n) const
    {
        double max = -9999999999;
        for(unsigned int i=0; i < n; i++)
        {
            if (max < imageValues[i])
            {
                max = imageValues[i];
            }
        }
        
        return max;
    }
        
    /**
     * ******************* CalculateIntensity *******************
     */
    
    double
    ReducedAATemplate
    ::CalculateIntensity(const std::vector<double> & imageValues, const unsigned int n, const std::vector<unsigned int> & positions, const unsigned int o) const
    {
        double templateIntensity =0;
        for(unsigned int i=0; i < n; i++)
        {
            if ( i != o)
            {
                templateIntensity += imageValues[i];
            }
        }
        
        return templateIntensity / static_cast<double>(n-1);
        
    }

    /**
     * ******************* CalculateRatio *******************
     */
    
    double
    ReducedAATemplate
    ::CalculateRatio(const double & imageValue, const double & templateIntensity, const unsigned int position, const unsigned int ref, const unsigned int total) const
    {
        return static_cast<double>(position!=ref)/static_cast<double>(total-1);
    }


    
}

#endif // end #ifndef _itkReducedAATemplate_HXX__
