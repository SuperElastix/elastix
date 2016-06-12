#ifndef _itkAATemplate_HXX__
#define _itkAATemplate_HXX__

#include "itkAATemplate.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk
{
    /**
     * ********************* Constructor ******************************
     */
    
    AATemplate
    ::AATemplate()
    {
    }
    
    /**
     * ******************* Destructor *******************
     */
    
    AATemplate
    ::~AATemplate()
    {
    }
    
    /**
     * ******************* Initialize *******************
     */
    
    void
    AATemplate
    ::Initialize( void ) throw ( ExceptionObject )
    {
        this->Superclass::Initialize();
    }
    
    /**
     * ******************* DetermineHistogramMin *******************
     */
    
    double
    AATemplate
    ::DetermineHistogramMin(const std::vector<double> & imageValues, const unsigned int n) const
    {;
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
    AATemplate
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
    AATemplate
    ::CalculateIntensity(const std::vector<double> & imageValues, const unsigned int n, const std::vector<unsigned int> & positions, const unsigned int o) const
    {
        double templateIntensity =0;
        for(unsigned int i=0; i < n; i++)
        {
            templateIntensity += imageValues[i];
        }
        
        return templateIntensity / static_cast<double>(n);
        
    }

    /**
     * ******************* CalculateRatio *******************
     */
    
    double
    AATemplate
    ::CalculateRatio(const double & imageValue, const double & templateIntensity, const unsigned int position, const unsigned int ref, const unsigned int total) const
    {
        return 1.0/static_cast<double>(total);
    }


    
}

#endif // end #ifndef _itkAATemplate_HXX__
