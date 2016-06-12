#ifndef _itkGATemplate_HXX__
#define _itkGATemplate_HXX__

#include "itkGATemplate.h"

#include "math.h"
#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk
{
    /**
     * ********************* Constructor ******************************
     */
    
    GATemplate
    ::GATemplate()
    {

    }
    
    /**
     * ******************* Destructor *******************
     */
    
    GATemplate
    ::~GATemplate()
    {
    }
    
    /**
     * ******************* Initialize *******************
     */
    
    void
    GATemplate
    ::Initialize( void ) throw ( ExceptionObject )
    {
        this->Superclass::Initialize();
    }
    /**
     * ******************* DetermineHistogramMin *******************
     */
    
    double
    GATemplate
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
    GATemplate
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
    GATemplate
    ::CalculateIntensity(const std::vector<double> & imageValues, const unsigned int n, const std::vector<unsigned int> & positions, const unsigned int o) const
    {
        double templateIntensity = 1.0;
        for(unsigned int i=0; i < n; i++)
        {
            templateIntensity *= pow((imageValues[i] + this->m_IntensityConstants[positions[i]]),static_cast<double>(1.0/static_cast<double>(n)));
        }
        
        return templateIntensity;
        
    }
    
    /**
     * ******************* CalculateRatio *******************
     */
    
    double
    GATemplate
    ::CalculateRatio(const double & imageValue, const double & templateIntensity, const unsigned int position, const unsigned int ref, const unsigned int total) const
    {
        if(imageValue + m_IntensityConstants[position] != 0)
        {
            return templateIntensity / ((imageValue + m_IntensityConstants[position])*static_cast<double>(total));
        }
        else
        {
            if ( templateIntensity != 0)
            {
                return (imageValue + m_IntensityConstants[position]) / (static_cast<double>(total) * templateIntensity);
            }
            else
            {
                return 1.0/static_cast<double>(total);
            }
        }
    }
    
}

#endif // end #ifndef _itkGATemplate_HXX__
