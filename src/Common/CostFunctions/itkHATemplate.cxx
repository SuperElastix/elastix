#ifndef _itkHATemplate_HXX__
#define _itkHATemplate_HXX__

#include "itkHATemplate.h"

#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk
{
    /**
     * ********************* Constructor ******************************
     */
    
    HATemplate
    ::HATemplate()
    {

    }
    
    /**
     * ******************* Destructor *******************
     */
    
    HATemplate
    ::~HATemplate()
    {
    }
    
    /**
     * ******************* Initialize *******************
     */
    
    void
    HATemplate
    ::Initialize( void ) throw ( ExceptionObject )
    {
        this->Superclass::Initialize();
    }
    
    /**
     * ******************* DetermineHistogramMin *******************
     */
    
    double
    HATemplate
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
    HATemplate
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
    HATemplate
    ::CalculateIntensity(const std::vector<double> & imageValues, const unsigned int n, const std::vector<unsigned int> & positions, const unsigned int o) const
    {
        double templateIntensity =0;
        for(unsigned int i=0; i < n; i++)
        {
            if( imageValues[i]+ this->m_IntensityConstants[i] != 0)
            {
                templateIntensity += static_cast<double>(1.0) / (imageValues[i]+ this->m_IntensityConstants[i]);
            }
            else
            {
                templateIntensity=0;
                return templateIntensity;
            }
        }
        
        templateIntensity = static_cast<double>(n)/ templateIntensity;
        
        return templateIntensity;
        
    }
    
    /**
     * ******************* CalculateRatio *******************
     */
    
    double
    HATemplate
    ::CalculateRatio(const double & imageValue, const double & templateIntensity, const unsigned int position, const unsigned int ref, const unsigned int total) const
    {
        if(imageValue + m_IntensityConstants[position] != 0)
        {
            return templateIntensity * templateIntensity / (imageValue + m_IntensityConstants[position]) / (imageValue + m_IntensityConstants[position]) / static_cast<double>(total);
        }
        else
        {
            if ( templateIntensity != 0)
            {
                return (imageValue + m_IntensityConstants[position]) * (imageValue + m_IntensityConstants[position]) / templateIntensity / templateIntensity / static_cast<double>(total);
            }
            else
            {
                return 1.0/static_cast<double>(total);
            }
        }
    }
    
}

#endif // end #ifndef _itkHATemplate_HXX__
