#ifndef _itkMedianTemplate_HXX__
#define _itkMedianTemplate_HXX__

#include "itkMedianTemplate.h"
#include <algorithm>
#ifdef ELASTIX_USE_OPENMP
#include <omp.h>
#endif

namespace itk
{
    /**
     * ********************* Constructor ******************************
     */
    
    MedianTemplate
    ::MedianTemplate()
    {
    }
    
    /**
     * ******************* Destructor *******************
     */
    
    MedianTemplate
    ::~MedianTemplate()
    {
    }
    
    /**
     * ******************* Initialize *******************
     */
    
    void
    MedianTemplate
    ::Initialize( void ) throw ( ExceptionObject )
    {
        this->Superclass::Initialize();
    }
    
    /**
     * ******************* DetermineHistogramMin *******************
     */
    
    double
    MedianTemplate
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
    MedianTemplate
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
    MedianTemplate
    ::CalculateIntensity(const std::vector<double> & imageValues, const unsigned int n, const std::vector<unsigned int> & positions, const unsigned int o) const
    {
        double median;
        std::vector<double> sorted = imageValues;
        size_t size = sorted.size();
        std::sort(sorted.begin(), sorted.end());
        
        if (size  % 2 == 0)
        {
            median = (sorted[size / 2 - 1] + sorted[size / 2]) / 2;
        }
        else
        {
            median = sorted[size / 2];
        }
        
        return median;
        
    }

    /**
     * ******************* CalculateRatio *******************
     */
    
    double
    MedianTemplate
    ::CalculateRatio(const double & imageValue, const double & templateIntensity, const unsigned int position, const unsigned int ref, const unsigned int total) const
    {
        return 1.0/static_cast<double>(total);
    }


    
}

#endif // end #ifndef _itkMedianTemplate_HXX__
