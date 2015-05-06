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
     * ******************* CalculateIntensity *******************
     */
    
    double
    HATemplate
    ::CalculateIntensity(const std::vector<double> & imageValues, const unsigned int n) const
    {
        double templateIntensity =0;
        for(unsigned int i=0; i < n; i++)
        {
            templateIntensity += static_cast<double>(1.0) / (imageValues[i]+ m_IntensityConstants[i]);
        }
        
        templateIntensity = static_cast<double>(n)/ templateIntensity;
        
        return templateIntensity;
        
    }
    
    /**
     * ******************* CalculateIntensity *******************
     */
    
    double
    HATemplate
    ::CalculateIntensity(const std::vector<double> & imageValues, const unsigned int n, const std::vector<unsigned int> & positions) const
    {
        double templateIntensity =0;
        for(unsigned int i=0; i < n; i++)
        {
            templateIntensity += static_cast<double>(1.0) / (imageValues[i]+ this->m_IntensityConstants[positions[i]]);
        }
        
        templateIntensity = static_cast<double>(n)/ templateIntensity;
        
        return templateIntensity;
        
    }
    
    /**
     * ******************* CalculateRatio *******************
     */
    
    double
    HATemplate
    ::CalculateRatio(const double & imageValue, const double & templateIntensity, const unsigned int position) const
    {
        return templateIntensity * templateIntensity / (imageValue + m_IntensityConstants[position]) / (imageValue + m_IntensityConstants[position]);
    }
    
}

#endif // end #ifndef _itkHATemplate_HXX__
