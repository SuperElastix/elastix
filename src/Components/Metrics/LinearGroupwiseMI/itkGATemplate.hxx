#ifndef _itkGATemplate_HXX__
#define _itkGATemplate_HXX__

#include "itkGATemplate.h"

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
     * ******************* CalculateIntensity *******************
     */
    
    double
    GATemplate
    ::CalculateIntensity(const std::vector<double> & imageValues, const unsigned int n) const
    {
        double templateIntensity = 1.0;
        for(unsigned int i=0; i < n; i++)
        {
            templateIntensity *= pow((imageValues[i] + this->m_IntensityConstants[i]),static_cast<double>(1.0/static_cast<double>(n)));
        }
        
        return templateIntensity;
        
    }
    
    /**
     * ******************* CalculateIntensity *******************
     */
    
    double
    GATemplate
    ::CalculateIntensity(const std::vector<double> & imageValues, const unsigned int n, const std::vector<unsigned int> & positions) const
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
    ::CalculateRatio(const double & imageValue, const double & templateIntensity, const unsigned int position) const
    {
        return templateIntensity / (imageValue + this->m_IntensityConstants[position]) ;
    }


    
}

#endif // end #ifndef _itkGATemplate_HXX__
